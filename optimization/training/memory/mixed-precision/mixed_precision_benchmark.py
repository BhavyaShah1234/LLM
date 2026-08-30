"""Mixed Precision -- isolated benchmark script (training, memory).

Compares fp32, fp16, and bf16 on real forward+backward passes, measuring
wall-clock step time and peak GPU memory. See root README's Quantization &
PEFT theory section for the fp16-vs-bf16 tradeoff (fp16 has more mantissa
precision but a narrower exponent range -- prone to overflow/NaN on
from-scratch training with large activations; bf16 matches fp32's exponent
range at the cost of mantissa precision, which is why every training
script in this project defaults to bf16).

**Real finding that shaped this script's default `--model`**: the first
version of this benchmark defaulted to `Qwen/Qwen3-1.7B-Base` (like the
flash-attention benchmark) with LoRA as memory scaffolding. fp32 OOM'd
anyway, even at `--batch_size 1 --seq_len 128` -- fp32 weights alone for a
1.7B model are ~6.8GB, leaving no headroom on this 8GB card regardless of
LoRA (LoRA freezes the base weights but doesn't change their dtype/size).
This is consistent with, not a violation of, this project's model-selection
philosophy (root README): it targets fp16/bf16 fitting in 8GB, not fp32.
Fixed by defaulting to this project's own from-scratch CLM checkpoint
(`./output/pretraining/clm`, 51M params, `pretraining/clm.py`) instead --
small enough that full-parameter fine-tuning fits comfortably in all three
dtypes, which is also why LoRA scaffolding is dropped entirely here (it
isn't needed at this scale, and GPT2-style `Conv1D` attention layers don't
even match `common/peft_setup.py`'s default LoRA target module names,
which target `q_proj`/`k_proj`/`v_proj`/`o_proj` -- the same naming
mismatch documented in `../quantization-aware-training/qat.py`).

Synthetic random-token batches are used instead of a real dataset -- same
rationale as the flash-attention benchmark: this measures compute/memory
characteristics of a numeric format, not model quality.

Usage:
    python mixed_precision_benchmark.py --debug_first_batch
    python mixed_precision_benchmark.py --batch_size 8 --seq_len 256 --num_steps 10
"""

import argparse
import time

import torch

from common.logging_utils import print_banner, print_config
from common.run_results import write_run_result
from common.seeding import set_all_seeds

ARCHITECTURE = "decoder-only"
DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for this benchmark.

    Returns:
        argparse.ArgumentParser: Parser with all benchmark options (model,
        synthetic-batch shape, step counts, seed, output dir, debug flag)
        registered.
    """
    p = argparse.ArgumentParser(description="Benchmark fp32 vs fp16 vs bf16 on real forward+backward passes.")
    p.add_argument("--model", type=str, default="./output/pretraining/clm", help="Model to benchmark (local path or HF Hub id). Default: this project's own from-scratch CLM checkpoint (51M params) -- fp32 of a larger model like Qwen3-1.7B-Base doesn't fit on an 8GB card at all, see module docstring.")
    p.add_argument("--batch_size", type=int, default=8, help="Synthetic batch size. Default: 8.")
    p.add_argument("--seq_len", type=int, default=256, help="Synthetic sequence length. Default: 256.")
    p.add_argument("--num_steps", type=int, default=10, help="Timed forward+backward steps per dtype (after warmup). Default: 10.")
    p.add_argument("--num_warmup", type=int, default=2, help="Untimed warmup steps. Default: 2.")
    p.add_argument("--seed", type=int, default=42, help="Random seed. Default: 42.")
    p.add_argument("--output_dir", type=str, default="./output/optimization/mixed_precision_benchmark", help="Where to write run_result.json.")
    p.add_argument("--debug_first_batch", action="store_true", default=False, help="Build one synthetic batch and run a single step for each dtype, print loss, then exit.")
    return p


def build_synthetic_batch(vocab_size: int, batch_size: int, seq_len: int, device: str):
    """Build a random-token CLM batch used to drive all three dtype configurations.

    Args:
        vocab_size (int): Vocabulary size to sample token ids from.
        batch_size (int): Number of sequences in the batch.
        seq_len (int): Length of each sequence.
        device (str): Torch device to place the tensors on.

    Returns:
        dict: `{"input_ids", "attention_mask", "labels"}` tensors, with
        `labels` a clone of `input_ids` (standard CLM setup).
    """
    input_ids = torch.randint(low=0, high=vocab_size, size=(batch_size, seq_len), device=device)
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def load_model_with_dtype(model_name: str, dtype: torch.dtype, device: str):
    """Load a causal LM in train mode at a given dtype.

    Args:
        model_name (str): Local path or HF Hub id of the checkpoint to load.
        dtype (torch.dtype): Dtype to load model weights in (fp32, fp16, or bf16).
        device (str): Torch device to move the model to.

    Returns:
        transformers.PreTrainedModel: The loaded model in `.train()` mode.
    """
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype, trust_remote_code=True).to(device)
    model.train()
    return model


def run_benchmark(model, batch, num_steps: int, num_warmup: int, device: str):
    """Time forward+backward steps for one dtype, tracking whether the loss ever goes NaN.

    Args:
        model (transformers.PreTrainedModel): Model to benchmark, already in train mode on `device`.
        batch (dict): Batch from `build_synthetic_batch`, already on `device`.
        num_steps (int): Timed forward+backward steps to run after warmup.
        num_warmup (int): Untimed warmup steps run before timing starts.
        device (str): Torch device the model and batch live on.

    Returns:
        tuple[float, float, float, bool]: `(avg_step_seconds, peak_memory_mb,
        final_loss, nan_seen)` -- mean wall-clock time per timed step, peak
        CUDA memory allocated during the timed steps in MB, the loss from
        the last timed step, and whether any timed step's loss was NaN
        (relevant mainly for fp16, which can overflow).
    """
    nan_seen = False
    for _ in range(num_warmup):
        model.zero_grad(set_to_none=True)
        outputs = model(**batch)
        outputs.loss.backward()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(device)

    start = time.time()
    final_loss = None
    for _ in range(num_steps):
        model.zero_grad(set_to_none=True)
        outputs = model(**batch)
        final_loss = outputs.loss.item()
        if final_loss != final_loss:  # NaN check without importing math
            nan_seen = True
        outputs.loss.backward()
    torch.cuda.synchronize()
    elapsed = time.time() - start

    peak_memory_mb = torch.cuda.max_memory_allocated(device) / 1e6
    avg_step_seconds = elapsed / num_steps
    return avg_step_seconds, peak_memory_mb, final_loss, nan_seen


def main():
    """Run the fp32/fp16/bf16 benchmark and write a `run_result.json`.

    Parses CLI args, builds a synthetic CLM batch, benchmarks each dtype
    in `DTYPES` (or just one debug step of each if `--debug_first_batch`
    is set), prints a summary comparing fp16/bf16 against the fp32
    baseline, and records the comparison via
    `common.run_results.write_run_result`.
    """
    args = build_arg_parser().parse_args()
    set_all_seeds(args.seed)
    print_config(args, "Mixed precision (fp32 vs fp16 vs bf16) benchmark")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: no CUDA device found -- fp16 in particular is not well-optimized on CPU, and these timings/memory numbers assume a GPU.")

    from transformers import AutoConfig

    vocab_size = AutoConfig.from_pretrained(args.model, trust_remote_code=True).vocab_size
    batch = build_synthetic_batch(vocab_size, args.batch_size, args.seq_len, device)
    print(f"Synthetic batch: batch_size={args.batch_size}, seq_len={args.seq_len}, vocab_size={vocab_size}")

    if args.debug_first_batch:
        for dtype_name, dtype in DTYPES.items():
            print_banner(f"DEBUG: dtype={dtype_name}")
            model = load_model_with_dtype(args.model, dtype, device)
            model.zero_grad(set_to_none=True)
            outputs = model(**batch)
            outputs.loss.backward()
            print(f"loss={outputs.loss.item():.4f}  logits.dtype={outputs.logits.dtype}")
            del model
            torch.cuda.empty_cache()
        print("--debug_first_batch set: exiting without the full benchmark.")
        return

    results = {}
    for dtype_name, dtype in DTYPES.items():
        print_banner(f"BENCHMARKING dtype={dtype_name}")
        model = load_model_with_dtype(args.model, dtype, device)
        avg_step_seconds, peak_memory_mb, final_loss, nan_seen = run_benchmark(model, batch, args.num_steps, args.num_warmup, device)
        results[dtype_name] = {"avg_step_seconds": avg_step_seconds, "peak_memory_mb": peak_memory_mb, "final_loss": final_loss, "nan_seen": nan_seen}
        print(f"avg_step_seconds={avg_step_seconds:.4f}  peak_memory_mb={peak_memory_mb:.1f}  final_loss={final_loss}  nan_seen={nan_seen}")
        del model
        torch.cuda.empty_cache()

    print_banner("SUMMARY")
    fp32 = results["fp32"]
    for name in ("fp16", "bf16"):
        r = results[name]
        speedup = fp32["avg_step_seconds"] / r["avg_step_seconds"]
        memory_reduction = 1 - (r["peak_memory_mb"] / fp32["peak_memory_mb"])
        print(f"{name} vs fp32: {speedup:.2f}x speedup, {100 * memory_reduction:.1f}% peak memory reduction, nan_seen={r['nan_seen']}")

    metrics = {}
    for name, r in results.items():
        for k, v in r.items():
            metrics[f"{name}_{k}"] = v
    metrics["fp16_speedup_vs_fp32"] = fp32["avg_step_seconds"] / results["fp16"]["avg_step_seconds"]
    metrics["bf16_speedup_vs_fp32"] = fp32["avg_step_seconds"] / results["bf16"]["avg_step_seconds"]
    metrics["fp16_memory_reduction_frac_vs_fp32"] = 1 - (results["fp16"]["peak_memory_mb"] / fp32["peak_memory_mb"])
    metrics["bf16_memory_reduction_frac_vs_fp32"] = 1 - (results["bf16"]["peak_memory_mb"] / fp32["peak_memory_mb"])

    write_run_result(
        output_dir=args.output_dir,
        stage="optimization",
        task="mixed_precision_benchmark",
        modality="text",
        architecture=ARCHITECTURE,
        model_name=args.model,
        dataset_name="synthetic random tokens (no dataset -- compute/memory benchmark, see module docstring)",
        hyperparameters=vars(args),
        metrics=metrics,
        num_train_samples=0,
        num_eval_samples=0,
        train_runtime_seconds=sum(r["avg_step_seconds"] * args.num_steps for r in results.values()),
    )
    print("Done.")


if __name__ == "__main__":
    main()
