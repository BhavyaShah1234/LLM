"""FlashAttention -- isolated benchmark script (training, execution-time).

Compares HF's `attn_implementation="eager"` (naive O(seq_len^2) attention,
materializes the full attention matrix) against `attn_implementation="sdpa"`
(PyTorch's `scaled_dot_product_attention`, which dispatches to a fused
flash-attention-family CUDA kernel) on real forward+backward passes,
measuring wall-clock step time and peak GPU memory.

**A real thing worth being explicit about, found before writing this
script**: there are two different things called "flash attention" here.
The standalone `flash_attn` pip package (what `attn_implementation=
"flash_attention_2"` would use) is NOT installed in this project's venv --
confirmed via `import flash_attn` failing, and it has no prebuilt wheel for
any Python version (source-only, see root README's environment section),
so installing it means compiling from source. This script instead compares
against **PyTorch's own built-in fused SDPA kernel**
(`torch.backends.cuda.flash_sdp_enabled()` confirmed `True` on this
machine), which implements the same flash-attention algorithm (tiled,
IO-aware, no full attention-matrix materialization) without the separate
package. `attn_implementation="sdpa"` is also HF `transformers`' own
default for most models today, so this comparison -- eager vs. this
project's actual default -- is the practically relevant one, not a
downgrade from testing the "real" flash-attn package.

LoRA is applied internally (not a CLI-exposed choice) purely as memory
scaffolding to keep a 1.7B model's full backward pass comfortably under
8GB regardless of which attention implementation is active -- LoRA doesn't
touch the attention computation path itself, so it doesn't confound the
comparison. LoRA itself as a technique is what
`../../memory/lora-qlora/lora_qlora_benchmark.py` studies.

Synthetic random-token batches are used instead of a real dataset: this
benchmark measures compute/memory characteristics of an attention
implementation, not model quality, so real text isn't needed -- unlike
../../../inference/memory/post-training-quantization/ptq.py and
../quantization-aware-training/qat.py, which measure ACCURACY and
therefore do need real held-out data.

Usage:
    python flash_attention_benchmark.py --debug_first_batch
    python flash_attention_benchmark.py --batch_size 2 --seq_len 256 --num_steps 10
"""

import argparse
import time

import torch

from common.logging_utils import print_banner, print_config
from common.peft_setup import apply_lora, build_lora_config
from common.run_results import write_run_result
from common.seeding import set_all_seeds

ARCHITECTURE = "decoder-only"
ATTN_IMPLEMENTATIONS = ["eager", "sdpa"]


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for the FlashAttention (eager vs. SDPA) benchmark.

    Returns:
        argparse.ArgumentParser: Parser covering model choice, synthetic batch
        shape, step/warmup counts, seed, and output path.
    """
    p = argparse.ArgumentParser(description="Benchmark eager vs. SDPA (fused/flash-attention-family) attention on real forward+backward passes.")
    p.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B-Base", help="Model to benchmark (local path or HF Hub id). Default: this project's usual base-model default.")
    p.add_argument("--batch_size", type=int, default=2, help="Synthetic batch size. Default: 2.")
    p.add_argument("--seq_len", type=int, default=256, help="Synthetic sequence length. Default: 256.")
    p.add_argument("--num_steps", type=int, default=10, help="Timed forward+backward steps per attention implementation (after warmup). Default: 10.")
    p.add_argument("--num_warmup", type=int, default=2, help="Untimed warmup steps (excludes CUDA/cuDNN kernel-selection and autograd-graph first-call overhead from the timed average). Default: 2.")
    p.add_argument("--seed", type=int, default=42, help="Random seed. Default: 42.")
    p.add_argument("--output_dir", type=str, default="./output/optimization/flash_attention_benchmark", help="Where to write run_result.json.")
    p.add_argument("--debug_first_batch", action="store_true", default=False, help="Build one synthetic batch and run a single step for each implementation, print shapes/loss, then exit.")
    return p


def build_synthetic_batch(vocab_size: int, batch_size: int, seq_len: int, device: str):
    """Build a fixed-shape batch of random token ids for a compute/memory-only benchmark.

    Args:
        vocab_size (int): Upper bound (exclusive) for sampled token ids.
        batch_size (int): Number of sequences in the batch.
        seq_len (int): Length of each sequence.
        device (str): Device to place the tensors on.

    Returns:
        dict: `{"input_ids", "attention_mask", "labels"}`, with `labels` equal to
        `input_ids` (self-supervised LM loss) and a full (unpadded) attention mask.
    """
    input_ids = torch.randint(low=0, high=vocab_size, size=(batch_size, seq_len), device=device)
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def load_model_with_attn_impl(model_name: str, attn_implementation: str, device: str):
    """Load a causal LM with a specific attention backend and wrap it in LoRA.

    LoRA is applied purely as memory scaffolding (see module docstring) and does
    not touch the attention computation path, so it doesn't confound the comparison.

    Args:
        model_name (str): Model to load (local path or HF Hub id).
        attn_implementation (str): One of `"eager"` or `"sdpa"`.
        device (str): Device to move the model to.

    Returns:
        The LoRA-wrapped model in train mode.
    """
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16, attn_implementation=attn_implementation, trust_remote_code=True
    ).to(device)
    lora_config = build_lora_config()
    model = apply_lora(model, lora_config, prepare_for_kbit=False, print_trainable=False)
    model.train()
    return model


def run_benchmark(model, batch, num_steps: int, num_warmup: int, device: str):
    """Run warmup then timed forward+backward steps on a fixed batch, tracking peak memory.

    Args:
        model: Model to benchmark, already in train mode.
        batch (dict): Fixed batch reused for every step (from `build_synthetic_batch`).
        num_steps (int): Number of timed steps.
        num_warmup (int): Number of untimed warmup steps run first.
        device (str): Device whose peak memory stats are tracked/reset.

    Returns:
        tuple[float, float, float]: `(avg_step_seconds, peak_memory_mb, final_loss)`.
    """
    for _ in range(num_warmup):
        model.zero_grad(set_to_none=True)
        outputs = model(**batch)
        outputs.loss.backward()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(device)

    start = time.time()
    for _ in range(num_steps):
        model.zero_grad(set_to_none=True)
        outputs = model(**batch)
        outputs.loss.backward()
    torch.cuda.synchronize()
    elapsed = time.time() - start

    peak_memory_mb = torch.cuda.max_memory_allocated(device) / 1e6
    avg_step_seconds = elapsed / num_steps
    return avg_step_seconds, peak_memory_mb, outputs.loss.item()


def main():
    """Run the end-to-end FlashAttention benchmark: build a synthetic batch, run
    forward+backward under both `eager` and `sdpa` attention, compare step time and
    peak memory, and record results via `write_run_result` (or exit early with
    `--debug_first_batch`).
    """
    args = build_arg_parser().parse_args()
    set_all_seeds(args.seed)
    print_config(args, "FlashAttention (eager vs. SDPA) benchmark")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: no CUDA device found -- this benchmark measures GPU attention kernels and its timings/memory numbers are not meaningful on CPU.")

    from transformers import AutoConfig

    vocab_size = AutoConfig.from_pretrained(args.model, trust_remote_code=True).vocab_size
    batch = build_synthetic_batch(vocab_size, args.batch_size, args.seq_len, device)
    print(f"Synthetic batch: batch_size={args.batch_size}, seq_len={args.seq_len}, vocab_size={vocab_size}")

    if args.debug_first_batch:
        for attn_impl in ATTN_IMPLEMENTATIONS:
            print_banner(f"DEBUG: attn_implementation={attn_impl!r}")
            model = load_model_with_attn_impl(args.model, attn_impl, device)
            model.zero_grad(set_to_none=True)
            outputs = model(**batch)
            outputs.loss.backward()
            print(f"loss={outputs.loss.item():.4f}  logits.shape={tuple(outputs.logits.shape)}")
            del model
            torch.cuda.empty_cache()
        print("--debug_first_batch set: exiting without the full benchmark.")
        return

    results = {}
    for attn_impl in ATTN_IMPLEMENTATIONS:
        print_banner(f"BENCHMARKING attn_implementation={attn_impl!r}")
        model = load_model_with_attn_impl(args.model, attn_impl, device)
        avg_step_seconds, peak_memory_mb, final_loss = run_benchmark(model, batch, args.num_steps, args.num_warmup, device)
        results[attn_impl] = {"avg_step_seconds": avg_step_seconds, "peak_memory_mb": peak_memory_mb, "final_loss": final_loss}
        print(f"avg_step_seconds={avg_step_seconds:.4f}  peak_memory_mb={peak_memory_mb:.1f}  final_loss={final_loss:.4f}")
        del model
        torch.cuda.empty_cache()

    print_banner("SUMMARY")
    eager = results["eager"]
    sdpa = results["sdpa"]
    speedup = eager["avg_step_seconds"] / sdpa["avg_step_seconds"]
    memory_reduction = 1 - (sdpa["peak_memory_mb"] / eager["peak_memory_mb"])
    print(f"SDPA vs eager speedup: {speedup:.2f}x")
    print(f"SDPA vs eager peak memory reduction: {100 * memory_reduction:.1f}%")

    write_run_result(
        output_dir=args.output_dir,
        stage="optimization",
        task="flash_attention_benchmark",
        modality="text",
        architecture=ARCHITECTURE,
        model_name=args.model,
        dataset_name="synthetic random tokens (no dataset -- compute/memory benchmark, see module docstring)",
        hyperparameters=vars(args),
        metrics={
            "eager_avg_step_seconds": eager["avg_step_seconds"],
            "eager_peak_memory_mb": eager["peak_memory_mb"],
            "sdpa_avg_step_seconds": sdpa["avg_step_seconds"],
            "sdpa_peak_memory_mb": sdpa["peak_memory_mb"],
            "sdpa_speedup": speedup,
            "sdpa_memory_reduction_frac": memory_reduction,
        },
        num_train_samples=0,
        num_eval_samples=0,
        train_runtime_seconds=(eager["avg_step_seconds"] + sdpa["avg_step_seconds"]) * args.num_steps,
    )
    print("Done.")


if __name__ == "__main__":
    main()
