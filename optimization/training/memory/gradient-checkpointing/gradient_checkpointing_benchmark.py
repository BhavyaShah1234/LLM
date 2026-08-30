"""Gradient Checkpointing -- isolated benchmark script (training, memory).

Compares training with vs. without gradient checkpointing on real
forward+backward passes, measuring wall-clock step time and peak GPU
memory. Gradient checkpointing trades compute for memory: instead of
storing every layer's activations for the backward pass, it stores only a
subset (checkpoints) and recomputes the rest during backward -- lower peak
memory, more forward-pass compute (roughly 1 extra forward pass worth),
hence slower per step. This is the opposite tradeoff direction from
`../mixed-precision/mixed_precision_benchmark.py` (bf16/fp16 there
reduced BOTH memory and time) and from
`../../execution-time/flash-attention/flash_attention_benchmark.py`
(SDPA there also reduced both) -- gradient checkpointing is a case where
you deliberately pay a time cost for a memory win, useful specifically
when memory (not speed) is the binding constraint, e.g. to fit a larger
batch size or longer sequence than would otherwise be possible.

Uses this project's own from-scratch CLM checkpoint
(`./output/pretraining/clm`, 51M params) as the default model, same
reasoning as `../mixed-precision/mixed_precision_benchmark.py`: keeps the
comparison simple and avoids any LoRA/dtype scaffolding, since gradient
checkpointing's memory-vs-time tradeoff is visible even on a small model
(the technique reduces *activation* memory specifically, which scales with
batch_size x seq_len x num_layers regardless of parameter count).

Synthetic random-token batches are used instead of a real dataset -- same
rationale as the other optimization/ compute benchmarks in this wave.

Usage:
    python gradient_checkpointing_benchmark.py --debug_first_batch
    python gradient_checkpointing_benchmark.py --batch_size 16 --seq_len 256 --num_steps 10
"""

import argparse
import time

import torch

from common.logging_utils import print_banner, print_config
from common.run_results import write_run_result
from common.seeding import set_all_seeds

ARCHITECTURE = "decoder-only"


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for this benchmark.

    Returns:
        argparse.ArgumentParser: Parser with all benchmark options (model,
        synthetic-batch shape, step counts, mixed precision, seed, output
        dir, debug flag) registered.
    """
    p = argparse.ArgumentParser(description="Benchmark training with vs. without gradient checkpointing.")
    p.add_argument("--model", type=str, default="./output/pretraining/clm", help="Model to benchmark (local path or HF Hub id). Default: this project's own from-scratch CLM checkpoint.")
    p.add_argument("--batch_size", type=int, default=16, help="Synthetic batch size. Default: 16.")
    p.add_argument("--seq_len", type=int, default=256, help="Synthetic sequence length. Default: 256.")
    p.add_argument("--num_steps", type=int, default=10, help="Timed forward+backward steps per configuration (after warmup). Default: 10.")
    p.add_argument("--num_warmup", type=int, default=2, help="Untimed warmup steps. Default: 2.")
    p.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"], help="Mixed precision (held fixed across both arms -- this benchmark isolates gradient checkpointing specifically, not precision, see ../mixed-precision/ for that). Default: bf16.")
    p.add_argument("--seed", type=int, default=42, help="Random seed. Default: 42.")
    p.add_argument("--output_dir", type=str, default="./output/optimization/gradient_checkpointing_benchmark", help="Where to write run_result.json.")
    p.add_argument("--debug_first_batch", action="store_true", default=False, help="Build one synthetic batch and run a single step for each configuration, print loss, then exit.")
    return p


def build_synthetic_batch(vocab_size: int, batch_size: int, seq_len: int, device: str):
    """Build a random-token CLM batch used to drive both benchmark configurations.

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


def load_model(model_name: str, dtype: torch.dtype, gradient_checkpointing: bool, device: str):
    """Load a causal LM in train mode, optionally with gradient checkpointing enabled.

    Args:
        model_name (str): Local path or HF Hub id of the checkpoint to load.
        dtype (torch.dtype): Dtype to load model weights in.
        gradient_checkpointing (bool): If True, enable gradient checkpointing and disable the KV cache (required, since cached states are incompatible with backward-pass recomputation).
        device (str): Torch device to move the model to.

    Returns:
        transformers.PreTrainedModel: The loaded model in `.train()` mode.
    """
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype, trust_remote_code=True).to(device)
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False  # required: cached KV states are incompatible with recomputation during backward
    model.train()
    return model


def run_benchmark(model, batch, num_steps: int, num_warmup: int, device: str):
    """Time forward+backward steps for one model configuration.

    Args:
        model (transformers.PreTrainedModel): Model to benchmark, already in train mode on `device`.
        batch (dict): Batch from `build_synthetic_batch`, already on `device`.
        num_steps (int): Timed forward+backward steps to run after warmup.
        num_warmup (int): Untimed warmup steps run before timing starts.
        device (str): Torch device the model and batch live on.

    Returns:
        tuple[float, float, float]: `(avg_step_seconds, peak_memory_mb, final_loss)`
        -- mean wall-clock time per timed step, peak CUDA memory allocated
        during the timed steps in MB, and the loss from the last timed step.
    """
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
        outputs.loss.backward()
    torch.cuda.synchronize()
    elapsed = time.time() - start

    peak_memory_mb = torch.cuda.max_memory_allocated(device) / 1e6
    avg_step_seconds = elapsed / num_steps
    return avg_step_seconds, peak_memory_mb, final_loss


def main():
    """Run the with/without gradient-checkpointing benchmark and write a `run_result.json`.

    Parses CLI args, builds a synthetic CLM batch, benchmarks both
    configurations (or just one debug step of each if
    `--debug_first_batch` is set), prints a summary, and records the
    comparison via `common.run_results.write_run_result`.
    """
    args = build_arg_parser().parse_args()
    set_all_seeds(args.seed)
    print_config(args, "Gradient checkpointing benchmark")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = {"no": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.mixed_precision]

    from transformers import AutoConfig

    vocab_size = AutoConfig.from_pretrained(args.model, trust_remote_code=True).vocab_size
    batch = build_synthetic_batch(vocab_size, args.batch_size, args.seq_len, device)
    print(f"Synthetic batch: batch_size={args.batch_size}, seq_len={args.seq_len}, vocab_size={vocab_size}, dtype={args.mixed_precision}")

    if args.debug_first_batch:
        for gc in (False, True):
            print_banner(f"DEBUG: gradient_checkpointing={gc}")
            model = load_model(args.model, dtype, gc, device)
            model.zero_grad(set_to_none=True)
            outputs = model(**batch)
            outputs.loss.backward()
            print(f"loss={outputs.loss.item():.4f}")
            del model
            torch.cuda.empty_cache()
        print("--debug_first_batch set: exiting without the full benchmark.")
        return

    results = {}
    for gc in (False, True):
        label = "with_checkpointing" if gc else "without_checkpointing"
        print_banner(f"BENCHMARKING gradient_checkpointing={gc}")
        model = load_model(args.model, dtype, gc, device)
        avg_step_seconds, peak_memory_mb, final_loss = run_benchmark(model, batch, args.num_steps, args.num_warmup, device)
        results[label] = {"avg_step_seconds": avg_step_seconds, "peak_memory_mb": peak_memory_mb, "final_loss": final_loss}
        print(f"avg_step_seconds={avg_step_seconds:.4f}  peak_memory_mb={peak_memory_mb:.1f}  final_loss={final_loss:.4f}")
        del model
        torch.cuda.empty_cache()

    print_banner("SUMMARY")
    without = results["without_checkpointing"]
    with_ = results["with_checkpointing"]
    slowdown = with_["avg_step_seconds"] / without["avg_step_seconds"]
    memory_reduction = 1 - (with_["peak_memory_mb"] / without["peak_memory_mb"])
    print(f"With checkpointing: {slowdown:.2f}x slower, {100 * memory_reduction:.1f}% peak memory reduction")

    write_run_result(
        output_dir=args.output_dir,
        stage="optimization",
        task="gradient_checkpointing_benchmark",
        modality="text",
        architecture=ARCHITECTURE,
        model_name=args.model,
        dataset_name="synthetic random tokens (no dataset -- compute/memory benchmark, see module docstring)",
        hyperparameters=vars(args),
        metrics={
            "without_checkpointing_avg_step_seconds": without["avg_step_seconds"],
            "without_checkpointing_peak_memory_mb": without["peak_memory_mb"],
            "with_checkpointing_avg_step_seconds": with_["avg_step_seconds"],
            "with_checkpointing_peak_memory_mb": with_["peak_memory_mb"],
            "slowdown": slowdown,
            "memory_reduction_frac": memory_reduction,
        },
        num_train_samples=0,
        num_eval_samples=0,
        train_runtime_seconds=(without["avg_step_seconds"] + with_["avg_step_seconds"]) * args.num_steps,
    )
    print("Done.")


if __name__ == "__main__":
    main()
