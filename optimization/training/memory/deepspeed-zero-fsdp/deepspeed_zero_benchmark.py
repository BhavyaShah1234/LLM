"""DeepSpeed ZeRO (Stage 2, CPU optimizer offload) -- isolated benchmark
script (training, memory).

Compares plain AdamW (all state on GPU) against DeepSpeed ZeRO Stage 2
with the optimizer state offloaded to CPU RAM, on a real forward +
backward + optimizer step. ZeRO's headline use case is sharding
optimizer/gradient/parameter state ACROSS MULTIPLE GPUs -- with this
project's single-GPU hardware (world_size=1), that specific benefit
doesn't apply (nothing to shard across). What DOES apply on a single GPU
is ZeRO-Offload: moving optimizer state to CPU RAM instead of GPU memory,
trading GPU memory for host RAM + PCIe transfer time -- a real, testable
technique here, and what this script measures.

Uses this project's own from-scratch CLM checkpoint
(`./output/pretraining/clm`, 51M params) rather than a larger model:
ZeRO-Offload's benefit is proportional to optimizer state SIZE (Adam's
two fp32 moment buffers per parameter), and this project's available CPU
RAM (~13GB free at benchmark time, `free -h` checked before writing this
script) can't comfortably hold a 1.7B-parameter model's full-parameter
Adam state (~13.6GB) alongside everything else already resident. The
small CLM checkpoint's Adam state (~408MB) is a size CPU-offload can
actually demonstrate cleanly on this hardware without itself becoming the
bottleneck.

**Real environment bug found and worked around**: DeepSpeed's CPU-offloaded
optimizer path defaults to `DeepSpeedCPUAdam`, a custom fused CPU/CUDA
kernel that JIT-compiles on first use. That compilation failed here with
`CUDAMismatchException: Installed CUDA version 13.3 does not match the
version torch was compiled with 13.0` -- a real toolkit/PyTorch-build
mismatch on this machine, not a DeepSpeed bug or something fixable by
changing this project's code alone. Worked around via DeepSpeed's own
documented escape hatch: `zero_force_ds_cpu_optimizer: false` at the
top-level config plus `torch_adam: true` in the optimizer params, which
falls back to plain `torch.optim.AdamW` for the CPU-offloaded optimizer
instead of the JIT-compiled kernel -- slower than the fused kernel would
be, but functionally correct and still demonstrates the real GPU-memory
trade this benchmark exists to measure. DeepSpeed's own error message for
the *other* failure mode (using a non-DeepSpeed optimizer without this
flag) names this exact flag directly, which is how it was found.

DeepSpeed requires a torch distributed process group even for a single
process -- initialized here via `deepspeed.init_distributed()` with
`RANK=WORLD_SIZE=LOCAL_RANK` set for a single-process "distributed" run.
The plain-AdamW baseline is measured BEFORE this initialization (order
matters: once a NCCL process group exists in a process, cleanly reverting
to non-distributed training in the same process isn't guaranteed).

Usage:
    python deepspeed_zero_benchmark.py --debug_first_batch
    python deepspeed_zero_benchmark.py --batch_size 8 --seq_len 256 --num_steps 10
"""

import argparse
import os
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
        synthetic-batch shape, step counts, learning rate, DeepSpeed master
        port, seed, output dir, debug flag) registered.
    """
    p = argparse.ArgumentParser(description="Benchmark plain AdamW vs. DeepSpeed ZeRO Stage 2 with CPU optimizer offload.")
    p.add_argument("--model", type=str, default="./output/pretraining/clm", help="Model to benchmark. Default: this project's own from-scratch CLM checkpoint -- see module docstring for why (CPU RAM budget for the offloaded optimizer state).")
    p.add_argument("--batch_size", type=int, default=8, help="Synthetic batch size. Default: 8.")
    p.add_argument("--seq_len", type=int, default=256, help="Synthetic sequence length. Default: 256.")
    p.add_argument("--num_steps", type=int, default=10, help="Timed optimizer steps per configuration (after warmup). Default: 10.")
    p.add_argument("--num_warmup", type=int, default=2, help="Untimed warmup steps. Default: 2.")
    p.add_argument("--learning_rate", type=float, default=1e-4, help="AdamW learning rate. Default: 1e-4.")
    p.add_argument("--master_port", type=str, default="29500", help="Port for DeepSpeed's single-process distributed init. Default: 29500.")
    p.add_argument("--seed", type=int, default=42, help="Random seed. Default: 42.")
    p.add_argument("--output_dir", type=str, default="./output/optimization/deepspeed_zero_benchmark", help="Where to write run_result.json.")
    p.add_argument("--debug_first_batch", action="store_true", default=False, help="Run one step of each configuration, print loss, then exit without the full benchmark.")
    return p


def build_synthetic_batch(vocab_size: int, batch_size: int, seq_len: int, device: str):
    """Build a random-token CLM batch used to drive both benchmark configurations.

    Args:
        vocab_size (int): Vocabulary size to sample token ids from.
        batch_size (int): Number of sequences in the batch.
        seq_len (int): Length of each sequence.
        device (str): Torch device to place the tensors on.

    Returns:
        dict: ``{"input_ids": Tensor, "labels": Tensor}`` with `labels` a
        clone of `input_ids` (standard CLM next-token-prediction setup).
    """
    input_ids = torch.randint(low=0, high=vocab_size, size=(batch_size, seq_len), device=device)
    labels = input_ids.clone()
    return {"input_ids": input_ids, "labels": labels}


def run_baseline(model_path: str, batch, num_steps: int, num_warmup: int, learning_rate: float, device: str):
    """Time plain AdamW with all optimizer state on GPU.

    Measured before DeepSpeed's distributed process group is ever created
    in this process (see module docstring: reverting a process from
    distributed back to non-distributed isn't guaranteed once a NCCL group
    exists).

    Args:
        model_path (str): Path or hub id of the causal LM checkpoint to load.
        batch (dict): Synthetic batch from `build_synthetic_batch`, already on `device`.
        num_steps (int): Timed optimizer steps to run after warmup.
        num_warmup (int): Untimed warmup steps run before timing starts.
        learning_rate (float): AdamW learning rate.
        device (str): Torch device to run on (e.g. `"cuda"`).

    Returns:
        tuple[float, float, float]: `(avg_step_seconds, peak_memory_mb, final_loss)`
        -- mean wall-clock time per timed step, peak CUDA memory allocated
        during the timed steps in MB, and the loss from the last timed step.
    """
    from common.model_loading import load_causal_lm

    model = load_causal_lm(model_path, quantization_config=None, torch_dtype=torch.bfloat16, gradient_checkpointing=False).to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for _ in range(num_warmup):
        optimizer.zero_grad(set_to_none=True)
        outputs = model(**batch)
        outputs.loss.backward()
        optimizer.step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(device)

    start = time.time()
    final_loss = None
    for _ in range(num_steps):
        optimizer.zero_grad(set_to_none=True)
        outputs = model(**batch)
        final_loss = outputs.loss.item()
        outputs.loss.backward()
        optimizer.step()
    torch.cuda.synchronize()
    elapsed = time.time() - start

    peak_memory_mb = torch.cuda.max_memory_allocated(device) / 1e6
    del model, optimizer
    torch.cuda.empty_cache()
    return elapsed / num_steps, peak_memory_mb, final_loss


def run_zero_offload(model_path: str, batch, num_steps: int, num_warmup: int, learning_rate: float, master_port: str):
    """Time DeepSpeed ZeRO Stage 2 with the optimizer state offloaded to CPU RAM.

    Initializes a single-process DeepSpeed distributed group, wraps the
    model with ZeRO Stage 2 CPU-offloaded AdamW, and times the same
    warmup/timed-step structure as `run_baseline`.

    Args:
        model_path (str): Path or hub id of the causal LM checkpoint to load.
        batch (dict): Synthetic batch from `build_synthetic_batch`; moved onto the DeepSpeed engine's device internally.
        num_steps (int): Timed optimizer steps to run after warmup.
        num_warmup (int): Untimed warmup steps run before timing starts.
        learning_rate (float): AdamW learning rate.
        master_port (str): Port used for DeepSpeed's single-process distributed init.

    Returns:
        tuple[float, float, float]: `(avg_step_seconds, peak_memory_mb, final_loss)`
        -- mean wall-clock time per timed step, peak CUDA memory allocated
        during the timed steps in MB, and the loss from the last timed step.
    """
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", master_port)

    import deepspeed

    from common.model_loading import load_causal_lm

    deepspeed.init_distributed(dist_backend="nccl")

    model = load_causal_lm(model_path, quantization_config=None, torch_dtype=torch.bfloat16, gradient_checkpointing=False)

    ds_config = {
        "train_batch_size": batch["input_ids"].shape[0],
        "train_micro_batch_size_per_gpu": batch["input_ids"].shape[0],
        "zero_optimization": {"stage": 2, "offload_optimizer": {"device": "cpu"}},
        "zero_force_ds_cpu_optimizer": False,  # see module docstring: works around a CUDA/PyTorch build-version mismatch that breaks the fused CPUAdam kernel's JIT compile
        "bf16": {"enabled": True},
        "optimizer": {"type": "AdamW", "params": {"lr": learning_rate, "torch_adam": True}},
    }

    model_engine, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=ds_config)
    device = model_engine.device
    batch = {k: v.to(device) for k, v in batch.items()}

    for _ in range(num_warmup):
        outputs = model_engine(**batch)
        model_engine.backward(outputs.loss)
        model_engine.step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(device)

    start = time.time()
    final_loss = None
    for _ in range(num_steps):
        outputs = model_engine(**batch)
        final_loss = outputs.loss.item()
        model_engine.backward(outputs.loss)
        model_engine.step()
    torch.cuda.synchronize()
    elapsed = time.time() - start

    peak_memory_mb = torch.cuda.max_memory_allocated(device) / 1e6
    return elapsed / num_steps, peak_memory_mb, final_loss


def main():
    """Run the baseline-vs-ZeRO-offload benchmark and write a `run_result.json`.

    Parses CLI args, builds a synthetic CLM batch, benchmarks plain AdamW
    then DeepSpeed ZeRO Stage 2 CPU offload (or just one debug step of each
    if `--debug_first_batch` is set), prints a summary, and records the
    comparison via `common.run_results.write_run_result`.
    """
    args = build_arg_parser().parse_args()
    set_all_seeds(args.seed)
    print_config(args, "DeepSpeed ZeRO Stage 2 (CPU optimizer offload) benchmark")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    from transformers import AutoConfig

    vocab_size = AutoConfig.from_pretrained(args.model).vocab_size
    batch = build_synthetic_batch(vocab_size, args.batch_size, args.seq_len, device)
    print(f"Synthetic batch: batch_size={args.batch_size}, seq_len={args.seq_len}, vocab_size={vocab_size}")

    if args.debug_first_batch:
        print_banner("DEBUG: baseline (plain AdamW, GPU)")
        avg_step_seconds, peak_memory_mb, final_loss = run_baseline(args.model, batch, num_steps=1, num_warmup=0, learning_rate=args.learning_rate, device=device)
        print(f"loss={final_loss:.4f}  peak_memory_mb={peak_memory_mb:.1f}")
        print_banner("DEBUG: DeepSpeed ZeRO Stage 2 (CPU offload)")
        avg_step_seconds, peak_memory_mb, final_loss = run_zero_offload(args.model, batch, num_steps=1, num_warmup=0, learning_rate=args.learning_rate, master_port=args.master_port)
        print(f"loss={final_loss:.4f}  peak_memory_mb={peak_memory_mb:.1f}")
        print("--debug_first_batch set: exiting without the full benchmark.")
        return

    print_banner("BENCHMARKING baseline (plain AdamW, all state on GPU)")
    baseline_step_seconds, baseline_peak_mb, baseline_loss = run_baseline(args.model, batch, args.num_steps, args.num_warmup, args.learning_rate, device)
    print(f"avg_step_seconds={baseline_step_seconds:.4f}  peak_memory_mb={baseline_peak_mb:.1f}  final_loss={baseline_loss:.4f}")

    print_banner("BENCHMARKING DeepSpeed ZeRO Stage 2 (optimizer state offloaded to CPU RAM)")
    zero_step_seconds, zero_peak_mb, zero_loss = run_zero_offload(args.model, batch, args.num_steps, args.num_warmup, args.learning_rate, args.master_port)
    print(f"avg_step_seconds={zero_step_seconds:.4f}  peak_memory_mb={zero_peak_mb:.1f}  final_loss={zero_loss:.4f}")

    print_banner("SUMMARY")
    slowdown = zero_step_seconds / baseline_step_seconds
    memory_reduction = 1 - (zero_peak_mb / baseline_peak_mb)
    print(f"ZeRO CPU offload vs baseline: {slowdown:.2f}x step time, {100 * memory_reduction:.1f}% peak GPU memory reduction")

    write_run_result(
        output_dir=args.output_dir,
        stage="optimization",
        task="deepspeed_zero_benchmark",
        modality="text",
        architecture=ARCHITECTURE,
        model_name=args.model,
        dataset_name="synthetic random tokens (no dataset -- compute/memory benchmark)",
        hyperparameters=vars(args),
        metrics={
            "baseline_avg_step_seconds": baseline_step_seconds,
            "baseline_peak_memory_mb": baseline_peak_mb,
            "zero_offload_avg_step_seconds": zero_step_seconds,
            "zero_offload_peak_memory_mb": zero_peak_mb,
            "slowdown": slowdown,
            "memory_reduction_frac": memory_reduction,
        },
        num_train_samples=0,
        num_eval_samples=0,
        train_runtime_seconds=(baseline_step_seconds + zero_step_seconds) * args.num_steps,
    )
    print("Done.")


if __name__ == "__main__":
    main()
