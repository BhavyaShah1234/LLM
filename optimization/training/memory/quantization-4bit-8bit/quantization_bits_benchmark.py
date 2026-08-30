"""4-bit / 8-bit Quantization for training -- isolated benchmark script
(training, memory).

Compares three bitsandbytes quantization levels for the FROZEN base model
during LoRA training -- `no` (bf16), `8bit`, `4bit` -- holding the LoRA
config fixed across all three. This isolates a different variable than
`../lora-qlora/lora_qlora_benchmark.py`, which held quantization fixed
(bf16 vs 4-bit-QLoRA) and varied whether LoRA was used at all (full vs
LoRA vs QLoRA). Here, LoRA is used throughout; only the base model's
quantization bit-width changes -- answering "given I'm already using
LoRA, does going from 8-bit to 4-bit save meaningful memory, and at what
speed cost?" specifically.

Real forward + backward + `optimizer.step()`, same reasoning as
`../lora-qlora/lora_qlora_benchmark.py`: the point is measuring the
trained-through memory/speed profile, not just a forward pass.

Usage:
    python quantization_bits_benchmark.py --debug_first_batch
    python quantization_bits_benchmark.py --batch_size 2 --seq_len 256 --num_steps 5
"""

import argparse
import time

import torch

from common.logging_utils import print_banner, print_config
from common.peft_setup import apply_lora, build_lora_config
from common.quantization import build_quantization_config
from common.run_results import write_run_result
from common.seeding import set_all_seeds

ARCHITECTURE = "decoder-only"
CONFIGS = ["no", "8bit", "4bit"]


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for this benchmark.

    Returns:
        argparse.ArgumentParser: Parser with all benchmark options (model,
        synthetic-batch shape, step counts, learning rate, seed, output
        dir, debug flag) registered.
    """
    p = argparse.ArgumentParser(description="Benchmark bf16 vs. 8-bit vs. 4-bit base-model quantization for LoRA training.")
    p.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B-Base", help="Model to benchmark (local path or HF Hub id). Default: this project's usual base-model default.")
    p.add_argument("--batch_size", type=int, default=2, help="Synthetic batch size. Default: 2.")
    p.add_argument("--seq_len", type=int, default=256, help="Synthetic sequence length. Default: 256.")
    p.add_argument("--num_steps", type=int, default=5, help="Timed optimizer steps per configuration (after warmup). Default: 5.")
    p.add_argument("--num_warmup", type=int, default=1, help="Untimed warmup steps. Default: 1.")
    p.add_argument("--learning_rate", type=float, default=1e-4, help="AdamW learning rate. Default: 1e-4.")
    p.add_argument("--seed", type=int, default=42, help="Random seed. Default: 42.")
    p.add_argument("--output_dir", type=str, default="./output/optimization/quantization_bits_benchmark", help="Where to write run_result.json.")
    p.add_argument("--debug_first_batch", action="store_true", default=False, help="Build one synthetic batch, load each config, print trainable-param counts, then exit without timing.")
    return p


def build_synthetic_batch(vocab_size: int, batch_size: int, seq_len: int, device: str):
    """Build a random-token CLM batch used to drive all three quantization configurations.

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


def load_model(model_name: str, config_name: str, device: str):
    """Load a LoRA-wrapped causal LM with the base model at a given quantization level.

    Args:
        model_name (str): Local path or HF Hub id of the base checkpoint to load.
        config_name (str): One of `CONFIGS` (`"no"`, `"8bit"`, `"4bit"`) selecting the base-model quantization level. `"no"` loads plain bf16; `"8bit"`/`"4bit"` build a bitsandbytes quantization config.
        device (str): Torch device to move the model to (unused when `config_name` is not `"no"`, which uses `device_map="auto"`).

    Returns:
        transformers.PreTrainedModel: The LoRA-wrapped model in `.train()` mode.
    """
    from transformers import AutoModelForCausalLM

    if config_name == "no":
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16, trust_remote_code=True).to(device)
        model = apply_lora(model, build_lora_config(), prepare_for_kbit=False, print_trainable=False)
    else:
        quantization_config = build_quantization_config(config_name, "bf16")
        model = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=quantization_config, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
        )
        model = apply_lora(model, build_lora_config(), prepare_for_kbit=True, print_trainable=False)
    model.train()
    return model


def count_params(model):
    """Count trainable vs. total parameters of a model.

    Args:
        model (torch.nn.Module): Model to inspect.

    Returns:
        tuple[int, int]: `(trainable_params, total_params)`.
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def run_benchmark(model, batch, num_steps: int, num_warmup: int, learning_rate: float, device: str):
    """Time a real optimizer step (forward + backward + `optimizer.step()`) for one quantization level.

    Args:
        model (torch.nn.Module): Model to benchmark, already in train mode on `device`.
        batch (dict): Batch from `build_synthetic_batch`, already on `device`.
        num_steps (int): Timed optimizer steps to run after warmup.
        num_warmup (int): Untimed warmup steps run before timing starts.
        learning_rate (float): AdamW learning rate for the LoRA parameters.
        device (str): Torch device the model and batch live on.

    Returns:
        tuple[float, float, float]: `(avg_step_seconds, peak_memory_mb, final_loss)`
        -- mean wall-clock time per timed step, peak CUDA memory allocated
        during the timed steps in MB, and the loss from the last timed step.
    """
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate)

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
    avg_step_seconds = elapsed / num_steps
    return avg_step_seconds, peak_memory_mb, final_loss


def main():
    """Run the bf16/8-bit/4-bit LoRA-training benchmark and write a `run_result.json`.

    Parses CLI args, builds a synthetic CLM batch, benchmarks each
    quantization level in `CONFIGS` (or just prints trainable-param counts
    if `--debug_first_batch` is set), catching and recording per-level OOMs
    rather than crashing, prints a summary, and records the comparison via
    `common.run_results.write_run_result`.
    """
    args = build_arg_parser().parse_args()
    set_all_seeds(args.seed)
    print_config(args, "4-bit vs 8-bit vs bf16 LoRA-training quantization benchmark")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    from transformers import AutoConfig

    vocab_size = AutoConfig.from_pretrained(args.model, trust_remote_code=True).vocab_size
    batch = build_synthetic_batch(vocab_size, args.batch_size, args.seq_len, device)
    print(f"Synthetic batch: batch_size={args.batch_size}, seq_len={args.seq_len}, vocab_size={vocab_size}")

    if args.debug_first_batch:
        for config_name in CONFIGS:
            print_banner(f"DEBUG: quantization={config_name}")
            model = load_model(args.model, config_name, device)
            trainable, total = count_params(model)
            print(f"trainable_params={trainable:,}  total_params={total:,}  ({100 * trainable / total:.3f}% trainable)")
            del model
            torch.cuda.empty_cache()
        print("--debug_first_batch set: exiting without the full benchmark.")
        return

    results = {}
    for config_name in CONFIGS:
        print_banner(f"BENCHMARKING quantization={config_name}")
        model = None
        try:
            model = load_model(args.model, config_name, device)
            trainable, total = count_params(model)
            print(f"trainable_params={trainable:,}  total_params={total:,}")
            avg_step_seconds, peak_memory_mb, final_loss = run_benchmark(model, batch, args.num_steps, args.num_warmup, args.learning_rate, device)
            results[config_name] = {"trainable_params": trainable, "total_params": total, "avg_step_seconds": avg_step_seconds, "peak_memory_mb": peak_memory_mb, "final_loss": final_loss, "oom": False}
            print(f"avg_step_seconds={avg_step_seconds:.4f}  peak_memory_mb={peak_memory_mb:.1f}  final_loss={final_loss:.4f}")
        except torch.OutOfMemoryError as e:
            print(f"OOM: quantization={config_name} does not fit. ({e})")
            results[config_name] = {"trainable_params": None, "total_params": None, "avg_step_seconds": None, "peak_memory_mb": None, "final_loss": None, "oom": True}
        finally:
            del model  # see optimization/training/memory/lora-qlora/README.md's real bug note: this must run even on OOM, or cascading failures follow
            torch.cuda.empty_cache()

    print_banner("SUMMARY")
    no_quant = results["no"]
    for name in ("8bit", "4bit"):
        r = results[name]
        if r["oom"] or no_quant["oom"]:
            print(f"{name}: OOM (or bf16 baseline OOM'd) -- no comparison possible")
            continue
        slowdown = r["avg_step_seconds"] / no_quant["avg_step_seconds"]
        memory_reduction = 1 - (r["peak_memory_mb"] / no_quant["peak_memory_mb"])
        print(f"{name} vs bf16: {slowdown:.2f}x step time, {100 * memory_reduction:.1f}% peak memory reduction")

    metrics = {}
    for config_name, r in results.items():
        for k, v in r.items():
            metrics[f"{config_name}_{k}"] = v

    write_run_result(
        output_dir=args.output_dir,
        stage="optimization",
        task="quantization_bits_benchmark",
        modality="text",
        architecture=ARCHITECTURE,
        model_name=args.model,
        dataset_name="synthetic random tokens (no dataset -- compute/memory benchmark)",
        hyperparameters=vars(args),
        metrics=metrics,
        num_train_samples=0,
        num_eval_samples=0,
        train_runtime_seconds=sum((r["avg_step_seconds"] or 0) * args.num_steps for r in results.values()),
    )
    print("Done.")


if __name__ == "__main__":
    main()
