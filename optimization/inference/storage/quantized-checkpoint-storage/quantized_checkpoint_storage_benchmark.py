"""Quantized Checkpoint Storage -- isolated benchmark script (inference, storage).

Measures the REAL on-disk size of a bitsandbytes-quantized checkpoint,
complementing `../../memory/post-training-quantization/ptq.py`'s
`estimated_size_mb_quantized` metric (a theoretical estimate from
parameter counts, necessary there because that script's fake-quantized
tensors are rounded then immediately dequantized back to float and never
actually change dtype on disk -- see that script's module docstring).

**Verified before implementing, not assumed**: bitsandbytes' quantized
`Params4bit`/`Int8Params` weights are NOT just an in-memory-only
representation that reverts to full precision on save --
`model.save_pretrained()` on a 4-bit-loaded model genuinely writes the
quantized bytes to disk. Confirmed directly: saving this project's own
51M-param CLM checkpoint after `from_pretrained(...,
quantization_config=BitsAndBytesConfig(load_in_4bit=True))` produced a
62MB `model.safetensors`, vs. 195MB for the same checkpoint saved at bf16
-- a real ~68% reduction, not a theoretical one.

Uses this project's own from-scratch CLM checkpoint
(`./output/pretraining/clm`) -- already on disk, no download needed.

Usage:
    python quantized_checkpoint_storage_benchmark.py --debug_first_batch
    python quantized_checkpoint_storage_benchmark.py
"""

import argparse
import os
import time

import torch

from common.logging_utils import print_banner, print_config
from common.quantization import build_quantization_config
from common.run_results import write_run_result
from common.seeding import set_all_seeds

ARCHITECTURE = "decoder-only"
CONFIGS = ["bf16", "8bit", "4bit"]


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Measure real on-disk size of bf16 vs. bnb 8-bit vs. bnb 4-bit saved checkpoints.")
    p.add_argument("--model", type=str, default="./output/pretraining/clm", help="Model checkpoint to re-save at each quantization level. Default: this project's own from-scratch CLM checkpoint.")
    p.add_argument("--output_dir", type=str, default="./output/optimization/quantized_checkpoint_storage_benchmark", help="Where to write the resaved checkpoints and run_result.json.")
    p.add_argument("--seed", type=int, default=42, help="Random seed. Default: 42.")
    p.add_argument("--debug_first_batch", action="store_true", default=False, help="Save just the bf16 variant, print its size, then exit without the full benchmark.")
    return p


def dir_size_bytes(path: str) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            total += os.path.getsize(os.path.join(root, f))
    return total


def save_and_time(model_name: str, config_name: str, output_dir: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    os.makedirs(output_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if config_name == "bf16":
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16)
    else:
        quantization_config = build_quantization_config(config_name, "bf16")
        model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=quantization_config, dtype=torch.bfloat16, device_map="auto")

    start = time.time()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    save_seconds = time.time() - start

    del model
    torch.cuda.empty_cache()
    return save_seconds, dir_size_bytes(output_dir)


def load_and_time(output_dir: str, config_name: str):
    from transformers import AutoModelForCausalLM

    start = time.time()
    if config_name == "bf16":
        model = AutoModelForCausalLM.from_pretrained(output_dir, dtype=torch.bfloat16)
    else:
        quantization_config = build_quantization_config(config_name, "bf16")
        model = AutoModelForCausalLM.from_pretrained(output_dir, quantization_config=quantization_config, dtype=torch.bfloat16, device_map="auto")
    load_seconds = time.time() - start
    del model
    torch.cuda.empty_cache()
    return load_seconds


def main():
    args = build_arg_parser().parse_args()
    set_all_seeds(args.seed)
    print_config(args, "Quantized checkpoint storage benchmark")

    print_banner("SAVING bf16 (reference)")
    bf16_dir = os.path.join(args.output_dir, "bf16")
    bf16_save_seconds, bf16_bytes = save_and_time(args.model, "bf16", bf16_dir)
    print(f"bf16: {bf16_bytes / 1e6:.1f} MB, saved in {bf16_save_seconds:.3f}s")

    if args.debug_first_batch:
        print("--debug_first_batch set: exiting without the full benchmark.")
        return

    results = {"bf16": {"bytes": bf16_bytes, "save_seconds": bf16_save_seconds}}
    for config_name in ("8bit", "4bit"):
        print_banner(f"SAVING {config_name}")
        out_dir = os.path.join(args.output_dir, config_name)
        save_seconds, size_bytes = save_and_time(args.model, config_name, out_dir)
        print(f"{config_name}: {size_bytes / 1e6:.1f} MB, saved in {save_seconds:.3f}s")
        results[config_name] = {"bytes": size_bytes, "save_seconds": save_seconds}

    print_banner("MEASURING LOAD TIMES (from the saved quantized checkpoints, not re-quantizing from bf16)")
    for config_name in CONFIGS:
        out_dir = os.path.join(args.output_dir, config_name)
        load_seconds = load_and_time(out_dir, config_name)
        results[config_name]["load_seconds"] = load_seconds
        print(f"{config_name} load: {load_seconds:.3f}s")

    print_banner("SUMMARY")
    for config_name in ("8bit", "4bit"):
        reduction = 1 - (results[config_name]["bytes"] / results["bf16"]["bytes"])
        print(f"{config_name} vs bf16: {100 * reduction:.1f}% smaller on disk")

    metrics = {}
    for config_name, r in results.items():
        for k, v in r.items():
            metrics[f"{config_name}_{k}"] = v
    metrics["8bit_vs_bf16_reduction_frac"] = 1 - (results["8bit"]["bytes"] / results["bf16"]["bytes"])
    metrics["4bit_vs_bf16_reduction_frac"] = 1 - (results["4bit"]["bytes"] / results["bf16"]["bytes"])

    write_run_result(
        output_dir=args.output_dir,
        stage="optimization",
        task="quantized_checkpoint_storage_benchmark",
        modality="text",
        architecture=ARCHITECTURE,
        model_name=args.model,
        dataset_name="none -- disk size / load time benchmark on a real checkpoint",
        hyperparameters=vars(args),
        metrics=metrics,
        num_train_samples=0,
        num_eval_samples=0,
        train_runtime_seconds=sum(r["save_seconds"] for r in results.values()),
    )
    print("Done.")


if __name__ == "__main__":
    main()
