"""Checkpoint Compression -- isolated benchmark script (training, storage).

Saves the same real checkpoint in three ways and measures actual on-disk
bytes and load time for each: fp32 safetensors, bf16 safetensors, and a
gzip-compressed archive of the bf16 safetensors file. Real file sizes via
`os.path.getsize`, not estimates -- unlike
`../../inference/memory/post-training-quantization/ptq.py`'s
`estimated_size_mb_quantized` (which computes a theoretical size from
parameter counts because that script's fake-quantized tensors never
actually change dtype on disk), this script writes real files and measures
them directly, since dtype casting for a save is a completely real,
uncomplicated operation (unlike fake-quantization).

Uses this project's own from-scratch CLM checkpoint
(`./output/pretraining/clm`) -- already on disk, no download needed, and
a size where all three variants save/load quickly enough for repeated
runs.

Usage:
    python checkpoint_compression_benchmark.py --debug_first_batch
    python checkpoint_compression_benchmark.py
"""

import argparse
import gzip
import os
import shutil
import time

import torch

from common.logging_utils import print_banner, print_config
from common.run_results import write_run_result
from common.seeding import set_all_seeds

ARCHITECTURE = "decoder-only"


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Benchmark checkpoint disk size and load time across save formats.")
    p.add_argument("--model", type=str, default="./output/pretraining/clm", help="Model checkpoint to re-save in different formats. Default: this project's own from-scratch CLM checkpoint.")
    p.add_argument("--output_dir", type=str, default="./output/optimization/checkpoint_compression_benchmark", help="Where to write the resaved checkpoints and run_result.json.")
    p.add_argument("--seed", type=int, default=42, help="Random seed. Default: 42.")
    p.add_argument("--debug_first_batch", action="store_true", default=False, help="Save just the fp32 variant, print its size, then exit without the full benchmark.")
    return p


def dir_size_bytes(path: str) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            total += os.path.getsize(os.path.join(root, f))
    return total


def save_and_time(model, tokenizer, output_dir: str, dtype: torch.dtype):
    os.makedirs(output_dir, exist_ok=True)
    model_cast = model.to(dtype)
    start = time.time()
    model_cast.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    save_seconds = time.time() - start
    return save_seconds, dir_size_bytes(output_dir)


def gzip_compress_and_time(src_path: str, dst_path: str):
    start = time.time()
    with open(src_path, "rb") as f_in, gzip.open(dst_path, "wb", compresslevel=6) as f_out:
        shutil.copyfileobj(f_in, f_out)
    compress_seconds = time.time() - start
    return compress_seconds, os.path.getsize(dst_path)


def gzip_decompress_and_time(src_path: str, dst_path: str):
    start = time.time()
    with gzip.open(src_path, "rb") as f_in, open(dst_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    decompress_seconds = time.time() - start
    return decompress_seconds


def find_weights_file(directory: str) -> str:
    for name in ("model.safetensors", "pytorch_model.bin"):
        candidate = os.path.join(directory, name)
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(f"No model.safetensors or pytorch_model.bin found in {directory}")


def main():
    args = build_arg_parser().parse_args()
    set_all_seeds(args.seed)
    print_config(args, "Checkpoint compression benchmark")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print_banner("LOADING CHECKPOINT")
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    fp32_dir = os.path.join(args.output_dir, "fp32")
    print_banner("SAVING fp32")
    fp32_save_seconds, fp32_bytes = save_and_time(model, tokenizer, fp32_dir, torch.float32)
    print(f"fp32: {fp32_bytes / 1e6:.1f} MB, saved in {fp32_save_seconds:.3f}s")

    if args.debug_first_batch:
        print("--debug_first_batch set: exiting without the full benchmark.")
        return

    bf16_dir = os.path.join(args.output_dir, "bf16")
    print_banner("SAVING bf16")
    bf16_save_seconds, bf16_bytes = save_and_time(model, tokenizer, bf16_dir, torch.bfloat16)
    print(f"bf16: {bf16_bytes / 1e6:.1f} MB, saved in {bf16_save_seconds:.3f}s")

    print_banner("GZIP-COMPRESSING bf16 WEIGHTS FILE")
    bf16_weights_path = find_weights_file(bf16_dir)
    gzip_path = bf16_weights_path + ".gz"
    gzip_compress_seconds, gzip_bytes = gzip_compress_and_time(bf16_weights_path, gzip_path)
    print(f"gzip(bf16): {gzip_bytes / 1e6:.1f} MB, compressed in {gzip_compress_seconds:.3f}s")

    print_banner("MEASURING LOAD TIMES")
    start = time.time()
    _ = AutoModelForCausalLM.from_pretrained(fp32_dir, dtype=torch.float32)
    fp32_load_seconds = time.time() - start

    start = time.time()
    _ = AutoModelForCausalLM.from_pretrained(bf16_dir, dtype=torch.bfloat16)
    bf16_load_seconds = time.time() - start

    decompressed_path = os.path.join(args.output_dir, "decompressed_model.safetensors")
    gzip_decompress_seconds = gzip_decompress_and_time(gzip_path, decompressed_path)
    gzip_effective_load_seconds = gzip_decompress_seconds + bf16_load_seconds  # decompress-then-load is the real end-to-end cost of shipping the .gz form

    print(f"fp32 load: {fp32_load_seconds:.3f}s")
    print(f"bf16 load: {bf16_load_seconds:.3f}s")
    print(f"gzip(bf16) decompress+load: {gzip_effective_load_seconds:.3f}s ({gzip_decompress_seconds:.3f}s decompress + {bf16_load_seconds:.3f}s load)")

    print_banner("SUMMARY")
    bf16_vs_fp32_reduction = 1 - (bf16_bytes / fp32_bytes)
    gzip_vs_bf16_reduction = 1 - (gzip_bytes / bf16_bytes)
    gzip_vs_fp32_reduction = 1 - (gzip_bytes / fp32_bytes)
    print(f"bf16 vs fp32: {100 * bf16_vs_fp32_reduction:.1f}% smaller")
    print(f"gzip(bf16) vs bf16: {100 * gzip_vs_bf16_reduction:.1f}% smaller, but +{gzip_decompress_seconds:.3f}s decompress cost before every load")
    print(f"gzip(bf16) vs fp32: {100 * gzip_vs_fp32_reduction:.1f}% smaller overall")

    write_run_result(
        output_dir=args.output_dir,
        stage="optimization",
        task="checkpoint_compression_benchmark",
        modality="text",
        architecture=ARCHITECTURE,
        model_name=args.model,
        dataset_name="none -- disk size / load time benchmark on a real checkpoint",
        hyperparameters=vars(args),
        metrics={
            "fp32_bytes": fp32_bytes,
            "fp32_save_seconds": fp32_save_seconds,
            "fp32_load_seconds": fp32_load_seconds,
            "bf16_bytes": bf16_bytes,
            "bf16_save_seconds": bf16_save_seconds,
            "bf16_load_seconds": bf16_load_seconds,
            "gzip_bf16_bytes": gzip_bytes,
            "gzip_compress_seconds": gzip_compress_seconds,
            "gzip_decompress_seconds": gzip_decompress_seconds,
            "gzip_effective_load_seconds": gzip_effective_load_seconds,
            "bf16_vs_fp32_reduction_frac": bf16_vs_fp32_reduction,
            "gzip_vs_bf16_reduction_frac": gzip_vs_bf16_reduction,
            "gzip_vs_fp32_reduction_frac": gzip_vs_fp32_reduction,
            "total_parameters": total_params,
        },
        num_train_samples=0,
        num_eval_samples=0,
        train_runtime_seconds=fp32_save_seconds + bf16_save_seconds + gzip_compress_seconds,
    )
    print("Done.")


if __name__ == "__main__":
    main()
