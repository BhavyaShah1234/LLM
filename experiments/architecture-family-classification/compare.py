"""Compare text classification across all three architecture families.

Reads the run_result.json written by the decoder-only, encoder-only, and
encoder-decoder text_classification_standard.py scripts (all trained on the
same dataset, Tohrumi/glue_sst2_10k), and prints a side-by-side table of
accuracy/F1 alongside parameter count and training wall-clock.

Usage:
    python compare.py
    python compare.py --decoder_only_dir ... --encoder_only_dir ... --encoder_decoder_dir ...
"""

import argparse
import os

from common.compare_runs import load_run_results, print_comparison_table

DEFAULT_DECODER_ONLY_DIR = "./output/supervised-finetuning/text/classification/decoder-only/standard"
DEFAULT_ENCODER_ONLY_DIR = "./output/supervised-finetuning/text/classification/encoder-only/standard"
DEFAULT_ENCODER_DECODER_DIR = "./output/supervised-finetuning/text/classification/encoder-decoder/standard"


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the three run output directories to compare.

    Returns:
        argparse.ArgumentParser: Parser with `--decoder_only_dir`,
        `--encoder_only_dir`, and `--encoder_decoder_dir` options.
    """
    p = argparse.ArgumentParser(description="Compare text classification across decoder-only, encoder-only, and encoder-decoder architectures.")
    p.add_argument("--decoder_only_dir", type=str, default=DEFAULT_DECODER_ONLY_DIR, help="Output dir passed to the decoder-only text_classification_standard.py's --output_dir.")
    p.add_argument("--encoder_only_dir", type=str, default=DEFAULT_ENCODER_ONLY_DIR, help="Output dir passed to the encoder-only text_classification_standard.py's --output_dir.")
    p.add_argument("--encoder_decoder_dir", type=str, default=DEFAULT_ENCODER_DECODER_DIR, help="Output dir passed to the encoder-decoder text_classification_standard.py's --output_dir.")
    return p


def main():
    """Load the three architectures' run_result.json files and print a comparison table."""
    args = build_arg_parser().parse_args()
    run_dirs = {
        "decoder-only": args.decoder_only_dir,
        "encoder-only": args.encoder_only_dir,
        "encoder-decoder": args.encoder_decoder_dir,
    }

    paths = []
    for label, run_dir in run_dirs.items():
        path = os.path.join(run_dir, "run_result.json")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No run_result.json found for {label} at {path}. "
                f"Run all three text_classification_standard.py scripts first "
                f"(see this folder's README.md)."
            )
        paths.append(path)

    results = load_run_results(paths)

    print_comparison_table(
        results,
        group_by="architecture",
        metric_keys=[
            "model_name",
            "metrics.accuracy",
            "metrics.f1_macro",
            "metrics.total_parameters",
            "num_train_samples",
            "train_runtime_seconds",
        ],
        title="Architecture Family Comparison: Text Classification (Tohrumi/glue_sst2_10k)",
    )

    print(
        "Note: these three runs use different base models at different scales\n"
        "(Qwen3-1.7B-Base, ModernBERT-base ~150M, t5-base ~223M), so parameter count\n"
        "and accuracy aren't controlled for model size -- this compares each\n"
        "architecture family's *typical* off-the-shelf small-model choice for this\n"
        "task, not a strictly parameter-matched ablation. Training wall-clock is a\n"
        "meaningful comparison point regardless (encoder-only/encoder-decoder full-\n"
        "parameter finetuning vs. decoder-only LoRA finetuning, if --lora was used).\n"
    )


if __name__ == "__main__":
    main()
