"""Compare text classification WITH Chain-of-Thought vs WITHOUT it.

Reads the run_result.json written by
supervised-finetuning/text/classification/decoder-only/text_classification_no_cot.py
and .../text_classification_cot.py, and prints a side-by-side table of
accuracy/F1/precision/recall plus CoT-specific metrics.

Assumes both scripts have already been run with matched hyperparameters
(same model, dataset, epochs, LoRA config -- only the CoT toggle differs) --
see this folder's README.md for the exact commands.

Usage:
    python compare.py
    python compare.py --no_cot_dir ./output/.../no_cot --cot_dir ./output/.../cot
"""

import argparse
import os

from common.compare_runs import load_run_results, print_comparison_table

DEFAULT_NO_COT_DIR = "./output/supervised-finetuning/text/classification/decoder-only/no_cot"
DEFAULT_COT_DIR = "./output/supervised-finetuning/text/classification/decoder-only/cot"


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the no-CoT and CoT run output directories.

    Returns:
        argparse.ArgumentParser: Parser with `--no_cot_dir` and `--cot_dir` options.
    """
    p = argparse.ArgumentParser(description="Compare CoT vs no-CoT text classification runs.")
    p.add_argument("--no_cot_dir", type=str, default=DEFAULT_NO_COT_DIR, help="Output dir passed to text_classification_no_cot.py's --output_dir.")
    p.add_argument("--cot_dir", type=str, default=DEFAULT_COT_DIR, help="Output dir passed to text_classification_cot.py's --output_dir.")
    return p


def main():
    """Load the CoT and no-CoT run_result.json files and print a comparison table."""
    args = build_arg_parser().parse_args()
    run_dirs = {"no_cot": args.no_cot_dir, "cot": args.cot_dir}

    paths = []
    for label, run_dir in run_dirs.items():
        path = os.path.join(run_dir, "run_result.json")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No run_result.json found for {label} at {path}. "
                f"Run text_classification_no_cot.py and text_classification_cot.py first "
                f"(see this folder's README.md)."
            )
        paths.append(path)

    results = load_run_results(paths)

    print_comparison_table(
        results,
        group_by="variant",
        metric_keys=[
            "metrics.accuracy",
            "metrics.f1_macro",
            "metrics.f1_weighted",
            "metrics.precision",
            "metrics.recall",
            "metrics.cot_usage_rate",
            "metrics.avg_cot_length_words",
            "train_runtime_seconds",
        ],
        title="CoT vs No-CoT Text Classification Comparison",
    )

    print(
        "Note: this compares the SAME dataset/model/hyperparameters with only the CoT\n"
        "toggle differing. A higher accuracy/F1 for 'cot' suggests reasoning supervision\n"
        "helped this task; a similar or lower score suggests it didn't (or that more\n"
        "training data/steps would be needed to see a benefit) -- domofon/fake_news_cot_reasoning\n"
        "is a modest-sized dataset, so run with a realistic --max_samples (not the default\n"
        "-1 == full split, unless you have the compute budget) for a meaningful comparison.\n"
    )


if __name__ == "__main__":
    main()
