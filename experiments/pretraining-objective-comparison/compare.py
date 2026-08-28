"""Compare CLM (decoder-only) vs MLM (encoder-only) vs span-corruption
(encoder-decoder) pretraining runs.

Reads the run_result.json written by pretraining/clm.py, pretraining/mlm.py,
and pretraining/span_corruption.py, and prints a side-by-side table of final
loss/perplexity, parameter count, and training wall-clock/throughput.

This assumes the three underlying scripts have already been run (e.g. with
matched --max_steps so the comparison reflects the same compute budget, not
just whichever run happened to train longer) -- see pretraining/README.md.

Usage:
    python compare.py
    python compare.py --clm_dir ./output/pretraining/clm --mlm_dir ./output/pretraining/mlm --span_corruption_dir ./output/pretraining/span_corruption
"""

import argparse
import os

from common.compare_runs import load_run_results, print_comparison_table


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Compare the three pretraining/ runs (one per architecture family).")
    p.add_argument("--clm_dir", type=str, default="./output/pretraining/clm", help="Output dir passed to clm.py's --output_dir.")
    p.add_argument("--mlm_dir", type=str, default="./output/pretraining/mlm", help="Output dir passed to mlm.py's --output_dir.")
    p.add_argument("--span_corruption_dir", type=str, default="./output/pretraining/span_corruption", help="Output dir passed to span_corruption.py's --output_dir.")
    return p


def main():
    args = build_arg_parser().parse_args()
    run_dirs = {
        "clm (decoder-only)": args.clm_dir,
        "mlm (encoder-only)": args.mlm_dir,
        "span_corruption (encoder-decoder)": args.span_corruption_dir,
    }

    paths = []
    for label, run_dir in run_dirs.items():
        path = os.path.join(run_dir, "run_result.json")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No run_result.json found for {label} at {path}. "
                f"Run pretraining/{{clm,mlm,span_corruption}}.py first (see pretraining/README.md)."
            )
        paths.append(path)

    results = load_run_results(paths)

    print_comparison_table(
        results,
        group_by="architecture",
        metric_keys=[
            "task",
            "metrics.total_parameters",
            "metrics.eval_loss",
            "metrics.perplexity",
            "num_train_samples",
            "train_runtime_seconds",
        ],
        title="Pretraining Objective Comparison: CLM vs MLM vs Span-Corruption",
    )

    print(
        "Note: these are toy-scale, single-GPU, from-scratch runs on a small shared\n"
        "corpus (roneneldan/TinyStories) -- differences here reflect how each\n"
        "objective/architecture learns under a *matched, small* compute budget, not\n"
        "a general claim about which architecture family is 'better'. Perplexity is\n"
        "directly comparable between clm.py and span_corruption.py (both true\n"
        "sequence/reconstruction perplexity); mlm.py's is a conventional\n"
        "pseudo-perplexity over masked positions only, so treat cross-family\n"
        "perplexity comparisons involving mlm.py as approximate.\n"
    )


if __name__ == "__main__":
    main()
