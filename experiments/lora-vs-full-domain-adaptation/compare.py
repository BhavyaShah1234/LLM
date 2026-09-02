"""Does base-weight-updated domain adaptation beat LoRA-weighted domain
adaptation, measured by downstream finetuning performance?

Reads run_result.json from two supervised-finetuning/text/mcq/decoder-only/
mcq_standard.py runs -- identical except which domain-adaptation output they
started from:
  - lora_dir: domain_adaptation.py's default (LoRA adapter, merged for a
    directly-loadable checkpoint).
  - full_dir: domain_adaptation.py --full_finetune (base weights updated).

See README.md for the full four-run protocol (both domain-adaptation arms,
then both downstream finetuning arms) and what this comparison does and
doesn't isolate.

Usage:
    python compare.py
    python compare.py --lora_dir ./output/supervised-finetuning/mcq-from-domain-adapted-lora --full_dir ./output/supervised-finetuning/mcq-from-domain-adapted-full
"""

import argparse
import os

from common.compare_runs import load_run_results, print_comparison_table


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the two downstream-finetuning run output directories.

    Returns:
        argparse.ArgumentParser: Parser with `--lora_dir` and `--full_dir` options.
    """
    p = argparse.ArgumentParser(description="Compare downstream finetuning performance starting from a LoRA- vs base-weight-domain-adapted checkpoint.")
    p.add_argument("--lora_dir", type=str, default="./output/supervised-finetuning/mcq-from-domain-adapted-lora", help="mcq_standard.py run started from the LoRA-domain-adapted checkpoint.")
    p.add_argument("--full_dir", type=str, default="./output/supervised-finetuning/mcq-from-domain-adapted-full", help="mcq_standard.py run started from the full-finetune-domain-adapted checkpoint.")
    return p


def _load(run_dir: str, label: str) -> dict:
    """Load a single run's run_result.json, raising a helpful error if it's missing.

    Args:
        run_dir (str): Directory the run wrote its `run_result.json` into.
        label (str): Human-readable name for this run, used in the error message.

    Returns:
        dict: The parsed run result for this run.

    Raises:
        FileNotFoundError: If no run_result.json exists at `run_dir`.
    """
    path = os.path.join(run_dir, "run_result.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No run_result.json at {path} ({label} run). See README.md for "
            f"the full four-run protocol -- this experiment isn't runnable "
            f"until both domain-adaptation arms and both downstream "
            f"finetuning arms have completed."
        )
    return load_run_results([run_dir])[0]


def main():
    """Parse CLI args, load both downstream-finetuning runs, and print the comparison."""
    args = build_arg_parser().parse_args()

    lora_result = _load(args.lora_dir, "LoRA-domain-adapted")
    full_result = _load(args.full_dir, "full-finetune-domain-adapted")

    print_comparison_table(
        [lora_result, full_result],
        group_by="model_name",
        metric_keys=["metrics.eval_loss", "metrics.eval_accuracy", "hyperparameters.max_samples", "hyperparameters.epochs"],
        title="Downstream MCQ finetuning: starting from LoRA- vs base-weight-domain-adapted checkpoint",
    )
    print(
        "Note: this isolates the domain-adaptation weight-update strategy's "
        "effect on downstream performance, holding the downstream data "
        "budget fixed -- it does NOT control for the full-finetune arm "
        "possibly needing a different --batch_size/--optimizer to fit in "
        "VRAM during domain-adaptation (see README.md)."
    )


if __name__ == "__main__":
    main()
