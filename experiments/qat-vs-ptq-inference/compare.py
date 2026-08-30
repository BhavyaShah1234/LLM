"""Does Quantization-Aware Training (QAT) beat Post-Training Quantization
(PTQ)? One of this project's original motivating research questions.

Reads run_result.json from five runs, all starting from the same checkpoint
(./output/pretraining/clm by default) and using the identical fake-quant
scheme (see optimization/training/memory/quantization-aware-training/qat.py
and optimization/inference/memory/post-training-quantization/ptq.py):

  - ptq_8bit / ptq_4bit: quantize the checkpoint, zero additional training.
  - qat_8bit / qat_4bit: continue training for --max_steps steps THROUGH
    the fake-quant wrapper.
  - qat_ablation (qat.py --skip_quantization): the SAME continued-training
    run as qat_8bit/qat_4bit, but with quantization turned off.

The ablation is what makes this a fair comparison. Comparing QAT's absolute
post-training perplexity against PTQ's zero-training perplexity conflates
two effects: "more training helps regardless of quantization" and "training
through quantization noise helps the model adapt to it." This script
isolates the second effect specifically:

    PTQ's marginal quantization cost = ptq_quantized_ppl - fp32_baseline_ppl
    QAT's marginal quantization cost = qat_quantized_ppl - qat_ablation_ppl   (same training budget on both sides)

Usage:
    python compare.py
    python compare.py --ptq_8bit_dir ./output/optimization/ptq --qat_8bit_dir ./output/optimization/qat ...
"""

import argparse
import os

from common.compare_runs import load_run_results, print_comparison_table


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the five PTQ/QAT/ablation run output directories.

    Returns:
        argparse.ArgumentParser: Parser with `--ptq_8bit_dir`, `--ptq_4bit_dir`,
        `--qat_8bit_dir`, `--qat_4bit_dir`, and `--qat_ablation_dir` options.
    """
    p = argparse.ArgumentParser(description="Compare QAT vs PTQ, isolating the quantization effect from the 'more training helps' confound.")
    p.add_argument("--ptq_8bit_dir", type=str, default="./output/optimization/ptq")
    p.add_argument("--ptq_4bit_dir", type=str, default="./output/optimization/ptq-4bit")
    p.add_argument("--qat_8bit_dir", type=str, default="./output/optimization/qat")
    p.add_argument("--qat_4bit_dir", type=str, default="./output/optimization/qat-4bit")
    p.add_argument("--qat_ablation_dir", type=str, default="./output/optimization/qat-ablation-no-quant", help="qat.py --skip_quantization run: same training budget as qat_8bit/qat_4bit, no quantization. Required to isolate the quantization-specific effect.")
    return p


def _load(run_dir: str, label: str) -> dict:
    """Load a single run's run_result.json, raising a helpful error if it's missing.

    Args:
        run_dir (str): Directory the run wrote its `run_result.json` into.
        label (str): Human-readable name for this run, used in the error message.

    Returns:
        dict: The parsed run result for this run.

    Raises:
        FileNotFoundError: If `run_dir/run_result.json` doesn't exist.
    """
    path = os.path.join(run_dir, "run_result.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No run_result.json found for {label} at {path}. "
            f"Run the corresponding qat.py / ptq.py invocation first -- see this "
            f"folder's README.md for the exact commands used to produce these results."
        )
    return load_run_results([path])[0]


def main():
    """Load all five runs, print the raw comparison table, and derive the isolated marginal quantization cost for PTQ vs QAT."""
    args = build_arg_parser().parse_args()

    ptq8 = _load(args.ptq_8bit_dir, "ptq 8-bit")
    ptq4 = _load(args.ptq_4bit_dir, "ptq 4-bit")
    qat8 = _load(args.qat_8bit_dir, "qat 8-bit")
    qat4 = _load(args.qat_4bit_dir, "qat 4-bit")
    ablation = _load(args.qat_ablation_dir, "qat ablation (no quantization)")

    print_comparison_table(
        [ptq8, ptq4, qat8, qat4, ablation],
        group_by="task",
        metric_keys=[
            "hyperparameters.num_bits",
            "hyperparameters.skip_quantization",
            "metrics.baseline_perplexity",
            "metrics.quantized_perplexity",
            "metrics.perplexity_degradation",
            "train_runtime_seconds",
        ],
        title="QAT vs PTQ -- raw run_result.json fields",
    )

    ablation_ppl = ablation["metrics"]["quantized_perplexity"]  # field name is generic; here it means "post-training-budget ppl, unquantized"
    ptq8_cost = ptq8["metrics"]["perplexity_degradation"]
    ptq4_cost = ptq4["metrics"]["perplexity_degradation"]
    qat8_cost = qat8["metrics"]["quantized_perplexity"] - ablation_ppl
    qat4_cost = qat4["metrics"]["quantized_perplexity"] - ablation_ppl

    print("=" * 80)
    print("Marginal quantization cost (isolated from 'more training helps' confound)")
    print("=" * 80)
    print(f"{'':10} {'PTQ (no training)':>22} {'QAT (trained through it)':>27} {'QAT advantage':>16}")
    print(f"{'8-bit':10} {ptq8_cost:>+22.4f} {qat8_cost:>+27.4f} {ptq8_cost - qat8_cost:>+16.4f}")
    print(f"{'4-bit':10} {ptq4_cost:>+22.4f} {qat4_cost:>+27.4f} {ptq4_cost - qat4_cost:>+16.4f}")
    print()
    print(
        "Reading: 'marginal quantization cost' is the perplexity INCREASE caused by\n"
        "quantization alone, after controlling for training budget. Lower is better\n"
        "(less damage from quantizing). 'QAT advantage' = PTQ's cost minus QAT's cost;\n"
        "positive means QAT reduced the quantization-induced degradation relative to\n"
        "PTQ; ~0 means no meaningful difference.\n"
    )
    print(
        "Finding on this toy 51M-param model (TinyStories, see this folder's README\n"
        "for the exact commands): at 8-bit, PTQ's marginal cost is already ~0 -- 256\n"
        "quantization levels barely perturb this model's weight distribution, so QAT\n"
        "has nothing to recover. At 4-bit (16 levels), PTQ's marginal cost is\n"
        "substantial (~+1.4 ppl) and QAT roughly halves it (~+0.6 ppl) for the same\n"
        "200-step training budget -- i.e. QAT's advantage over PTQ is real but only\n"
        "shows up once quantization is aggressive enough to actually hurt.\n"
    )


if __name__ == "__main__":
    main()
