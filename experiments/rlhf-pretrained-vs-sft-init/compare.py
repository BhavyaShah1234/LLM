"""RLHF: does GRPO need an instruction-tuned starting point?

Compares two GRPO runs (rlhf/grpo/grpo.py)
that are identical in every way except --model:
  - instruct_dir: Qwen/Qwen3-1.7B (vendor instruction-tuned, this project's default)
  - pretrained_dir: Qwen/Qwen3-1.7B-Base (raw pretrained, no instruction tuning)

Both used --max_samples 40 --max_eval_samples 20 --lora --quantization 4bit
--batch_size 4 --gradient_accumulation_steps 2 --num_generations 4
--max_completion_length 48 --seed 42 -- same data, same steps, same
everything except the starting checkpoint.

The raw eval_reward numbers alone are NOT the interesting finding here (see
README.md) -- they land close enough (0.425 vs 0.375) that reading only
this column would understate the real gap. entropy and frac_reward_zero_std
tell the actual story, and this script's printed note quotes real generated
completions (captured once via direct model.generate() on the saved
adapter, hardcoded below since that's a one-time qualitative check, not
something worth re-running on every `compare.py` invocation) that make the
mechanism concrete.

Usage:
    python compare.py
"""

import argparse
import os

from common.compare_runs import load_run_results, print_comparison_table


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the two GRPO run output directories.

    Returns:
        argparse.ArgumentParser: Parser with `--instruct_dir` and `--pretrained_dir` options.
    """
    p = argparse.ArgumentParser(description="Compare GRPO starting from an instruction-tuned vs raw pretrained checkpoint.")
    p.add_argument("--instruct_dir", type=str, default="./output/rlhf/grpo")
    p.add_argument("--pretrained_dir", type=str, default="./output/rlhf/grpo-pretrained-init")
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
            f"No run_result.json found for {label} at {path}. See this folder's "
            f"README.md for the exact grpo.py commands used to produce both runs."
        )
    return load_run_results([path])[0]


def main():
    """Load both GRPO runs and print the comparison table plus a detailed reward-hacking analysis note."""
    args = build_arg_parser().parse_args()
    instruct = _load(args.instruct_dir, "instruct-init")
    pretrained = _load(args.pretrained_dir, "pretrained-init")

    print_comparison_table(
        [instruct, pretrained],
        group_by="model_name",
        metric_keys=[
            "metrics.eval_reward",
            "metrics.eval_entropy",
            "metrics.eval_frac_reward_zero_std",
            "metrics.eval_completions/clipped_ratio",
            "metrics.eval_completions/mean_terminated_length",
        ],
        title="RLHF Prerequisite: GRPO from Instruction-Tuned vs Raw Pretrained Checkpoint",
    )

    print(
        "Reading the raw eval_reward column alone (0.425 vs 0.375) understates the gap --\n"
        "on this tiny 40-sample/3-step run the pretrained-init policy looks only slightly\n"
        "worse by that one number. The real story is in the other columns:\n"
        "  - eval_entropy: 0.49 (instruct) vs 1.92 (pretrained) -- ~4x higher, a much less\n"
        "    peaked/confident output distribution.\n"
        "  - eval_frac_reward_zero_std: 0.70 (instruct) vs 0.15 (pretrained) -- the instruct\n"
        "    model gives consistent answers across its 4 sampled completions per prompt far\n"
        "    more often; the pretrained-init model's samples disagree with each other much\n"
        "    more, i.e. an erratic, not-yet-converged policy.\n"
        "  - eval_completions/mean_terminated_length: 0.0 (instruct -- every completion hit\n"
        "    the 48-token cap, a known behavior documented in grpo/README.md) vs 3.7\n"
        "    (pretrained -- completions that DO stop do so almost immediately).\n\n"
        "Direct inspection of the pretrained-init adapter's greedy generations (captured via\n"
        "model.generate(), see experiments/rlhf-pretrained-vs-sft-init/README.md) shows why:\n"
        "it converged to a degenerate shortcut -- echoing option A's text back almost every\n"
        "time -- rather than learning the task:\n"
        "  Label='A'  Completion='A. Impulse through myelinated fibers is slower than...'  (correct by luck)\n"
        "  Label='A'  Completion='A. The oncotic pressure of the fluid leaving...'          (correct by luck)\n"
        "  Label='C'  Completion='C. Amniotic fluid samples plus chromosomal analysis...'   (correct, has real explanation)\n"
        "  Label='C'  Completion='A. Antegrade'                                             (wrong -- defaulted to A again)\n\n"
        "This is a more precise and more interesting finding than 'the pretrained-init run\n"
        "just scores lower': at this tiny compute budget, GRPO's sparse verifiable reward\n"
        "gave the raw pretrained model enough signal to find a cheap ~25%-of-the-time-correct\n"
        "shortcut (always guess A) rather than genuinely learning to answer -- exactly the\n"
        "kind of reward-hacking failure mode RLHF-without-an-SFT-prior is known for, and\n"
        "exactly why rlhf/README.md documents an\n"
        "SFT/instruction-tuned starting point as a real prerequisite, not a formality.\n"
    )


if __name__ == "__main__":
    main()
