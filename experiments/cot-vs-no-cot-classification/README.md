# CoT vs No-CoT Classification

Status: **done (Wave 2)** -- real, working comparison.

## Question

Does training with explicit Chain-of-Thought reasoning improve text
classification accuracy over the same task/dataset/model without it?

## Protocol

1. Run the two decoder-only fake-news classification scripts with matched
   hyperparameters (same model, dataset, epochs, LoRA config -- only the CoT
   toggle differs):
   ```bash
   python ../../supervised-finetuning/text/classification/decoder-only/text_classification_no_cot.py \
       --max_samples 2000 --lora --output_dir ./output/supervised-finetuning/text/classification/decoder-only/no_cot
   python ../../supervised-finetuning/text/classification/decoder-only/text_classification_cot.py \
       --max_samples 2000 --lora --output_dir ./output/supervised-finetuning/text/classification/decoder-only/cot
   ```
2. Run `compare.py`, which loads both runs' `run_result.json` via
   `common/compare_runs.py` and prints a side-by-side table (accuracy,
   F1, precision, recall, CoT usage rate, average CoT length).

## What this does and doesn't tell you

Both scripts train on `domofon/fake_news_cot_reasoning` -- a fixed,
moderate-size corpus -- so the comparison is meaningful for *this* task and
*this* base model (`Qwen/Qwen3-1.7B-Base` by default), not a universal claim
about CoT. If both variants converge to similar accuracy, that could mean
CoT genuinely doesn't help for this classification task (it's binary
real/fake, which may not require much reasoning), or that more
training data/steps would surface a gap that a small `--max_samples` run
doesn't. The CoT-specific metrics (usage rate, average length) matter
independently of accuracy -- a model that learned to *always* emit a
`<think>` block but with degenerate/repetitive content would show a high
usage rate without genuinely better reasoning.
