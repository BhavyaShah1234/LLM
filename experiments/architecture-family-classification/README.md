# Architecture Family Classification

Status: **done (Wave 2)** -- real, working comparison.

## Question

For the same text classification task/dataset, how do decoder-only,
encoder-only, and encoder-decoder architectures compare on accuracy,
parameter count, and training wall-clock?

## Protocol

1. Run all three `text_classification_standard.py` scripts (same dataset,
   `Tohrumi/glue_sst2_10k`):
   ```bash
   python ../../supervised-finetuning/text/classification/decoder-only/text_classification_standard.py --max_samples 2000
   python ../../supervised-finetuning/text/classification/encoder-only/text_classification_standard.py --max_samples 2000
   python ../../supervised-finetuning/text/classification/encoder-decoder/text_classification_standard.py --max_samples 2000
   ```
2. Run `compare.py`, which loads all three runs' `run_result.json` via
   `common/compare_runs.py` and prints a side-by-side table.

## What this does and doesn't tell you

The three scripts default to different base models at different scales
(`Qwen/Qwen3-1.7B-Base`, `answerdotai/ModernBERT-base` ~150M params,
`t5-base` ~223M params) and different training regimes (decoder-only
optionally LoRA-adapted; encoder-only/encoder-decoder full-parameter
finetuned, since they're already small). This is therefore a comparison of
each architecture family's *typical* off-the-shelf choice for this task
size class, not a strictly parameter-matched ablation isolating architecture
alone. Still meaningful: it directly shows the practical tradeoffs a
practitioner faces (a 150M-param encoder-only classifier trains much faster
and needs no generation-based eval, vs. a 1.7B decoder-only model that can
also handle CoT/instruction-following if needed later).
