# Pretraining Objective Comparison

Status: **done (Wave 1)** -- the first real, working experiment.

## Question

For the same small corpus and a matched, small compute budget, how do the
three core pretraining objectives -- causal language modelling (decoder-only),
masked language modelling (encoder-only), and span corruption
(encoder-decoder) -- compare on final loss/perplexity, parameter count, and
training wall-clock?

## Protocol

1. Run the three scripts in `pretraining/` with matched settings (same
   corpus -- they all default to `roneneldan/TinyStories` -- and ideally the
   same `--max_steps` so the comparison reflects the same compute budget):
   ```bash
   python ../../pretraining/clm.py --max_steps 500 --output_dir ./output/pretraining/clm
   python ../../pretraining/mlm.py --max_steps 500 --output_dir ./output/pretraining/mlm
   python ../../pretraining/span_corruption.py --max_steps 500 --output_dir ./output/pretraining/span_corruption
   ```
2. Run `compare.py`, which loads each run's `run_result.json` via
   `common/compare_runs.py` and prints a side-by-side table.

## What this does and doesn't tell you

This is a toy-scale, single-GPU, from-scratch comparison on a small shared
corpus -- see `pretraining/README.md`'s toy-scale caveat. It's useful for
building intuition about how each objective trains (loss curve shape,
relative speed, parameter efficiency) under identical, small-budget
conditions. It is **not** a general claim about which architecture family is
"better" -- that depends heavily on scale, corpus, and downstream task, none
of which this toy comparison controls for.

Perplexity is directly comparable between `clm.py` and `span_corruption.py`
(both are true sequence/reconstruction perplexity). `mlm.py`'s reported
perplexity is a conventional pseudo-perplexity computed only over masked
positions (the standard proxy used in BERT-style training logs), so
cross-family perplexity comparisons involving `mlm.py` are approximate, not
exact.
