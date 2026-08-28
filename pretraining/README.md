# Pretraining

Status: **done (Wave 1)** -- 3 scripts, one per model architecture family.

## What this stage is

Pretraining is where a model first learns language (or vision, or any other
modality) from raw, unlabeled data, before any task-specific finetuning. The
three scripts here each train a small model **completely from scratch**
(random weight initialization, no downloaded checkpoint) using the objective
that corresponds to its architecture family:

| Script | Architecture | Objective | Attention pattern |
|---|---|---|---|
| `clm.py` | decoder-only (GPT/Llama/Qwen-style) | Causal Language Modelling: predict the next token given only the tokens before it | causal (each position sees only earlier positions) |
| `mlm.py` | encoder-only (BERT/DeBERTa-style) | Masked Language Modelling: predict randomly masked tokens given the *whole* surrounding context | bidirectional (every position sees every other position) |
| `span_corruption.py` | encoder-decoder (T5/BART-style) | Span corruption: mask contiguous spans with sentinel tokens, decoder reconstructs the dropped spans | encoder is bidirectional, decoder is causal + cross-attends to the encoder |

**Why these three objectives map to these three architectures**: causal
attention only makes sense with a causal *objective* (if the model could see
future tokens during MLM-style training, "predicting" a masked token would be
trivial and the model would learn nothing useful for generation). Bidirectional
attention only works when there's no autoregressive generation to protect --
which is exactly the encoder-only case. Encoder-decoder needs both: a
bidirectional encoder to build a rich representation of the (corrupted) input,
and a causal decoder to generate a coherent target sequence conditioned on it.

## Toy-scale, on purpose

These scripts run on a single 8GB laptop GPU, which means the models are
small (~10-50M non-embedding parameters -- each script prints its model's
exact parameter breakdown, including how much is embeddings vs. everything
else) and the training run is short relative to real pretraining runs (which
use billions of tokens and thousands of GPU-hours). **This teaches the CLM /
MLM / span-corruption mechanics correctly and lets you compare architecture
families' training dynamics on equal footing -- it does not produce a
competitive small language model.** Don't read too much into the absolute
perplexity numbers; the point is the mechanics and the *relative* comparison
across architectures (see `experiments/pretraining-objective-comparison/`).

One concrete consequence of "small": with a full-size tokenizer, the
embedding matrix (vocab_size × hidden_size) can dwarf the rest of the model's
parameters. E.g. GPT-2's ~50k-token vocabulary at hidden_size=512 is already
~25.7M embedding parameters -- comparable to or larger than the entire rest
of an 8-layer transformer at that size. Each script prints this breakdown
explicitly (`MODEL (FROM SCRATCH, RANDOM INIT)` banner) so it's visible, not
hidden. Training a smaller custom tokenizer instead of reusing an existing
one is a reasonable extension if you want to push non-embedding parameter
share higher, but it's out of scope here -- these scripts reuse an existing
open tokenizer per architecture family (GPT-2's for `clm.py`, BERT's for
`mlm.py`, T5's for `span_corruption.py`).

## Dataset

All three scripts use **[`roneneldan/TinyStories`](https://huggingface.co/datasets/roneneldan/TinyStories)**
(`train` + `validation` splits, single `text` field) -- a corpus of short,
simple synthetic children's stories purpose-built for training small language
models from scratch. It's small enough to download and hold in memory, and
its simple vocabulary/grammar means even a toy-scale model can learn
something coherent from it in a short run, which makes the mechanics easy to
inspect (`--debug_first_batch`) and the comparison across architectures fair
(same corpus, same rough parameter budget). Verified to load via
`load_dataset("roneneldan/TinyStories", split="train", streaming=True)` with
a `text` field, and confirmed to expose `train`/`validation` splits.

## Data preparation

`mlm.py` and `clm.py` tokenize each story, concatenate all stories in the
(possibly `--max_samples`-truncated) split into one long token stream, and
chop it into fixed-length blocks (`--block_size`, default 256) -- the
standard "packing" approach used in most from-scratch LM pretraining code, so
no padding is needed and every position in every training example is real
signal. `span_corruption.py` does the same packing step first, then applies
T5-style span corruption to each packed block (see the script for the
masking algorithm, adapted from the T5 paper's span-corruption procedure).

`mlm.py` deliberately **omits BERT's original Next-Sentence-Prediction
objective and explicit [CLS]/[SEP]-wrapped sentence pairs** in favor of
single-stream packed MLM -- this isn't a corner cut, it mirrors RoBERTa's
finding that NSP doesn't help and can be dropped.

## Shared CLI conventions specific to this stage

Pretraining scripts diverge from every later stage's CLI conventions in two
ways, both documented in the root README:
- **No `--quantization` / `--lora*` flags.** There's no pretrained base model
  to quantize or adapt -- every parameter is trained from a random init, so
  full-parameter training is the only mode.
- **No `--save_strategy` choice.** `common/model_saving.py`'s
  adapter/merged/base_reference distinction only makes sense once there's a
  pretrained base to adapt; these scripts always call
  `save_model(..., strategy="full")`.

Every script still supports the shared `--max_samples` (default `-1` = full
split) and `--sample_selection {random,first,last}` (default `random`)
convention, `--debug_first_batch` (prints real formatted examples and the
model's real parameter count, then exits without training), `--seed`, and
`--output_dir` (where the trained model, tokenizer, and `run_result.json` get
written).

## Dataset Reference (this folder's slice of the root README's master table)

| Script | Dataset | Split(s) | Role |
|---|---|---|---|
| `clm.py` | `roneneldan/TinyStories` | `train` (training), `validation` (eval) | primary + only corpus |
| `mlm.py` | `roneneldan/TinyStories` | `train`, `validation` | primary + only corpus |
| `span_corruption.py` | `roneneldan/TinyStories` | `train`, `validation` | primary + only corpus |

To swap the corpus for any of these scripts, change the `DATASET_NAME`
constant near the top of the file and re-verify its schema matches what
`verify_dataset()` expects (a `text` field, `train`/`validation` splits).

## Usage

```bash
# Quick sanity check: loads data, builds the model, prints formatted examples, exits
python clm.py --debug_first_batch --max_samples 20
python mlm.py --debug_first_batch --max_samples 20
python span_corruption.py --debug_first_batch --max_samples 20

# Short real training run
python clm.py --max_steps 500 --output_dir ./output/pretraining/clm
python mlm.py --max_steps 500 --output_dir ./output/pretraining/mlm
python span_corruption.py --max_steps 500 --output_dir ./output/pretraining/span_corruption
```

Each run writes its trained model + tokenizer + `run_result.json` to
`--output_dir`. See `experiments/pretraining-objective-comparison/` for a
script that compares all three runs' results side by side.
