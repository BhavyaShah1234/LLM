# Continued Pretraining

Status: **done (Wave 3)**.

## What this stage is

Continues training an **already-pretrained** checkpoint with the same
objective it was originally trained with, on more or different general-
purpose text -- as opposed to `domain-adaptation/` (this project's other
Wave-3 stage), which specializes a model to a *specific* domain instead.

`continued_pretraining.py` defaults to continuing this project's own
`pretraining/clm.py` output (`./output/pretraining/clm`) on
**`iohadrubin/wikitext-103-raw-v1`** -- real Wikipedia prose, a genuinely
different and harder text distribution than the short, simple-vocabulary
synthetic stories (`roneneldan/TinyStories`) that toy model was originally
pretrained on. This is the cleanest way to demonstrate the concept: same
model, same objective (CLM), new data.

`--model` also accepts any causal-LM checkpoint (local path or HF Hub id),
so this script works just as well continuing pretraining of a real model
like `Qwen/Qwen3-1.7B-Base` -- in which case `--lora`/`--quantization`
become relevant for the same reason they do in `supervised-finetuning/`
(full-parameter training of a 1.7B model doesn't fit in 8GB VRAM, confirmed
empirically).

## Dataset

**`iohadrubin/wikitext-103-raw-v1`** (`train`/`validation`/`test` splits,
single `text` field, one row per article) -- a namespaced community mirror
of the classic wikitext-103 corpus. The original un-namespaced `wikitext`
dataset repo does **not** resolve correctly against this project's
`datasets==5.0.1` pin (`HfUriError: Repository id must be 'namespace/name'`,
confirmed empirically), so this mirror is used instead. Its `validation`
split is much smaller than the classic per-line wikitext-103 splits (60 rows
here, since this mirror groups each split by whole article rather than by
line) -- this is what surfaced a real, general bug (see below).

## A real bug found (and fixed generally, not just here)

Smoke testing this script with the default `--max_eval_samples 2000` against
`iohadrubin/wikitext-103-raw-v1`'s 60-row validation split crashed with
`IndexError: Index 1999 out of range for dataset of size 60`.
`common/data_selection.py`'s `select_samples()` didn't clamp `max_samples`
to the dataset's actual size -- fixed there (not just in this script), since
every script in this project uses the same function and any dataset mirror
with a smaller-than-expected split could hit the same crash.

## Usage

```bash
# Continue this project's own toy pretrained model on new general text
python continued_pretraining.py --debug_first_batch --max_samples 20
python continued_pretraining.py --max_steps 500

# Continue a real pretrained model instead (needs LoRA to fit 8GB VRAM)
python continued_pretraining.py --model Qwen/Qwen3-1.7B-Base --lora --quantization 4bit --max_samples 2000
```

## Dataset Reference (this folder's slice of the root README's master table)

| Script | Dataset | Split(s) | Role |
|---|---|---|---|
| `continued_pretraining.py` | `iohadrubin/wikitext-103-raw-v1` | `train` (training), `validation` (eval) | primary + only corpus |
