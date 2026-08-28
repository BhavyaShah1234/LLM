# Domain Adaptation

Status: **done (Wave 3)**.

## What this stage is

Continued pretraining specialized to a target domain's text distribution --
here, **medical** text -- so downstream domain-specific finetuning (e.g.
`supervised-finetuning/text/mcq/`'s MedMCQA scripts) starts from a base
model that has already seen a lot of in-domain language, not just general
web/book text. Distinct from `continued-pretraining/` (this project's other
Wave-3 stage), which continues a model on *more general* text of the kind it
already saw, not a narrow domain.

`domain_adaptation.py` defaults to `Qwen/Qwen3-1.7B-Base` -- a real, capable
pretrained model, deliberately different from `continued-pretraining/`'s toy
from-scratch checkpoint, since domain specialization is most meaningfully
demonstrated starting from a model that already has broad language ability.
Because full-parameter CLM training of a 1.7B model does not fit in 8GB VRAM
(confirmed empirically, same finding as `supervised-finetuning/`'s decoder-only
scripts), `--lora` is the realistic default for real training runs here.

## Dataset

The **`Explanation`** field of **`araag2/MedMCQA`** (config `processed`) --
the same dataset `supervised-finetuning/text/mcq/decoder-only/mcq_standard.py`
uses, reused here as a plain medical-text corpus (its long clinical-reasoning
explanations, not the question/option/answer structure). About 11% of rows
have an empty `Explanation` (confirmed empirically by sampling 200 rows),
filtered out during ingestion. Splits: `train` (training), `dev` (eval).

## Usage

```bash
python domain_adaptation.py --debug_first_batch --max_samples 20
python domain_adaptation.py --lora --max_samples 2000
python domain_adaptation.py --lora --quantization 4bit --max_samples 5000  # more VRAM headroom
```

## Dataset Reference (this folder's slice of the root README's master table)

| Script | Dataset | Split(s) | Role |
|---|---|---|---|
| `domain_adaptation.py` | `araag2/MedMCQA` (config `processed`, `Explanation` field) | `train` (training), `dev` (eval) | primary + only corpus |
