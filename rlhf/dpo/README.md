# DPO

Status: **done (Wave 4)**.

Direct Preference Optimization -- see the parent folder's README for the
RLHF prerequisite and real memory/API findings, and root README's "RLHF
algorithm family" for the theory.

## Dataset

**`trl-lib/ultrafeedback_binarized`** (`train`: 62,135 rows, `test`: 1,000
rows) -- the standard TRL example dataset for DPO. Fields: `chosen`,
`rejected` (both conversational `[{"role","content"}]` lists, the exact
format `DPOTrainer` expects natively), `score_chosen`, `score_rejected`.

## Usage

```bash
python dpo.py --debug_first_batch --max_samples 20
python dpo.py --lora --quantization 4bit --max_samples 2000   # needed to fit 8GB VRAM, see parent README
```

## Dataset Reference (this folder's slice of the root README's master table)

| Script | Dataset | Split(s) | Role |
|---|---|---|---|
| `dpo.py` | `trl-lib/ultrafeedback_binarized` | `train` (training), `test` (eval) | primary + only corpus |
