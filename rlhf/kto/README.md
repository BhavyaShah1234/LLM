# KTO

Status: **done (Wave 5)**.

Kahneman-Tversky Optimization -- aligns a model using UNPAIRED binary
feedback (a single (prompt, completion, label) row, label = desirable or
undesirable), unlike DPO's matched (chosen, rejected) pairs for the same
prompt. See the parent folder's README for the RLHF prerequisite, and root
README's "RLHF algorithm family" for the Kahneman-Tversky prospect-theory
motivation behind `--desirable_weight`/`--undesirable_weight`.

## Dataset

**`trl-lib/kto-mix-14k`** -- the standard TRL example dataset for KTO.
13,500 train / 1,500 test rows, `prompt`/`completion` (conversational
`[{"role","content"}]` lists) + `label` (bool). Balanced 50/50 on `label`
at full scale -- verified live before implementation.
`trl-lib/ultrafeedback_binarized` (used by `dpo.py`) is not reusable here:
KTO needs a label per individual (prompt, completion) row, not
chosen/rejected pairs.

## Real API constraint found (not documented up front, found via a live
`ValueError`)

`KTOTrainer` requires **actual per-device batch size > 1** --
`--batch_size 1` (even with high `--gradient_accumulation_steps`, which
works fine for DPO/GRPO) raises:

```
ValueError: Actual (not effective) batch size must be > 1. KTO will not
work properly because the KL term will be equivalent to the implied reward.
```

This is a real algorithmic constraint, not an arbitrary API limitation:
KTO estimates a KL term from the *actual* mini-batch by comparing each
example's reward against the batch's other examples, so a batch of 1 gives
a degenerate (zero-information) KL estimate. Fixed by using
`--batch_size 2 --gradient_accumulation_steps 4` (same effective batch
size of 8 as `dpo.py`'s default, just split differently) -- this still fits
the reference-model-doubled memory footprint on an 8GB GPU with
`--quantization 4bit --lora`.

## Real finding: small `--max_samples` truncation can skew label balance

At `--max_samples 40`, a real training run drew 27 desirable / 13
undesirable examples from a dataset that's balanced 50/50 at full scale --
random sampling from a small subset doesn't guarantee proportional
representation. `trl` itself surfaces this with a warning recommending
adjusted `--desirable_weight`/`--undesirable_weight` to compensate. Worth
knowing before trusting metrics from a small `--max_samples` smoke run:
imbalance here isn't a dataset bug, it's a real (if easily-triggered)
consequence of subsampling a small slice.

## Verified working run

```bash
python kto.py --lora --quantization 4bit --batch_size 2 --gradient_accumulation_steps 4 --max_length 256 --max_samples 40 --max_eval_samples 20
```

`eval_rewards/margins: 0.0147` (positive -- desirable examples get a
higher implied reward than undesirable ones, the correct direction),
`eval_kl: 0.503` (real, nonzero divergence from the reference model).

## Usage

```bash
python kto.py --debug_first_batch --max_samples 20
python kto.py --lora --quantization 4bit --batch_size 2 --gradient_accumulation_steps 4 --max_samples 2000   # batch_size must be > 1, see above
```

## Dataset Reference (this folder's slice of the root README's master table)

| Script | Dataset | Split(s) | Role |
|---|---|---|---|
| `kto.py` | `trl-lib/kto-mix-14k` | `train` (training), `test` (eval) | primary + only corpus |
