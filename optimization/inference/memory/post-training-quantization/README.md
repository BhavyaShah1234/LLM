# Post-Training Quantization / PTQ (inference, memory)

Status: **done (Wave 4)**.

Applies the identical fake-quantization scheme as
`../../../training/memory/quantization-aware-training/qat.py` to a
*finished* checkpoint, with zero additional training -- the defining
difference from QAT. Cross-linked to `training/memory/quantization-aware-training/`
(QAT) and `experiments/qat-vs-ptq-inference/`, which is now real (not a
stub); see the QAT README for the full real-run findings and the
methodology used to make the QAT-vs-PTQ comparison fair (a matched-
training-budget ablation is required to isolate the quantization-only
effect -- comparing this script's zero-training result directly against
QAT's post-training result would be confounded by "more training helps
regardless of quantization").

This implementation weight-quantizes only (no activation quantization), so
unlike production PTQ pipelines (GPTQ, AWQ, ...) it needs no calibration
dataset -- see `ptq.py`'s module docstring.

## Real findings

On this project's own 51M-param CLM checkpoint (`./output/pretraining/clm`,
TinyStories validation split, 1519 eval blocks):

| num_bits | fp32 baseline ppl | PTQ ppl | Degradation |
|---|---|---|---|
| 8 | 60.2057 | 60.2074 | +0.0017 (noise-level) |
| 4 | 60.2057 | 61.6018 | +1.3961 |

8-bit weight-only PTQ is genuinely near-lossless for this model -- verified
directly (not assumed) by probing a single weight tensor's quantization
error in isolation (max abs error 0.0004, ~1.2% relative, 231/256 levels
used) and confirming the resulting loss shift on a real batch is ~6e-5,
below 4-decimal display precision. 4-bit (16 levels) is where quantization
starts to visibly cost something, which is exactly why the QAT-vs-PTQ
comparison in the parent README uses 4-bit to show QAT's advantage --
comparing at 8-bit alone would have made QAT look like it "doesn't help"
when really there was nothing to recover at that bit-width on this model.

## Usage

```bash
python ptq.py --debug_first_batch --max_eval_samples 200
python ptq.py --num_bits 8 --max_eval_samples 2000
python ptq.py --num_bits 4 --max_eval_samples 2000 --output_dir ./output/optimization/ptq-4bit
```

## Dataset Reference (this folder's slice of the root README's master table)

| Script | Dataset | Split(s) | Role |
|---|---|---|---|
| `ptq.py` | `roneneldan/TinyStories` | `validation` (eval only -- no training split needed) | same checkpoint/dataset as `qat.py` for a like-for-like comparison |
