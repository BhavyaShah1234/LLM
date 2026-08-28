# Quantization-Aware Training / QAT (training, memory)

Status: **done (Wave 4)**.

Fake-quantization modules (round-to-N-bits-then-dequantize, straight-through
estimator on the backward pass) wrapped around every weight-bearing
projection in the model, then the model is TRAINED through that simulated
quantization noise -- structurally different from the 4-bit/8-bit bnb
loading flag used elsewhere in this project (`--quantization 4bit/8bit`,
which quantizes a frozen base for memory savings; nothing adapts to it).
Cross-linked to `../../inference/memory/post-training-quantization/` (PTQ)
and `experiments/qat-vs-ptq-inference/` -- comparing these two is one of
this project's original motivating research questions, and that experiment
is now real (not a stub) using the runs described below.

See `qat.py`'s module docstring for two implementation notes worth reading
before touching this code:
1. Why the fake-quant wrapper is hand-rolled instead of using
   `torch.ao.quantization`'s module-based APIs (GPT2's HF implementation
   uses `transformers.pytorch_utils.Conv1D`, not `nn.Linear`, for every
   attention/MLP projection -- the built-in APIs key off module type and
   would silently quantize almost nothing).
2. Why this measures the ACCURACY effect of quantization only, not real
   memory reduction (fake-quantized weights are rounded then immediately
   dequantized back to float -- the tensors never actually shrink on the
   GPU). Real VRAM savings from quantization are already demonstrated
   elsewhere in this project via bitsandbytes.

## Real findings from real runs

All runs start from `./output/pretraining/clm` (this project's own
from-scratch 51M-param CLM checkpoint, Wave 1) and evaluate perplexity on
the same TinyStories validation split.

**The critical methodological finding**: naively comparing QAT's
post-training perplexity to PTQ's zero-training perplexity is confounded.
QAT's 200 additional optimizer steps improve perplexity a lot on their own
(this checkpoint was only trained 200 steps to begin with in Wave 1) --
*regardless* of quantization. To isolate the quantization-specific effect,
`qat.py` has a `--skip_quantization` ablation: the identical training loop,
same seed, same steps, but no fake-quant wrapper. That gives a same-budget
"more training, no quantization" reference to subtract out.

| Run | num_bits | Perplexity | vs. fp32 baseline (60.21) |
|---|---|---|---|
| fp32 baseline (no training, no quant) | -- | 60.21 | -- |
| PTQ (no training) | 8 | 60.21 | +0.00 |
| PTQ (no training) | 4 | 61.60 | +1.40 |
| QAT (200 steps training) | 8 | 40.81 | -19.40 |
| QAT (200 steps training) | 4 | 41.42 | -18.79 |
| **ablation**: 200 steps training, no quant | -- | 40.81 | -19.40 |

Isolating the quantization-only effect (`quantized_ppl - same_budget_reference`):

| Bit-width | PTQ's marginal cost | QAT's marginal cost | QAT advantage |
|---|---|---|---|
| 8-bit | +0.0017 | -0.0026 (noise) | ~0 |
| 4-bit | +1.396 | +0.605 | **+0.79 (QAT roughly halves it)** |

**Answer to "does QAT beat PTQ"**: it depends on how aggressive the
quantization is. At 8-bit (256 levels), this small model's weights are
already so tolerant of quantization that PTQ's degradation is
indistinguishable from noise -- QAT has nothing to recover, so it doesn't
"win." At 4-bit (16 levels), PTQ measurably hurts (+1.4 ppl) and training
through that noise for the same step budget roughly halves the damage
(+0.6 ppl). QAT's advantage is real, but it only shows up once quantization
is aggressive enough to actually cost something -- a genuinely useful,
non-obvious result that a naive (confounded) comparison would have
overstated at 8-bit and gotten right largely by accident at 4-bit.

## Usage

```bash
python qat.py --debug_first_batch --max_samples 200
python qat.py --num_bits 8 --max_samples 20000 --max_steps 200
python qat.py --num_bits 4 --max_samples 20000 --max_steps 200 --output_dir ./output/optimization/qat-4bit
python qat.py --skip_quantization --max_samples 20000 --max_steps 200 --output_dir ./output/optimization/qat-ablation-no-quant   # required reference run for the comparison above
```

## Dataset Reference (this folder's slice of the root README's master table)

| Script | Dataset | Split(s) | Role |
|---|---|---|---|
| `qat.py` | `roneneldan/TinyStories` | `train` (QAT adaptation), `validation` (eval) | same dataset `pretraining/clm.py` used, so the starting checkpoint's own training distribution isn't a confound |
