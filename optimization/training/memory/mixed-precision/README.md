# Mixed Precision (training, memory)

Status: **done (Wave 5)**.

`mixed_precision_benchmark.py` compares fp32, fp16, and bf16 on real
forward+backward passes over synthetic batches, measuring wall-clock step
time, peak GPU memory, and whether NaN losses appear. See root README's
Quantization & PEFT theory section for the fp16-vs-bf16 tradeoff (fp16 has
more mantissa precision but a narrower exponent range -- prone to
overflow/NaN on training with large activations; bf16 matches fp32's
exponent range at the cost of mantissa precision, which is why every
training script in this project defaults to bf16).

## Real finding that changed this script's default `--model`

The first version defaulted to `Qwen/Qwen3-1.7B-Base` with LoRA as memory
scaffolding (same pattern as
`../../execution-time/flash-attention/flash_attention_benchmark.py`). fp32
OOM'd anyway, even at `--batch_size 1 --seq_len 128`: fp32 weights alone
for a 1.7B model are ~6.8GB, leaving no headroom on this 8GB card
regardless of LoRA (LoRA freezes the base weights but doesn't change their
dtype or size). This is consistent with, not a violation of, this
project's model-selection philosophy -- it targets fp16/bf16 fitting in
8GB, not fp32. Fixed by defaulting to this project's own from-scratch CLM
checkpoint (`./output/pretraining/clm`, 51M params) instead, where
full-parameter fine-tuning fits comfortably in all three dtypes -- which
also meant dropping the LoRA scaffolding entirely (unneeded at this scale,
and GPT2-style `Conv1D` layers don't match `common/peft_setup.py`'s
default LoRA target module names anyway, the same naming mismatch
documented in `../quantization-aware-training/qat.py`).

## Real result (batch_size=8, seq_len=256, `./output/pretraining/clm`)

| dtype | avg step time | peak memory | NaN seen |
|---|---|---|---|
| fp32 | 0.1615s | 2836.9 MB | no |
| fp16 | 0.0640s | 2054.9 MB | no |
| bf16 | 0.0645s | 2054.9 MB | no |

**fp16: 2.52x speedup, 27.6% peak memory reduction. bf16: 2.50x speedup,
27.6% peak memory reduction (essentially identical to fp16).** Both halves
land within noise of each other, as expected -- they're the same bit-width,
differing only in exponent/mantissa split, which doesn't show up in raw
speed/memory. No NaN losses at this scale/duration for either -- consistent
with fp16's known failure mode being a risk that grows with training scale
and duration, not something guaranteed to appear in a short benchmark run
on a small model. This project's scripts still default to bf16 everywhere
for the training-stability margin, not because this specific benchmark
showed a speed/memory difference between the two.

## Usage

```bash
python mixed_precision_benchmark.py --debug_first_batch
python mixed_precision_benchmark.py --batch_size 8 --seq_len 256 --num_steps 10
```

No dataset needed -- see `../../execution-time/flash-attention/README.md`
for why synthetic batches are appropriate for a compute/memory benchmark
like this one.
