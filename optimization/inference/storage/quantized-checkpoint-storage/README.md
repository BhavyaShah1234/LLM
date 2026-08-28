# Quantized Checkpoint Storage (inference, storage)

Status: **done (Wave 6)**.

`quantized_checkpoint_storage_benchmark.py` measures the REAL on-disk size
of a bitsandbytes-quantized checkpoint, complementing
`../../memory/post-training-quantization/ptq.py`'s
`estimated_size_mb_quantized` metric (a theoretical estimate, necessary
there because that script's fake-quantized tensors are rounded then
immediately dequantized back to float and never actually change dtype on
disk). This script verified first, rather than assumed, that bnb's
quantized weights aren't just an in-memory-only representation:
`model.save_pretrained()` on a 4-bit-loaded model genuinely writes the
quantized bytes to disk.

## Real result (`./output/pretraining/clm`, 51M params)

| Format | Size | vs. bf16 | Save time | Load time |
|---|---|---|---|---|
| bf16 (reference) | 105.7 MB | -- | 0.868s | 0.022s |
| bnb 8-bit | 80.7 MB | -23.7% | 0.063s | 0.182s |
| bnb 4-bit | 68.4 MB | -35.3% | 0.062s | 0.073s |

**4-bit's disk-size reduction is smaller than its ~46% GPU-memory
reduction found in `../../memory/quantization-4bit-8bit/README.md`, on
the same model family.** Real, and explained by this specific
checkpoint's architecture, not a discrepancy to be alarmed by: bnb
quantizes `nn.Linear` layers only, not embedding tables, and this
project's from-scratch CLM checkpoint has an unusually large embedding
share -- 50.6% of total parameters (documented in `pretraining/README.md`).
Roughly half this checkpoint's weight bytes are structurally exempt from
quantization regardless of bit-width, capping the achievable disk-size
reduction well below the naive 4x (16-bit -> 4-bit) or 2x (16-bit -> 8-bit)
ceiling a purely-Linear-layer model would approach. A model with a smaller
embedding-to-total ratio (larger hidden size relative to vocabulary, e.g.
`Qwen/Qwen3-1.7B-Base`) would show a reduction closer to that ceiling.

**4-bit loads faster than 8-bit from disk** (0.073s vs. 0.182s) --
consistent with the training-time finding in
`../../memory/quantization-4bit-8bit/README.md` that bnb's 4-bit (NF4)
kernel path is generally more optimized than its 8-bit path.

## Usage

```bash
python quantized_checkpoint_storage_benchmark.py --debug_first_batch
python quantized_checkpoint_storage_benchmark.py
```

No dataset needed -- this benchmark measures real file sizes and load
times of an already-trained checkpoint, not model quality.
