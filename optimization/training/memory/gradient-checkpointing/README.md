# Gradient Checkpointing (training, memory)

Status: **done (Wave 5)**.

`gradient_checkpointing_benchmark.py` compares training with vs. without
gradient checkpointing on real forward+backward passes, measuring
wall-clock step time and peak GPU memory. Unlike every other benchmark
built in this wave (SDPA attention, fp16/bf16 -- see the sibling READMEs
in `../../execution-time/flash-attention/` and `../mixed-precision/`),
gradient checkpointing is a **memory-for-time trade**, not a free win: it
discards most layers' activations during the forward pass and recomputes
them during backward instead of storing them, so peak memory drops but
each step does roughly one extra forward pass worth of compute.

## Real result (batch_size=16, seq_len=256, `./output/pretraining/clm`)

| | avg step time | peak memory |
|---|---|---|
| without checkpointing | 0.1382s | 3990.7 MB |
| with checkpointing | 0.1559s | 3051.4 MB |

**1.13x slower, 23.5% peak memory reduction.** A clean, real demonstration
of the intended trade: real time cost, real memory saving, in the expected
direction and a plausible magnitude for a small (8-layer) model.

## Real finding: this benchmark is more OOM-prone at larger batch sizes than expected, and not because of activations

Attempts at `--batch_size 32` and `--batch_size 64` (to make the
activation-memory effect even more pronounced) both OOM'd -- but tracing
the failures showed the dominant cost wasn't layer activations (gradient
checkpointing's actual target) but the `[batch_size, seq_len, vocab_size]`
logits tensor HF upcasts to fp32 for numerically-stable cross-entropy loss
computation (`GPT2LMHeadModel`'s `vocab_size=50257` here). At
`batch_size=32, seq_len=256`, that tensor alone is
`32 x 256 x 50257 x 4 bytes ~= 1.65GB`, before backward-pass copies -- and
it's identical whether or not gradient checkpointing is active, since
checkpointing only affects intermediate *layer* activations, not the final
loss computation. This is the same root cause documented in
`../../execution-time/flash-attention/README.md`'s seq_len=1024 OOM
finding: at typical vocabulary sizes, the loss/logits tensor can be the
actual binding memory constraint, independent of whichever technique is
under test. Kept at `--batch_size 16` for a benchmark that reliably
succeeds; a real memory-per-technique comparison at larger batch sizes
would need a chunked/fused cross-entropy implementation to remove this
confound first.

## Usage

```bash
python gradient_checkpointing_benchmark.py --debug_first_batch
python gradient_checkpointing_benchmark.py --batch_size 16 --seq_len 256 --num_steps 10
```

No dataset needed -- see `../../execution-time/flash-attention/README.md`
for why synthetic batches are appropriate for a compute/memory benchmark
like this one.
