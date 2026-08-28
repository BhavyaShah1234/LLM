# FlashAttention (training, execution-time)

Status: **done (Wave 5)**.

`flash_attention_benchmark.py` compares HF's `attn_implementation="eager"`
against `"sdpa"` (PyTorch's built-in fused flash-attention-family kernel)
on real forward+backward passes over synthetic batches, measuring wall-clock
step time and peak GPU memory.

## Real clarification found before implementing

The standalone `flash_attn` pip package (what `attn_implementation=
"flash_attention_2"` needs) is **not installed** in this project's venv --
confirmed via `import flash_attn` failing -- and it has no prebuilt wheel
for any Python version (source-only `.tar.gz`, see root README's
environment section), so adding it means compiling from source. This
script instead benchmarks against **PyTorch's own built-in SDPA fused
kernel** (`torch.backends.cuda.flash_sdp_enabled()` confirmed `True` on
this machine), which implements the same tiled, IO-aware attention
algorithm without the separate package -- and is also `transformers`' own
default `attn_implementation` for most models today, so eager-vs-SDPA is
the practically relevant comparison, not a downgraded substitute for
"real" flash attention.

## Real result (seq_len=256, batch_size=2, Qwen/Qwen3-1.7B-Base + LoRA)

| | avg step time | peak memory |
|---|---|---|
| `eager` | 0.279s | 6689.6 MB |
| `sdpa` | 0.265s | 6279.5 MB |

**SDPA: 1.05x speedup, 6.1% peak memory reduction.** Real, but modest at
this sequence length -- expected, since flash-attention's advantage is
specifically about avoiding the *quadratic* full attention-matrix
materialization, and at seq_len=256 that matrix (256×256 per head) is
still small relative to everything else in the forward pass (weights,
MLP activations, output logits).

## Real finding: both implementations OOM at seq_len=1024, for a reason
unrelated to attention

Retrying at seq_len=1024 (even down to `--batch_size 1`) OOM'd for **both**
`eager` and `sdpa` on this 8GB GPU with `Qwen/Qwen3-1.7B-Base`. Tracing the
actual failure point in each case showed this ISN'T primarily an
attention-implementation problem at this model scale:
- `eager` OOM'd inside its own attention softmax/matmul, as expected.
- `sdpa` OOM'd later, inside `nn.functional.cross_entropy` computing the
  training loss -- i.e. *after* attention had already succeeded. The
  culprit: `Qwen/Qwen3-1.7B-Base`'s 151,936-token vocabulary means the
  `[seq_len, vocab_size]` logits tensor alone is `1024 × 151936 × 4 bytes
  ≈ 622MB` (fp32, upcast for numerically-stable cross-entropy), before
  accounting for backward-pass copies.

**Takeaway**: at this model's vocabulary size, the output projection +
loss computation can become the binding memory constraint at longer
sequences, independent of which attention implementation is used --
flash/SDPA attention is necessary but not sufficient to unlock longer
context on 8GB hardware here. Fixing that specifically needs a different
technique (chunked or fused linear-cross-entropy loss, computing the loss
in vocabulary-sized chunks instead of materializing the full logits
tensor at once) that's out of scope for this benchmark -- worth flagging
as a real, non-obvious result rather than papering over the failed
seq_len=1024 run.

## Usage

```bash
python flash_attention_benchmark.py --debug_first_batch
python flash_attention_benchmark.py --batch_size 2 --seq_len 256 --num_steps 10
```

No dataset needed -- this benchmark measures compute/memory
characteristics of an attention implementation on synthetic random-token
batches, not model quality (unlike the QAT/PTQ scripts, which measure
accuracy and need real held-out data).
