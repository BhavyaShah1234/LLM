# Pruning (other, execution-time)

Status: **done (Wave 6)**.

`pruning_benchmark.py` compares a dense checkpoint against a structurally
pruned version where each transformer block's MLP intermediate width is
physically shrunk (not just zeroed), on `./output/pretraining/clm` and the
same TinyStories perplexity evaluation `qat.py`/`ptq.py` use, for a
directly comparable degradation number across all three techniques.

## Real methodology decision: structured, not unstructured pruning

Verified before implementing, not assumed: `torch.nn.utils.prune`'s
unstructured pruning (`l1_unstructured`, what most tutorials show) zeros
individual weight *values* but doesn't shrink the underlying tensor. On
dense GPU hardware without a sparse-aware kernel, a 50%-zeroed dense
matmul costs the same FLOPs as an unpruned one -- **no real inference
speedup**, which contradicts this folder's own stub note that "inference
speedup is the headline benefit." Instead, this script performs genuine
structured pruning: ranking each MLP's intermediate channels by L1 norm,
keeping the top-k%, and physically constructing smaller `c_fc`/`c_proj`
weight matrices (GPT2's 4x-hidden-size MLP bottleneck) -- real matmuls
with real fewer FLOPs, not masked ones.

## Real result (`prune_ratio=0.5`, no recovery finetuning)

| | Dense | Pruned (50% of MLP width removed) |
|---|---|---|
| Parameters | 51,082,752 | 42,685,952 (-16.4%) |
| Perplexity | 60.21 | 70.81 (+10.61) |
| Inference latency (batch=8, seq_len=256) | 48.65 ms | 40.01 ms |
| **Speedup** | -- | **1.22x** |

A real, if modest, speedup (16.4% fewer parameters doesn't translate 1:1
to speedup, since attention layers and embeddings -- 50.6% of this
checkpoint's parameters, see `pretraining/README.md` -- are untouched by
this MLP-only pruning) at a real, meaningful perplexity cost. This
script deliberately does **not** include a recovery fine-tuning phase
after pruning (the stub this folder started from names "structured/
unstructured pruning + short recovery fine-tune" as the fuller technique)
-- the +10.61 perplexity degradation shown here is pruning's raw,
unmitigated cost, useful as an honest upper bound. A production pruning
pipeline would fine-tune the pruned model briefly afterward to recover
most of this loss; that's a natural follow-up run using this same pruned
checkpoint as a starting point, left as a documented next step rather than
built here.

## Usage

```bash
python pruning_benchmark.py --debug_first_batch --max_eval_samples 200
python pruning_benchmark.py --prune_ratio 0.5 --max_eval_samples 2000
```
