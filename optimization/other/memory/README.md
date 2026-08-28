# Other (memory)

Status: **done (Wave 6)** -- cross-referenced from `../execution-time/pruning/pruning_benchmark.py`, not a separate script.

This folder started as a genuine placeholder ("future techniques not yet
categorized") with no specific technique pre-assigned. Resolved by
reporting the memory-axis measurement of the one `other/` technique this
project actually built -- structured pruning -- rather than inventing an
unrelated technique just to fill the folder.

## Real result

Measured directly on `./output/pretraining/clm` (dense) vs. the same
checkpoint after `../execution-time/pruning/pruning_benchmark.py`'s
50%-MLP-width structured pruning, real forward+backward peak GPU memory
(batch_size=8, seq_len=256):

| | Peak memory (fwd+bwd) |
|---|---|
| Dense | 3002.1 MB |
| Pruned | 2886.8 MB |
| **Reduction** | **3.8%** |

A real but modest reduction -- much smaller than the 16.4% parameter-count
reduction the same pruning produced (see `../storage/README.md` for the
disk-size number, which DOES track parameter count closely). The gap is
expected: peak training memory at this batch/sequence scale is dominated
by activations and gradients across all layers, not just the pruned MLP
weights, so removing 16.4% of parameters doesn't remove 16.4% of total
GPU memory. Pruning's cleanest, most direct win is on-disk storage (see
`../storage/README.md`) and, more modestly, inference latency (see
`../execution-time/pruning/README.md`) -- not training-time memory,
which this measurement makes concrete rather than assumed.
