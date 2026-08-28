# Other (storage)

Status: **done (Wave 6)** -- cross-referenced from `../execution-time/pruning/pruning_benchmark.py`, not a separate script.

This folder started as a genuine placeholder ("future techniques not yet
categorized") with no specific technique pre-assigned. Resolved the same
way as `../memory/README.md`: reporting the storage-axis measurement of
the one `other/` technique this project actually built -- structured
pruning -- rather than inventing an unrelated technique just to fill the
folder.

## Real result

Real on-disk checkpoint size, `./output/pretraining/clm` (dense) vs. the
same checkpoint after `../execution-time/pruning/pruning_benchmark.py`'s
50%-MLP-width structured pruning:

| | Checkpoint size |
|---|---|
| Dense | 204.3 MB |
| Pruned | 170.8 MB |
| **Reduction** | **16.4%** |

Tracks parameter-count reduction almost exactly (16.4% fewer parameters,
16.4% smaller on disk -- unlike quantization, which changes *bits per
parameter* rather than parameter *count*, so its disk-size reduction
doesn't track parameter count at all, see
`../../inference/storage/quantized-checkpoint-storage/README.md`).
Structured pruning and quantization are complementary levers on storage
for the same underlying reason distillation and quantization are
(`../../distillation/storage/README.md`): pruning removes parameters,
quantization represents the remaining ones with fewer bits -- the two
could in principle stack for a checkpoint smaller than either alone.
