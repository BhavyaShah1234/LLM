# Distillation (execution-time)

Status: **done (Wave 6)** -- cross-referenced from `../memory/distillation_benchmark.py`, not a separate script.

Teacher-student training procedure; this axis folder tracks the student
model's execution-time characteristics relative to the teacher. Real
result from the same distillation run documented in
`../memory/README.md`:

| | Teacher | Student (distilled) |
|---|---|---|
| Inference latency (single batch, GPU) | 474.8 ms | 16.3 ms |
| **Speedup** | -- | **29.1x** |

This is the largest real, uncomplicated speedup measured in this
project's `optimization/` folder next to `../../inference/execution-time/kv-cache-paged-attention/`'s
17.8x -- and the most intuitive one: a ~2.3%-of-teacher-size model doing
~2.3%-of-teacher's compute per forward pass tracks roughly with the
observed 29x latency reduction (fewer FLOPs per layer × fewer layers).
See `../memory/README.md` for the full run (dataset, hyperparameters,
accuracy tradeoff, and why this speedup came at a real accuracy cost this
toy-scale training budget didn't try to close).
