# Distillation (storage)

Status: **done (Wave 6)** -- cross-referenced from `../memory/distillation_benchmark.py`, not a separate script.

Teacher-student training procedure; this axis folder tracks the student
model's storage characteristics relative to the teacher. Real result from
the same distillation run documented in `../memory/README.md`:

| | Teacher | Student (distilled) |
|---|---|---|
| Checkpoint size on disk | 1201.8 MB | 7.9 MB |
| **Reduction** | -- | **99.3% smaller** |

Unlike quantization (`../../training/memory/quantization-4bit-8bit/`,
`../../inference/storage/quantized-checkpoint-storage/`), which shrinks a
checkpoint by representing the SAME parameters with fewer bits per value,
distillation shrinks it by training a model with genuinely FEWER
parameters (2.3% of the teacher's count here) -- the two techniques are
complementary, not competing, and could in principle stack (a distilled
student could itself be quantized for further reduction). See
`../memory/README.md` for the full run (dataset, hyperparameters, and the
real accuracy tradeoff behind this size reduction).
