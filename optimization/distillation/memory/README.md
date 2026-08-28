# Distillation (memory) -- primary placement

Status: **done (Wave 6)**.

`distillation_benchmark.py` trains a small, randomly-initialized ("from
scratch," zero prior knowledge) ~2M-param ViT-Tiny student purely by
distilling from this project's own trained ViT-base teacher
(`supervised-finetuning/image/image_classification.py`'s output, 85.8M
params, 80.6%-at-training-time / 82.3%-on-this-run's-eval-slice CIFAR-10
accuracy). Standard combined loss (Hinton et al. 2015):
`alpha * KL(student_soft, teacher_soft) + (1-alpha) * CE(student, hard_labels)`.

This is the **primary placement** for distillation -- see
`../execution-time/README.md` and `../storage/README.md`, which
cross-reference this same script and run rather than duplicating it, per
this project's "one primary placement, cross-referenced elsewhere" policy
for techniques spanning multiple resource axes.

## Real result (`--max_samples 3000 --max_eval_samples 1000 --epochs 5`)

| | Teacher | Student (distilled) |
|---|---|---|
| Parameters | 85,806,346 | 1,967,434 (2.3% of teacher) |
| Accuracy | 0.823 | 0.263 |
| F1 (macro) | 0.819 | 0.177 |
| Inference latency | 474.8 ms/batch | 16.3 ms/batch (**29.1x faster**) |
| Checkpoint size | 1201.8 MB | 7.9 MB (**99.3% smaller**) |

**Honest reading, not oversold**: distillation transferred *some* real
signal -- 26.3% accuracy is well above the 10% random-guess baseline for
10 classes, and the loss curves show genuine (if modest) learning, not a
stuck/degenerate training run like the one found and fixed in
`supervised-finetuning/audio/README.md`. But at this toy-scale training
budget (3000 images, 5 epochs, a student with only 2.3% of the teacher's
capacity, trained from complete random initialization with no pretraining
at all), the student comes nowhere near teacher-level accuracy. This
matches the real literature on distillation: getting a from-scratch tiny
ViT close to a much larger teacher's accuracy (the DeiT paper's actual
approach) needs orders of magnitude more training data and epochs than
this toy-scale run uses. The result this benchmark actually demonstrates
well is the **efficiency side of the tradeoff** -- massive, real wins in
inference speed (29x) and storage (99.3% smaller) -- while being honest
that the accuracy side of the tradeoff requires a training budget this
project's 8GB-GPU toy-scale runs don't attempt to match.

## Usage

```bash
python distillation_benchmark.py --debug_first_batch --max_samples 200
python distillation_benchmark.py --max_samples 3000 --max_eval_samples 1000 --epochs 5
```

Requires `supervised-finetuning/image/image_classification.py` to have
been run first (the default `--teacher_dir`).

## Dataset Reference (this folder's slice of the root README's master table)

| Script | Dataset | Split(s) | Role |
|---|---|---|---|
| `distillation_benchmark.py` | `uoft-cs/cifar10` | `train` (training), `test` (eval) | same dataset as the teacher (`image_classification.py`) |
