# Experiments

Cross-run research questions -- the comparisons that motivate this whole
project (e.g. "does CoT help?", "does QAT beat PTQ?", "does RLHF need an
SFT'd starting point?"). Kept separate from the per-stage training scripts
because an experiment's job is to *read* multiple runs' `run_result.json`
outputs and produce a comparison, not to train anything itself.

## How an experiment works

Each subfolder is one research question, with:
- a `README.md` documenting the question and the protocol (which scripts to
  run, with which flags, in what order) to reproduce the comparison, and
- a `compare.py` that loads the relevant runs' `run_result.json` files via
  `common/compare_runs.py` (`load_run_results` + `print_comparison_table`)
  and prints a comparison table, plus a short written note on what the
  comparison does and doesn't tell you.

`compare.py` scripts never train anything -- they assume the runs they
compare have already been produced by the relevant stage's scripts.

## Status

| Experiment | Status | Depends on |
|---|---|---|
| `pretraining-objective-comparison/` | **done (Wave 1)** | `pretraining/{clm,mlm,span_corruption}.py` (Wave 1) |
| `cot-vs-no-cot-classification/` | **done (Wave 2)** | `supervised-finetuning/text/classification/decoder-only/{no_cot,cot}.py` (Wave 2) |
| `architecture-family-classification/` | **done (Wave 2)** | `supervised-finetuning/text/classification/{decoder-only,encoder-only,encoder-decoder}/text_classification_standard.py` (Wave 2) |
| `rlhf-pretrained-vs-sft-init/` | **done (Wave 4)** | `rlhf/grpo/` (Wave 4, done) |
| `qat-vs-ptq-inference/` | **done (Wave 4)** | `optimization/{training/memory/quantization-aware-training,inference/memory/post-training-quantization}/` (Wave 4, done) |
| `quantized-training-vs-quantized-inference/` | **done (Wave 5)** | `supervised-finetuning/text/mcq/decoder-only/mcq_standard.py` (done) |
| `lora-vs-full-domain-adaptation/` | **planned** | `domain-adaptation/domain_adaptation.py` (done), `supervised-finetuning/text/mcq/decoder-only/mcq_standard.py` (done) |

See each subfolder's `README.md` for its specific question and protocol.
