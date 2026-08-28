# Quantized-Training vs Quantized-Inference

Status: **done (Wave 5)**.

Question: is the performance of a model fine-tuned WITHOUT quantization
and then quantized for inference the same as a model fine-tuned WITH
quantization (QLoRA) from the start? Subtly different from
`experiments/qat-vs-ptq-inference/` -- that one compares quantization
*schemes* (QAT vs. PTQ, weight-only fake-quant) on a from-scratch toy
model; this one compares quantized *training* (QLoRA) against quantizing
an already-adapter-trained model for inference, both real LoRA finetuning
runs on a real downstream task (MedMCQA MCQ accuracy).

## Setup

Three runs, all evaluating the same 40 MedMCQA dev rows (`--seed 42`):

```bash
# Reference: bf16 training, bf16 eval
python supervised-finetuning/text/mcq/decoder-only/mcq_standard.py --lora --quantization no --max_samples 150 --max_eval_samples 40 --batch_size 4 --gradient_accumulation_steps 4 --epochs 2 --output_dir ./output/experiments/quantized-training-vs-quantized-inference/train-bf16

# Quantized training (QLoRA): 4-bit training, 4-bit eval
python supervised-finetuning/text/mcq/decoder-only/mcq_standard.py --lora --quantization 4bit --max_samples 150 --max_eval_samples 40 --batch_size 4 --gradient_accumulation_steps 4 --epochs 2 --output_dir ./output/experiments/quantized-training-vs-quantized-inference/train-qlora

# Quantized inference of the bf16-trained adapter: re-load base in 4-bit, re-attach the same adapter
python experiments/quantized-training-vs-quantized-inference/quantize_and_eval.py

python experiments/quantized-training-vs-quantized-inference/compare.py
```

## Real result

| Arm | Accuracy | vs. bf16 reference |
|---|---|---|
| bf16-trained, bf16-eval (reference) | 0.4250 | -- |
| bf16-trained, **4-bit-eval** (quantized after the fact) | 0.3500 | **-0.075** |
| **4-bit-trained (QLoRA)**, 4-bit-eval (quantized from the start) | 0.4750 | **+0.050** |

**Answer: no, they are not the same.** Quantizing a bf16-trained adapter's
base model after the fact loses real accuracy relative to the bf16
reference (-0.075) -- the adapter was never exposed to quantization noise
during training, so it has no mechanism to compensate for it. Training
WITH quantization from the start (QLoRA) doesn't have this problem: its
adapter learns against the same quantized base it will be evaluated with,
and on this run it even beat the bf16 reference (+0.050). This is a
small-sample toy-scale result (150 train / 40 eval rows) -- treat the
exact magnitude as illustrative, not a precise estimate -- but the
*direction* (train-time quantization exposure beats post-hoc quantization)
is the mechanistically expected one and matches the standard motivation
for preferring QLoRA over "finetune then quantize."

Note the LoRA adapter weights themselves are never quantized in any of the
three arms -- bitsandbytes' 4-bit quantization applies only to the frozen
base `nn.Linear` weights. What differs across arms is only whether the
*base* weights the adapter was trained against were quantized during
training.

## Usage

```bash
python quantize_and_eval.py --debug_first_batch
python quantize_and_eval.py --adapter_dir ./output/experiments/quantized-training-vs-quantized-inference/train-bf16
python compare.py
```

## Dataset Reference (this folder's slice of the root README's master table)

| Script | Dataset | Split(s) | Role |
|---|---|---|---|
| `quantize_and_eval.py` | `araag2/MedMCQA` (config `processed`) | `dev` (eval only) | same split/prompt-format as `mcq_standard.py`/`grpo.py`, needed for a like-for-like comparison |
