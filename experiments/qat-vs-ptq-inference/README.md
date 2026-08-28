# QAT vs PTQ (Inference Accuracy)

Status: **done (Wave 4)**.

Question: does a model quantized only at inference time (PTQ) give the same
performance as a model trained with Quantization-Aware Training (QAT)? One
of this project's original motivating research questions.

`compare.py` reads five `run_result.json` files -- PTQ at 8-bit and 4-bit,
QAT at 8-bit and 4-bit, and a QAT `--skip_quantization` ablation (same
training budget, no quantization) -- and computes each method's *marginal*
quantization cost, controlling for the fact that QAT's extra training steps
help regardless of quantization. See
`optimization/training/memory/quantization-aware-training/README.md` for
the full write-up; short version: at 8-bit, quantization barely hurts this
project's toy 51M-param model, so QAT has nothing to recover (~0 advantage
over PTQ); at 4-bit, PTQ's degradation is real (+1.4 ppl) and QAT roughly
halves it (+0.6 ppl) for the same training budget.

## Usage

```bash
# Reproduce the underlying runs first (from repo root):
python optimization/inference/memory/post-training-quantization/ptq.py --num_bits 8 --max_eval_samples 2000
python optimization/inference/memory/post-training-quantization/ptq.py --num_bits 4 --max_eval_samples 2000 --output_dir ./output/optimization/ptq-4bit
python optimization/training/memory/quantization-aware-training/qat.py --num_bits 8 --max_samples 20000 --max_steps 200
python optimization/training/memory/quantization-aware-training/qat.py --num_bits 4 --max_samples 20000 --max_steps 200 --output_dir ./output/optimization/qat-4bit
python optimization/training/memory/quantization-aware-training/qat.py --skip_quantization --max_samples 20000 --max_steps 200 --output_dir ./output/optimization/qat-ablation-no-quant

# Then compare:
python compare.py
```
