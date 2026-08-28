"""Post-Training Quantization (PTQ) -- isolated benchmark script.

Applies the exact same fake-quantization scheme as
../../../training/memory/quantization-aware-training/qat.py to the exact
same starting checkpoint, but with ZERO additional training: weights are
quantized once and evaluated immediately. This is what "PTQ" means --
quantization applied to an already-finished checkpoint, no training loop
involved. See qat.py's module docstring for why the fake-quant scheme is
hand-rolled instead of using torch.ao.quantization (GPT2's projections are
`transformers.pytorch_utils.Conv1D`, not `nn.Linear` -- the built-in
module-type-keyed APIs would quantize almost nothing) and for the
weight-only-quantization caveat (this measures the accuracy effect of
quantization, not real memory reduction -- see the parent optimization/
README and rlhf/README for where
this project measures real VRAM savings via bitsandbytes instead).

This implementation weight-quantizes only (no activation quantization), so
unlike production PTQ pipelines (GPTQ, AWQ, ...) it needs no calibration
dataset -- the per-tensor scale is derived directly from each weight
tensor's own value range. Calibration-based activation quantization is out
of scope here; this comparison isolates a single variable (trained through
quantization noise vs not), which weight-only PTQ is sufficient to test.

Usage:
    python ptq.py --debug_first_batch --max_eval_samples 200
    python ptq.py --num_bits 8
    python ptq.py --num_bits 4   # more aggressive, larger expected degradation
"""

import argparse
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import DataCollatorForLanguageModeling
from transformers.pytorch_utils import Conv1D

from common.data_selection import select_samples
from common.logging_utils import print_banner, print_config
from common.model_loading import load_causal_lm, load_tokenizer
from common.run_results import write_run_result
from common.seeding import set_all_seeds

DATASET_NAME = "roneneldan/TinyStories"
TOKENIZER_NAME = "gpt2"
ARCHITECTURE = "decoder-only"


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Post-Training Quantization benchmark: quantize a finished checkpoint with no further training.")
    p.add_argument("--model", type=str, default="./output/pretraining/clm", help="Pretrained checkpoint to quantize (local dir or HF Hub id). Default: this project's own from-scratch CLM checkpoint -- same default as qat.py, for a like-for-like comparison.")
    p.add_argument("--num_bits", type=int, default=8, choices=[4, 8], help="Simulated quantization bit-width for weights. Default: 8.")
    p.add_argument("--block_size", type=int, default=256, help="Context length per packed eval example. Must match --model's n_positions. Default: 256.")
    p.add_argument("--max_eval_samples", type=int, default=2000, help="Number of validation rows for perplexity evaluation. Default: 2000.")
    p.add_argument("--batch_size", type=int, default=16, help="Eval batch size. Default: 16.")
    p.add_argument("--seed", type=int, default=42, help="Random seed. Default: 42.")
    p.add_argument("--output_dir", type=str, default="./output/optimization/ptq", help="Where to write run_result.json. This benchmark script does not save model weights.")
    p.add_argument("--debug_first_batch", action="store_true", default=False, help="Load data and model, quantize, print a sanity check, then exit without evaluating the full set.")
    return p


class FakeQuantSTE(torch.autograd.Function):
    """See qat.py -- identical scheme, duplicated here since it's the
    technique under study for both scripts, not shared plumbing. Backward
    (STE) is unused by this script (no training) but kept identical to
    qat.py so both scripts apply exactly the same forward-pass quantization.
    """

    @staticmethod
    def forward(ctx, weight, num_bits):
        qmax = 2 ** (num_bits - 1) - 1
        qmin = -(2 ** (num_bits - 1))
        scale = weight.detach().abs().max().clamp(min=1e-8) / qmax
        quantized = torch.clamp(torch.round(weight / scale), qmin, qmax)
        return quantized * scale

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class FakeQuantLinear(nn.Module):
    def __init__(self, orig: nn.Linear, num_bits: int):
        super().__init__()
        self.weight = orig.weight
        self.bias = orig.bias
        self.num_bits = num_bits

    def forward(self, x):
        w = FakeQuantSTE.apply(self.weight, self.num_bits)
        return F.linear(x, w, self.bias)


class FakeQuantConv1D(nn.Module):
    def __init__(self, orig: Conv1D, num_bits: int):
        super().__init__()
        self.weight = orig.weight
        self.bias = orig.bias
        self.nf = orig.nf
        self.num_bits = num_bits

    def forward(self, x):
        w = FakeQuantSTE.apply(self.weight, self.num_bits)
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), w)
        return x.view(size_out)


def apply_fake_quant(model: nn.Module, num_bits: int) -> int:
    replaced = 0
    modules = list(model.modules())
    for module in modules:
        for child_name, child in list(module.named_children()):
            if isinstance(child, nn.Linear):
                setattr(module, child_name, FakeQuantLinear(child, num_bits))
                replaced += 1
            elif isinstance(child, Conv1D):
                setattr(module, child_name, FakeQuantConv1D(child, num_bits))
                replaced += 1
    return replaced


def tokenize_and_pack(dataset, tokenizer, block_size: int, desc: str):
    def tokenize_fn(examples):
        texts_with_eos = [t + tokenizer.eos_token for t in examples["text"]]
        return tokenizer(texts_with_eos)

    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=dataset.column_names, desc=f"Tokenizing ({desc})")

    def group_texts(examples):
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = (len(concatenated["input_ids"]) // block_size) * block_size
        result = {k: [t[i : i + block_size] for i in range(0, total_length, block_size)] for k, t in concatenated.items()}
        result["labels"] = result["input_ids"].copy()
        return result

    return tokenized.map(group_texts, batched=True, desc=f"Packing into {block_size}-token blocks ({desc})")


def evaluate_perplexity(model, eval_dataset, data_collator, batch_size, device):
    model.eval()
    loader = torch.utils.data.DataLoader(eval_dataset, batch_size=batch_size, collate_fn=data_collator)
    total_loss, total_tokens = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            num_tokens = int((batch["labels"] != -100).sum().item())
            total_loss += outputs.loss.item() * num_tokens
            total_tokens += num_tokens
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss) if avg_loss < 20 else float("inf")
    return avg_loss, perplexity


def main():
    args = build_arg_parser().parse_args()
    set_all_seeds(args.seed)
    print_config(args, "Post-Training Quantization (PTQ) benchmark")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = load_tokenizer(TOKENIZER_NAME)
    model = load_causal_lm(args.model, quantization_config=None, torch_dtype=torch.float32, gradient_checkpointing=False).to(device)

    print_banner("LOADING DATASET")
    raw_eval = load_dataset(DATASET_NAME, split="validation")
    raw_eval = select_samples(raw_eval, args.max_eval_samples, "first", args.seed)
    eval_dataset = tokenize_and_pack(raw_eval, tokenizer, args.block_size, "eval")
    print(f"Eval blocks: {len(eval_dataset)}")

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    print_banner("BASELINE EVALUATION (full precision, unquantized)")
    baseline_loss, baseline_ppl = evaluate_perplexity(model, eval_dataset, data_collator, args.batch_size, device)
    print(f"Baseline eval loss: {baseline_loss:.4f}  |  Baseline perplexity: {baseline_ppl:.2f}")

    print_banner(f"APPLYING {args.num_bits}-BIT POST-TRAINING QUANTIZATION (no further training)")
    start = time.time()
    num_replaced = apply_fake_quant(model, args.num_bits)
    quantize_runtime_seconds = time.time() - start
    quantized_param_count = sum(
        m.weight.numel() for m in model.modules() if isinstance(m, (FakeQuantLinear, FakeQuantConv1D))
    )
    total_param_count = sum(p.numel() for p in model.parameters())
    print(f"Replaced {num_replaced} Linear/Conv1D modules ({quantized_param_count:,} / {total_param_count:,} params, "
          f"{100 * quantized_param_count / total_param_count:.1f}% of total) in {quantize_runtime_seconds:.3f}s")

    if args.debug_first_batch:
        model.eval()
        with torch.no_grad():
            batch = data_collator([eval_dataset[i] for i in range(min(2, len(eval_dataset)))])
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
        print(f"Sanity forward pass through fake-quantized model: loss={out.loss.item():.4f}")
        print("--debug_first_batch set: exiting without full evaluation.")
        return

    print_banner("FINAL EVALUATION (quantized weights, zero additional training)")
    ptq_loss, ptq_ppl = evaluate_perplexity(model, eval_dataset, data_collator, args.batch_size, device)
    print(f"PTQ eval loss: {ptq_loss:.4f}  |  PTQ perplexity: {ptq_ppl:.2f}")
    print(f"Perplexity degradation vs. full-precision baseline: {ptq_ppl - baseline_ppl:+.2f}")

    estimated_size_mb_fp32 = quantized_param_count * 4 / 1e6
    estimated_size_mb_quantized = quantized_param_count * args.num_bits / 8 / 1e6

    write_run_result(
        output_dir=args.output_dir,
        stage="optimization",
        task="post_training_quantization",
        modality="text",
        architecture=ARCHITECTURE,
        model_name=args.model,
        dataset_name=DATASET_NAME,
        hyperparameters=vars(args),
        metrics={
            "baseline_eval_loss": baseline_loss,
            "baseline_perplexity": baseline_ppl,
            "quantized_eval_loss": ptq_loss,
            "quantized_perplexity": ptq_ppl,
            "perplexity_degradation": ptq_ppl - baseline_ppl,
            "num_bits": args.num_bits,
            "num_quantized_modules": num_replaced,
            "quantized_param_count": quantized_param_count,
            "total_param_count": total_param_count,
            "estimated_size_mb_fp32": estimated_size_mb_fp32,
            "estimated_size_mb_quantized": estimated_size_mb_quantized,
            "estimated_compression_ratio": estimated_size_mb_fp32 / estimated_size_mb_quantized,
        },
        num_train_samples=0,
        num_eval_samples=len(eval_dataset),
        train_runtime_seconds=quantize_runtime_seconds,
    )
    print("Done.")


if __name__ == "__main__":
    main()
