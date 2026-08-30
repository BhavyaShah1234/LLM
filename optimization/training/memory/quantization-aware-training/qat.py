"""Quantization-Aware Training (QAT) -- isolated benchmark script.

Wraps every weight-bearing projection layer of a pretrained checkpoint in a
fake-quantization module (round-to-`num_bits`-then-dequantize, straight-
through estimator for the backward pass, per Jacob et al. 2018), then
CONTINUES TRAINING through that simulated quantization noise for a short
adaptation phase. The idea QAT tests: does letting the model see and adapt
to quantization error *during* training recover accuracy that naive
post-training quantization (see ../../inference/memory/post-training-quantization/ptq.py)
loses? Both scripts start from the exact same checkpoint and apply the exact
same fake-quant scheme so the only variable is "trained through it" vs not
-- that comparison is what experiments/qat-vs-ptq-inference/ reads back out.

Two things this script deliberately does NOT do, both worth being explicit
about:
  1. It doesn't use torch.ao.quantization's module-based APIs
     (`prepare_qat`/`convert`). Those key off module *type* (nn.Linear,
     nn.Conv2d, ...) via a backend qconfig mapping. GPT2's HF implementation
     uses `transformers.pytorch_utils.Conv1D` for every attention/MLP
     projection -- NOT nn.Linear -- so the built-in APIs would silently
     quantize nothing but the (tied) lm_head. Confirmed by inspecting
     `type(m).__name__` across the pretrained checkpoint's modules before
     writing this script. A hand-rolled fake-quant wrapper keyed on both
     types avoids that trap and stays backend-agnostic (plain autograd ops,
     runs on GPU -- torch.ao.quantization's real INT8 kernels are CPU-only
     via fbgemm/qnnpack, which would have forced this whole comparison onto
     CPU).
  2. It measures the ACCURACY dimension of quantization only -- weights are
     rounded to `num_bits` levels and immediately dequantized back to
     float, so the tensors on the GPU never actually shrink. Real memory/
     disk reduction from quantization is already exercised elsewhere in
     this project via bitsandbytes (`--quantization 4bit/8bit`, verified
     with real VRAM drops via nvidia-smi throughout supervised-finetuning/
     and rlhf/). This script instead
     isolates the question those flags can't answer: holding the
     quantization scheme fixed, does training through it help?

Usage:
    python qat.py --debug_first_batch --max_samples 200
    python qat.py --num_bits 8 --max_steps 200
    python qat.py --num_bits 4 --max_steps 200   # more aggressive, larger expected degradation
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
    """Build the CLI argument parser for this benchmark.

    Returns:
        argparse.ArgumentParser: Parser with all benchmark options (model,
        quantization bit-width, dataset sizing, training hyperparameters,
        seed, output dir, logging/debug flags) registered.
    """
    p = argparse.ArgumentParser(description="Quantization-Aware Training benchmark: fine-tune a pretrained checkpoint through simulated weight quantization.")
    p.add_argument("--model", type=str, default="./output/pretraining/clm", help="Pretrained checkpoint to start from (local dir or HF Hub id). Default: this project's own from-scratch CLM checkpoint.")
    p.add_argument("--num_bits", type=int, default=8, choices=[4, 8], help="Simulated quantization bit-width for weights. Default: 8.")
    p.add_argument("--skip_quantization", action="store_true", default=False, help="Ablation: run the identical training loop WITHOUT fake-quant wrapping. Isolates 'more training helped' from 'training through quantization helped' -- see README for why this matters (continuing training at all improves an undertrained checkpoint regardless of quantization).")

    p.add_argument("--block_size", type=int, default=256, help="Context length per packed training example. Must match --model's n_positions. Default: 256.")
    p.add_argument("--max_samples", type=int, default=-1, help="Number of raw training rows to use before packing. -1 (default) = use the full split.")
    p.add_argument("--sample_selection", type=str, default="random", choices=["random", "first", "last"], help="Which rows to pick when --max_samples is positive. Default: random.")
    p.add_argument("--max_eval_samples", type=int, default=2000, help="Number of validation rows for perplexity evaluation. Default: 2000.")

    p.add_argument("--batch_size", type=int, default=16, help="Per-step batch size. Default: 16.")
    p.add_argument("--gradient_accumulation_steps", type=int, default=2, help="Gradient accumulation steps. Default: 2.")
    p.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for the QAT adaptation phase -- lower than from-scratch pretraining since this is fine-tuning an already-trained checkpoint. Default: 1e-4.")
    p.add_argument("--max_steps", type=int, default=200, help="Optimizer steps for the QAT adaptation phase. Default: 200.")

    p.add_argument("--seed", type=int, default=42, help="Random seed. Default: 42.")
    p.add_argument("--output_dir", type=str, default="./output/optimization/qat", help="Where to write run_result.json. This benchmark script does not save model weights (see module docstring point 2).")
    p.add_argument("--logging_steps", type=int, default=20, help="Logging frequency. Default: 20.")

    p.add_argument("--debug_first_batch", action="store_true", default=False, help="Load data and model, wrap layers in fake-quant, print a sanity check, then exit without training.")
    return p


class FakeQuantSTE(torch.autograd.Function):
    """Round-to-`num_bits`-then-dequantize, straight-through estimator on
    the backward pass (gradient passes through unchanged -- the standard
    trick from Jacob et al. 2018 that makes quantization noise
    differentiable). Per-tensor symmetric scale; per-channel would reduce
    quantization error further but adds bookkeeping this illustrative
    benchmark doesn't need.
    """

    @staticmethod
    def forward(ctx, weight, num_bits):
        """Round `weight` to `num_bits` symmetric levels and dequantize back to float.

        Args:
            ctx (torch.autograd.function.FunctionCtx): Autograd context (unused; no tensors are saved since backward is a straight-through pass).
            weight (torch.Tensor): Weight tensor to fake-quantize.
            num_bits (int): Bit-width to simulate (per-tensor symmetric scale derived from `weight`'s max absolute value).

        Returns:
            torch.Tensor: `weight`, rounded to `num_bits` levels and rescaled back to the original range.
        """
        qmax = 2 ** (num_bits - 1) - 1
        qmin = -(2 ** (num_bits - 1))
        scale = weight.detach().abs().max().clamp(min=1e-8) / qmax
        quantized = torch.clamp(torch.round(weight / scale), qmin, qmax)
        return quantized * scale

    @staticmethod
    def backward(ctx, grad_output):
        """Pass the gradient through unchanged (straight-through estimator).

        Args:
            ctx (torch.autograd.function.FunctionCtx): Autograd context (unused).
            grad_output (torch.Tensor): Gradient w.r.t. this function's output.

        Returns:
            tuple[torch.Tensor, None]: `(grad_output, None)` -- the gradient
            passed straight through for `weight`, and `None` for the
            non-differentiable `num_bits` argument.
        """
        return grad_output, None


class FakeQuantLinear(nn.Module):
    """Drop-in replacement for `nn.Linear` that fake-quantizes its weight on every forward pass.

    Attributes:
        weight (torch.nn.Parameter): The original layer's weight Parameter (shared, not copied, to preserve weight tying).
        bias (torch.nn.Parameter): The original layer's bias Parameter.
        num_bits (int): Simulated quantization bit-width applied to `weight` each forward pass.
    """

    def __init__(self, orig: nn.Linear, num_bits: int):
        """Wrap an existing `nn.Linear` layer for fake quantization.

        Args:
            orig (nn.Linear): Layer to wrap; its weight/bias Parameters are reused directly (not cloned), which preserves weight tying (e.g. lm_head <-> embedding).
            num_bits (int): Simulated quantization bit-width.
        """
        super().__init__()
        self.weight = orig.weight  # same Parameter object -- preserves weight tying (lm_head <-> wte)
        self.bias = orig.bias
        self.num_bits = num_bits

    def forward(self, x):
        """Apply fake-quantized weights to a standard linear transform.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: `F.linear(x, fake_quantized_weight, self.bias)`.
        """
        w = FakeQuantSTE.apply(self.weight, self.num_bits)
        return F.linear(x, w, self.bias)


class FakeQuantConv1D(nn.Module):
    """Fake-quantized replacement for transformers.pytorch_utils.Conv1D --
    GPT2's attention/MLP projections, weight shape (in_features, out_features)
    (transposed relative to nn.Linear), forward via torch.addmm.

    Attributes:
        weight (torch.nn.Parameter): The original layer's weight Parameter (shared, not copied).
        bias (torch.nn.Parameter): The original layer's bias Parameter.
        nf (int): Output feature count, taken from the original `Conv1D` (needed to reshape the `addmm` output).
        num_bits (int): Simulated quantization bit-width applied to `weight` each forward pass.
    """

    def __init__(self, orig: Conv1D, num_bits: int):
        """Wrap an existing `Conv1D` layer for fake quantization.

        Args:
            orig (Conv1D): Layer to wrap; its weight/bias Parameters and `nf` are reused directly (not cloned).
            num_bits (int): Simulated quantization bit-width.
        """
        super().__init__()
        self.weight = orig.weight
        self.bias = orig.bias
        self.nf = orig.nf
        self.num_bits = num_bits

    def forward(self, x):
        """Apply fake-quantized weights to `Conv1D`'s `addmm`-based transform.

        Args:
            x (torch.Tensor): Input tensor of shape `(..., in_features)`.

        Returns:
            torch.Tensor: Output tensor of shape `(..., self.nf)`.
        """
        w = FakeQuantSTE.apply(self.weight, self.num_bits)
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), w)
        return x.view(size_out)


def apply_fake_quant(model: nn.Module, num_bits: int) -> int:
    """In-place: replace every nn.Linear / Conv1D child with a fake-quant
    wrapper around the SAME weight Parameter. Returns the count replaced.

    Args:
        model (nn.Module): Model to mutate in place.
        num_bits (int): Simulated quantization bit-width passed to each wrapper.

    Returns:
        int: Number of `nn.Linear`/`Conv1D` child modules replaced.
    """
    replaced = 0
    modules = list(model.modules())  # materialize first: safe to mutate children while iterating this list
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
    """Tokenize raw text rows and pack them into fixed-length CLM training blocks.

    Appends the tokenizer's EOS token to each text, tokenizes, concatenates
    all tokens, then chunks into contiguous `block_size`-token blocks
    (dropping any remainder shorter than `block_size`), with `labels` set
    equal to `input_ids`.

    Args:
        dataset (datasets.Dataset): Raw dataset with a `"text"` column.
        tokenizer (transformers.PreTrainedTokenizerBase): Tokenizer used to encode text and supply the EOS token.
        block_size (int): Length of each packed training block.
        desc (str): Short label used in the `datasets.Dataset.map` progress-bar description.

    Returns:
        datasets.Dataset: Packed dataset with `input_ids` and `labels` columns of length `block_size` each.
    """
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
    """Compute token-weighted average loss and perplexity over an eval set.

    Temporarily switches `model` to eval mode (restored to train mode
    before returning) and disables gradients while iterating.

    Args:
        model (transformers.PreTrainedModel): Model to evaluate, currently possibly fake-quantized.
        eval_dataset (datasets.Dataset): Packed eval dataset from `tokenize_and_pack`.
        data_collator (transformers.DataCollatorForLanguageModeling): Collator used to batch examples.
        batch_size (int): Evaluation batch size.
        device (str): Torch device to run evaluation on.

    Returns:
        tuple[float, float]: `(avg_loss, perplexity)` -- the token-weighted
        mean cross-entropy loss, and `exp(avg_loss)` (or `inf` if `avg_loss`
        is large enough that `exp` would overflow).
    """
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
    model.train()
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss) if avg_loss < 20 else float("inf")
    return avg_loss, perplexity


def main():
    """Run the QAT benchmark: baseline eval, fake-quantize, adapt, re-eval, and write a `run_result.json`.

    Parses CLI args, loads the model/tokenizer/dataset, evaluates
    full-precision baseline perplexity, applies fake quantization (or
    skips it under `--skip_quantization`), optionally exits early on
    `--debug_first_batch`, otherwise runs the QAT adaptation training loop,
    re-evaluates perplexity, and records the comparison via
    `common.run_results.write_run_result`.
    """
    args = build_arg_parser().parse_args()
    set_all_seeds(args.seed)
    print_config(args, "Quantization-Aware Training (QAT) benchmark")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = load_tokenizer(TOKENIZER_NAME)
    model = load_causal_lm(args.model, quantization_config=None, torch_dtype=torch.float32, gradient_checkpointing=False).to(device)

    print_banner("LOADING DATASET")
    raw_train = load_dataset(DATASET_NAME, split="train")
    raw_eval = load_dataset(DATASET_NAME, split="validation")
    raw_train = select_samples(raw_train, args.max_samples, args.sample_selection, args.seed)
    raw_eval = select_samples(raw_eval, args.max_eval_samples, "first", args.seed)
    train_dataset = tokenize_and_pack(raw_train, tokenizer, args.block_size, "train")
    eval_dataset = tokenize_and_pack(raw_eval, tokenizer, args.block_size, "eval")
    print(f"Train blocks: {len(train_dataset)}  |  Eval blocks: {len(eval_dataset)}")

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    print_banner("BASELINE EVALUATION (full precision, unquantized)")
    baseline_loss, baseline_ppl = evaluate_perplexity(model, eval_dataset, data_collator, args.batch_size, device)
    print(f"Baseline eval loss: {baseline_loss:.4f}  |  Baseline perplexity: {baseline_ppl:.2f}")

    total_param_count = sum(p.numel() for p in model.parameters())
    if args.skip_quantization:
        print_banner("SKIPPING QUANTIZATION (--skip_quantization ablation: plain continued training)")
        num_replaced = 0
        quantized_param_count = 0
    else:
        print_banner(f"APPLYING {args.num_bits}-BIT FAKE QUANTIZATION")
        num_replaced = apply_fake_quant(model, args.num_bits)
        quantized_param_count = sum(
            m.weight.numel() for m in model.modules() if isinstance(m, (FakeQuantLinear, FakeQuantConv1D))
        )
        print(f"Replaced {num_replaced} Linear/Conv1D modules ({quantized_param_count:,} / {total_param_count:,} params, "
              f"{100 * quantized_param_count / total_param_count:.1f}% of total)")

    if args.debug_first_batch:
        model.eval()
        with torch.no_grad():
            batch = data_collator([train_dataset[i] for i in range(min(2, len(train_dataset)))])
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
        print(f"Sanity forward pass through fake-quantized model: loss={out.loss.item():.4f}")
        print("--debug_first_batch set: exiting without training.")
        return

    print_banner("QAT TRAINING (adapting weights to quantization noise)")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=data_collator
    )
    model.train()
    step = 0
    start = time.time()
    optimizer.zero_grad()
    done = False
    while not done:
        for i, batch in enumerate(loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss / args.gradient_accumulation_steps
            loss.backward()
            if (i + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                step += 1
                if step % args.logging_steps == 0:
                    print(f"step {step}/{args.max_steps}  loss={outputs.loss.item():.4f}")
                if step >= args.max_steps:
                    done = True
                    break
    train_runtime_seconds = time.time() - start

    print_banner("FINAL EVALUATION (quantized weights)")
    qat_loss, qat_ppl = evaluate_perplexity(model, eval_dataset, data_collator, args.batch_size, device)
    print(f"QAT eval loss: {qat_loss:.4f}  |  QAT perplexity: {qat_ppl:.2f}")
    print(f"Perplexity degradation vs. full-precision baseline: {qat_ppl - baseline_ppl:+.2f}")

    estimated_size_mb_fp32 = quantized_param_count * 4 / 1e6
    estimated_size_mb_quantized = quantized_param_count * args.num_bits / 8 / 1e6
    estimated_compression_ratio = (estimated_size_mb_fp32 / estimated_size_mb_quantized) if quantized_param_count else None

    write_run_result(
        output_dir=args.output_dir,
        stage="optimization",
        task="quantization_aware_training",
        modality="text",
        architecture=ARCHITECTURE,
        model_name=args.model,
        dataset_name=DATASET_NAME,
        hyperparameters=vars(args),
        metrics={
            "baseline_eval_loss": baseline_loss,
            "baseline_perplexity": baseline_ppl,
            "quantized_eval_loss": qat_loss,
            "quantized_perplexity": qat_ppl,
            "perplexity_degradation": qat_ppl - baseline_ppl,
            "num_bits": args.num_bits,
            "num_quantized_modules": num_replaced,
            "quantized_param_count": quantized_param_count,
            "total_param_count": total_param_count,
            "estimated_size_mb_fp32": estimated_size_mb_fp32,
            "estimated_size_mb_quantized": estimated_size_mb_quantized,
            "estimated_compression_ratio": estimated_compression_ratio,
            "skip_quantization": args.skip_quantization,
        },
        num_train_samples=len(train_dataset),
        num_eval_samples=len(eval_dataset),
        train_runtime_seconds=train_runtime_seconds,
    )
    print("Done.")


if __name__ == "__main__":
    main()
