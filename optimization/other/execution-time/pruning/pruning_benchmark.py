"""Structured Pruning (MLP width) -- isolated benchmark script (other,
execution-time -- primary placement, memory/storage are secondary
benefits reported alongside, per this folder's own stub note).

Compares a dense checkpoint against a structurally-pruned version where
each transformer block's MLP intermediate width (`c_fc`'s output /
`c_proj`'s input, GPT2's 4x-hidden-size bottleneck) is physically
shrunk -- not just zeroed.

**Why structured, not the more common unstructured (weight-magnitude)
pruning tutorial**: verified before implementing, not assumed --
`torch.nn.utils.prune`'s unstructured pruning (`l1_unstructured`, the
version most tutorials show) sets individual weight VALUES to zero but
does NOT shrink the underlying tensor. On dense GPU hardware without a
sparse-aware kernel, a 50%-zeroed dense matmul costs the same FLOPs as an
unpruned one -- no real inference speedup, only a (real, but different)
storage/compression opportunity if the zeros are exploited later (e.g. via
sparse serialization). Since this folder's stated headline benefit is
inference *speed*, this script instead performs real structured pruning:
ranking each MLP's intermediate channels by L1 norm, keeping the top-k%,
and constructing genuinely smaller `c_fc`/`c_proj` weight matrices --
which DOES reduce real FLOPs and produces a measurable speedup.

Uses this project's own from-scratch CLM checkpoint
(`./output/pretraining/clm`) and the same TinyStories perplexity
evaluation as `../../training/memory/quantization-aware-training/qat.py`
/ `../../inference/memory/post-training-quantization/ptq.py`, for a
directly comparable "how much does this technique hurt perplexity"
number across all three.

Usage:
    python pruning_benchmark.py --debug_first_batch --max_eval_samples 200
    python pruning_benchmark.py --prune_ratio 0.5
"""

import argparse
import copy
import math
import time

import torch
import torch.nn as nn
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
    p = argparse.ArgumentParser(description="Structured MLP-width pruning benchmark: real inference speedup vs. perplexity cost.")
    p.add_argument("--model", type=str, default="./output/pretraining/clm", help="Model to prune. Default: this project's own from-scratch CLM checkpoint.")
    p.add_argument("--prune_ratio", type=float, default=0.5, help="Fraction of each MLP's intermediate channels to REMOVE (not just zero). Default: 0.5 (keep the top 50%% by L1 norm).")
    p.add_argument("--block_size", type=int, default=256, help="Context length per packed eval example. Default: 256.")
    p.add_argument("--max_eval_samples", type=int, default=2000, help="Number of validation rows for perplexity evaluation. Default: 2000.")
    p.add_argument("--batch_size", type=int, default=16, help="Eval batch size. Default: 16.")
    p.add_argument("--inference_batch_size", type=int, default=8, help="Batch size for the inference-speed measurement. Default: 8.")
    p.add_argument("--inference_seq_len", type=int, default=256, help="Sequence length for the inference-speed measurement. Default: 256.")
    p.add_argument("--num_inference_repeats", type=int, default=20, help="Repeats for the inference-speed measurement. Default: 20.")
    p.add_argument("--seed", type=int, default=42, help="Random seed. Default: 42.")
    p.add_argument("--output_dir", type=str, default="./output/optimization/pruning_benchmark", help="Where to write run_result.json.")
    p.add_argument("--debug_first_batch", action="store_true", default=False, help="Prune the model, print the new parameter count, run one forward pass, then exit.")
    return p


def prune_mlp_width(model: nn.Module, prune_ratio: int) -> int:
    """GPT2-style blocks: mlp.c_fc (hidden -> 4*hidden, Conv1D, weight
    shape (in, out)) then mlp.c_proj (4*hidden -> hidden). Ranks c_fc's
    OUTPUT channels (= c_proj's INPUT channels) by L1 norm, keeps the
    top (1 - prune_ratio) fraction, and rebuilds both layers with
    genuinely smaller weight tensors -- not masked, physically smaller.
    """
    total_removed = 0
    for block in model.transformer.h:
        mlp = block.mlp
        c_fc, c_proj = mlp.c_fc, mlp.c_proj
        assert isinstance(c_fc, Conv1D) and isinstance(c_proj, Conv1D)

        intermediate_size = c_fc.nf
        keep_size = max(1, int(intermediate_size * (1 - prune_ratio)))
        channel_importance = c_fc.weight.abs().sum(dim=0)  # (intermediate_size,) -- L1 norm of each output channel's incoming weights
        keep_indices = torch.argsort(channel_importance, descending=True)[:keep_size].sort().values

        new_c_fc = Conv1D(keep_size, c_fc.weight.shape[0])
        new_c_fc.weight.data = c_fc.weight.data[:, keep_indices].clone()
        new_c_fc.bias.data = c_fc.bias.data[keep_indices].clone()

        new_c_proj = Conv1D(c_proj.nf, keep_size)
        new_c_proj.weight.data = c_proj.weight.data[keep_indices, :].clone()
        new_c_proj.bias.data = c_proj.bias.data.clone()  # c_proj's output dim (hidden_size) is untouched -- only its INPUT shrinks

        mlp.c_fc = new_c_fc.to(c_fc.weight.device, c_fc.weight.dtype)
        mlp.c_proj = new_c_proj.to(c_proj.weight.device, c_proj.weight.dtype)
        total_removed += intermediate_size - keep_size
    return total_removed


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


@torch.no_grad()
def benchmark_inference(model, batch_size: int, seq_len: int, vocab_size: int, device: str, num_repeats: int):
    model.eval()
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    for _ in range(3):
        model(input_ids=input_ids)
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(num_repeats):
        model(input_ids=input_ids)
    torch.cuda.synchronize()
    return (time.time() - start) / num_repeats


def main():
    args = build_arg_parser().parse_args()
    set_all_seeds(args.seed)
    print_config(args, "Structured pruning (MLP width) benchmark")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = load_tokenizer(TOKENIZER_NAME)
    model = load_causal_lm(args.model, quantization_config=None, torch_dtype=torch.float32, gradient_checkpointing=False).to(device)

    print_banner("LOADING DATASET")
    raw_eval = load_dataset(DATASET_NAME, split="validation")
    raw_eval = select_samples(raw_eval, args.max_eval_samples, "first", args.seed)
    eval_dataset = tokenize_and_pack(raw_eval, tokenizer, args.block_size, "eval")
    print(f"Eval blocks: {len(eval_dataset)}")
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    print_banner("BASELINE EVALUATION (dense)")
    baseline_loss, baseline_ppl = evaluate_perplexity(model, eval_dataset, data_collator, args.batch_size, device)
    baseline_params = sum(p.numel() for p in model.parameters())
    print(f"Baseline eval loss: {baseline_loss:.4f}  |  Baseline perplexity: {baseline_ppl:.2f}  |  params: {baseline_params:,}")

    print_banner(f"STRUCTURALLY PRUNING MLP WIDTH (prune_ratio={args.prune_ratio})")
    pruned_model = copy.deepcopy(model)
    channels_removed = prune_mlp_width(pruned_model, args.prune_ratio)
    pruned_params = sum(p.numel() for p in pruned_model.parameters())
    print(f"Removed {channels_removed} MLP intermediate channels total across {len(pruned_model.transformer.h)} blocks")
    print(f"Params: {baseline_params:,} -> {pruned_params:,} ({100 * (1 - pruned_params / baseline_params):.1f}% fewer)")

    if args.debug_first_batch:
        pruned_model.eval()
        with torch.no_grad():
            batch = data_collator([eval_dataset[i] for i in range(min(2, len(eval_dataset)))])
            batch = {k: v.to(device) for k, v in batch.items()}
            out = pruned_model(**batch)
        print(f"Sanity forward pass through pruned model: loss={out.loss.item():.4f}")
        print("--debug_first_batch set: exiting without the full benchmark.")
        return

    print_banner("PRUNED MODEL EVALUATION (no recovery finetuning)")
    pruned_loss, pruned_ppl = evaluate_perplexity(pruned_model, eval_dataset, data_collator, args.batch_size, device)
    print(f"Pruned eval loss: {pruned_loss:.4f}  |  Pruned perplexity: {pruned_ppl:.2f}")
    print(f"Perplexity degradation: {pruned_ppl - baseline_ppl:+.2f}")

    print_banner("INFERENCE SPEED (dense vs. pruned)")
    vocab_size = len(tokenizer)
    dense_latency = benchmark_inference(model, args.inference_batch_size, args.inference_seq_len, vocab_size, device, args.num_inference_repeats)
    pruned_latency = benchmark_inference(pruned_model, args.inference_batch_size, args.inference_seq_len, vocab_size, device, args.num_inference_repeats)
    speedup = dense_latency / pruned_latency
    print(f"Dense: {dense_latency * 1000:.2f} ms/batch  |  Pruned: {pruned_latency * 1000:.2f} ms/batch  |  {speedup:.2f}x speedup")

    write_run_result(
        output_dir=args.output_dir,
        stage="optimization",
        task="pruning_benchmark",
        modality="text",
        architecture=ARCHITECTURE,
        model_name=args.model,
        dataset_name=DATASET_NAME,
        hyperparameters=vars(args),
        metrics={
            "baseline_eval_loss": baseline_loss,
            "baseline_perplexity": baseline_ppl,
            "baseline_params": baseline_params,
            "pruned_eval_loss": pruned_loss,
            "pruned_perplexity": pruned_ppl,
            "perplexity_degradation": pruned_ppl - baseline_ppl,
            "pruned_params": pruned_params,
            "param_reduction_frac": 1 - (pruned_params / baseline_params),
            "channels_removed": channels_removed,
            "dense_latency_seconds": dense_latency,
            "pruned_latency_seconds": pruned_latency,
            "inference_speedup": speedup,
        },
        num_train_samples=0,
        num_eval_samples=len(eval_dataset),
        train_runtime_seconds=0.0,
    )
    print("Done.")


if __name__ == "__main__":
    main()
