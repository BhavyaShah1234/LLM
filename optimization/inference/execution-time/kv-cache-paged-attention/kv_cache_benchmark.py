"""KV-Cache -- isolated benchmark script (inference, execution-time).

Compares autoregressive generation with `use_cache=True` (the default --
each step reuses cached key/value projections from all previous tokens)
against `use_cache=False` (every step recomputes attention over the ENTIRE
sequence so far from scratch), via real generation and wall-clock timing.
This isolates the fundamental algorithmic benefit KV-caching provides --
turning each generation step from O(sequence_length) attention compute
into O(1) (amortized) -- from the more sophisticated allocator-level
technique this folder's name also references.

**Paged attention specifically** (solving KV-cache memory *fragmentation*
across many concurrent requests, vLLM's core contribution) is a
serving-infrastructure technique, not something a single-request HF
`generate()` call can demonstrate in isolation -- it only matters once
you're managing a KV-cache pool across many simultaneous requests with
different lengths. That's exactly what `production/serve_and_benchmark.py`
already exercises via vLLM's engine (which uses paged attention internally
for every request it serves) -- see that script's README for the real
throughput numbers. This script's job is narrower and complementary: prove
the basic caching mechanism itself matters, with a single-request,
framework-agnostic (`transformers`, not vLLM) comparison.

Usage:
    python kv_cache_benchmark.py --debug_first_batch
    python kv_cache_benchmark.py --num_prompts 5 --max_new_tokens 64
"""

import argparse
import time

import torch

from common.data_selection import select_samples
from common.logging_utils import print_banner, print_config
from common.model_loading import load_causal_lm, load_tokenizer
from common.run_results import write_run_result
from common.seeding import set_all_seeds

ARCHITECTURE = "decoder-only"
DATASET_NAME = "araag2/MedMCQA"
DATASET_CONFIG = "processed"
INSTRUCTION = "Answer the following medical multiple choice question by selecting the correct option (A, B, C, or D)."


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for the KV-cache benchmark.

    Returns:
        argparse.ArgumentParser: Parser covering model choice, prompt count,
        generation length, seed, and output path.
    """
    p = argparse.ArgumentParser(description="Benchmark generation with vs. without KV-caching.")
    p.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B-Base", help="Model to benchmark. Default: this project's usual base-model default.")
    p.add_argument("--num_prompts", type=int, default=5, help="Number of MedMCQA dev prompts to generate on. Default: 5 (use_cache=False is slow -- O(seq_len) recompute every step -- so this stays small).")
    p.add_argument("--max_new_tokens", type=int, default=64, help="Max generated tokens per prompt. Default: 64.")
    p.add_argument("--seed", type=int, default=42, help="Random seed. Default: 42.")
    p.add_argument("--output_dir", type=str, default="./output/optimization/kv_cache_benchmark", help="Where to write run_result.json.")
    p.add_argument("--debug_first_batch", action="store_true", default=False, help="Generate for 1 prompt with each mode, print outputs, then exit without the full benchmark.")
    return p


def format_question(row) -> str:
    """Render a MedMCQA row's question and four options as plain text.

    Args:
        row: Dataset row with `Question`, `Option_A`-`Option_D`.

    Returns:
        str: Question followed by lettered options.
    """
    return f"{row['Question']}\n\nA. {row['Option_A']}\nB. {row['Option_B']}\nC. {row['Option_C']}\nD. {row['Option_D']}"


def build_prompt(row) -> str:
    """Wrap a formatted MedMCQA question in the instruction/input/response template.

    Args:
        row: Dataset row with `Question`, `Option_A`-`Option_D`.

    Returns:
        str: Full prompt text ending at `### Response:\n`, ready for generation.
    """
    return f"### Instruction:\n{INSTRUCTION}\n\n### Input:\n{format_question(row)}\n\n### Response:\n"


def load_prompts(num_prompts: int, seed: int):
    """Load a fixed slice of the MedMCQA dev split and render it into prompts.

    Args:
        num_prompts (int): Number of prompts to load.
        seed (int): Seed passed through to `select_samples`.

    Returns:
        list[str]: Rendered prompts, one per selected row.
    """
    from datasets import load_dataset

    eval_raw = load_dataset(DATASET_NAME, DATASET_CONFIG, split="dev")
    eval_raw = select_samples(eval_raw, num_prompts, "first", seed)
    return [build_prompt(row) for row in eval_raw]


def generate_and_time(model, tokenizer, prompts, max_new_tokens: int, use_cache: bool):
    """Greedy-generate a completion for each prompt and time the total wall-clock cost.

    Args:
        model: Causal LM to generate from.
        tokenizer: Tokenizer paired with `model`.
        prompts (list[str]): Prompts to generate on, one request at a time.
        max_new_tokens (int): Max tokens to generate per prompt.
        use_cache (bool): Whether to enable KV-caching during generation.

    Returns:
        tuple[float, int]: `(elapsed_seconds, total_generated_tokens)`.
    """
    device = model.device
    total_tokens = 0
    start = time.time()
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, use_cache=use_cache, pad_token_id=tokenizer.pad_token_id)
        total_tokens += output.shape[1] - inputs["input_ids"].shape[1]
    elapsed = time.time() - start
    return elapsed, total_tokens


def main():
    """Run the end-to-end KV-cache benchmark: generate the same prompts with and
    without KV-caching, compare throughput, and record results via
    `write_run_result` (or exit early with `--debug_first_batch`).
    """
    args = build_arg_parser().parse_args()
    set_all_seeds(args.seed)
    print_config(args, "KV-cache benchmark")

    print_banner("LOADING DATASET (MedMCQA dev split -- same as rlhf/grpo/grpo.py)")
    prompts = load_prompts(args.num_prompts, args.seed)
    print(f"Prompts: {len(prompts)}")

    print_banner("LOADING MODEL")
    tokenizer = load_tokenizer(args.model)
    model = load_causal_lm(args.model, quantization_config=None, torch_dtype=torch.bfloat16, gradient_checkpointing=False)
    model.eval()

    if args.debug_first_batch:
        debug_prompts = prompts[:1]
        print_banner("DEBUG: use_cache=True")
        elapsed, tokens = generate_and_time(model, tokenizer, debug_prompts, args.max_new_tokens, use_cache=True)
        print(f"{tokens} tokens in {elapsed:.2f}s")
        print_banner("DEBUG: use_cache=False")
        elapsed, tokens = generate_and_time(model, tokenizer, debug_prompts, args.max_new_tokens, use_cache=False)
        print(f"{tokens} tokens in {elapsed:.2f}s")
        print("--debug_first_batch set: exiting without the full benchmark.")
        return

    print_banner("GENERATION: use_cache=True")
    cached_elapsed, cached_tokens = generate_and_time(model, tokenizer, prompts, args.max_new_tokens, use_cache=True)
    cached_throughput = cached_tokens / cached_elapsed
    print(f"{cached_tokens} tokens in {cached_elapsed:.2f}s -> {cached_throughput:.1f} tok/s")

    print_banner("GENERATION: use_cache=False (recompute full attention every step)")
    uncached_elapsed, uncached_tokens = generate_and_time(model, tokenizer, prompts, args.max_new_tokens, use_cache=False)
    uncached_throughput = uncached_tokens / uncached_elapsed
    print(f"{uncached_tokens} tokens in {uncached_elapsed:.2f}s -> {uncached_throughput:.1f} tok/s")

    print_banner("SUMMARY")
    speedup = uncached_elapsed / cached_elapsed
    print(f"KV-cache speedup: {speedup:.2f}x")

    write_run_result(
        output_dir=args.output_dir,
        stage="optimization",
        task="kv_cache_benchmark",
        modality="text",
        architecture=ARCHITECTURE,
        model_name=args.model,
        dataset_name=f"{DATASET_NAME} ({DATASET_CONFIG})",
        hyperparameters=vars(args),
        metrics={
            "cached_elapsed_seconds": cached_elapsed,
            "cached_tokens": cached_tokens,
            "cached_throughput_tokens_per_sec": cached_throughput,
            "uncached_elapsed_seconds": uncached_elapsed,
            "uncached_tokens": uncached_tokens,
            "uncached_throughput_tokens_per_sec": uncached_throughput,
            "speedup": speedup,
        },
        num_train_samples=0,
        num_eval_samples=len(prompts),
        train_runtime_seconds=cached_elapsed + uncached_elapsed,
    )
    print("Done.")


if __name__ == "__main__":
    main()
