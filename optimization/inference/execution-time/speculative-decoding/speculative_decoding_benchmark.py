"""Speculative Decoding -- isolated benchmark script (inference, execution-time).

Compares plain autoregressive generation against speculative decoding
(a smaller "draft" model proposes several tokens per step, the target
model verifies them in a single forward pass, accepting the prefix that
matches what it would have generated itself) via HF `generate()`'s built-in
`assistant_model` argument. Real generation, real wall-clock, same prompts,
same target model, same decoding settings -- only the presence of the
draft model differs.

Target: Qwen/Qwen3-1.7B-Base (this project's usual base-model default).
Draft: Qwen/Qwen3-0.6B-Base -- same tokenizer/vocab_size (151936,
confirmed via AutoConfig/AutoTokenizer before implementing; speculative
decoding requires the draft and target to share a vocabulary so draft
tokens can be verified directly against the target's own logits), same
model family, ~3x fewer parameters.

Usage:
    python speculative_decoding_benchmark.py --debug_first_batch
    python speculative_decoding_benchmark.py --num_prompts 10 --max_new_tokens 64
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
    p = argparse.ArgumentParser(description="Benchmark plain vs. speculative-decoding generation throughput.")
    p.add_argument("--target_model", type=str, default="Qwen/Qwen3-1.7B-Base", help="Target model whose output distribution is preserved exactly. Default: this project's usual base-model default.")
    p.add_argument("--draft_model", type=str, default="Qwen/Qwen3-0.6B-Base", help="Smaller draft/assistant model proposing tokens for the target to verify. Default: same family/tokenizer as --target_model, ~3x fewer params.")
    p.add_argument("--num_prompts", type=int, default=10, help="Number of MedMCQA dev prompts to generate on. Default: 10.")
    p.add_argument("--max_new_tokens", type=int, default=64, help="Max generated tokens per prompt. Default: 64.")
    p.add_argument("--seed", type=int, default=42, help="Random seed. Default: 42.")
    p.add_argument("--output_dir", type=str, default="./output/optimization/speculative_decoding_benchmark", help="Where to write run_result.json.")
    p.add_argument("--debug_first_batch", action="store_true", default=False, help="Generate for 1 prompt with each mode, print outputs, then exit without the full benchmark.")
    return p


def format_question(row) -> str:
    return f"{row['Question']}\n\nA. {row['Option_A']}\nB. {row['Option_B']}\nC. {row['Option_C']}\nD. {row['Option_D']}"


def build_prompt(row) -> str:
    return f"### Instruction:\n{INSTRUCTION}\n\n### Input:\n{format_question(row)}\n\n### Response:\n"


def load_prompts(num_prompts: int, seed: int):
    from datasets import load_dataset

    eval_raw = load_dataset(DATASET_NAME, DATASET_CONFIG, split="dev")
    eval_raw = select_samples(eval_raw, num_prompts, "first", seed)
    return [build_prompt(row) for row in eval_raw]


def generate_and_time(model, tokenizer, prompts, max_new_tokens: int, assistant_model=None):
    device = model.device
    total_tokens = 0
    start = time.time()
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                assistant_model=assistant_model, pad_token_id=tokenizer.pad_token_id,
            )
        total_tokens += output.shape[1] - inputs["input_ids"].shape[1]
    elapsed = time.time() - start
    return elapsed, total_tokens


def main():
    args = build_arg_parser().parse_args()
    set_all_seeds(args.seed)
    print_config(args, "Speculative decoding benchmark")

    print_banner("LOADING DATASET (MedMCQA dev split -- same as rlhf/grpo/grpo.py)")
    prompts = load_prompts(args.num_prompts, args.seed)
    print(f"Prompts: {len(prompts)}")

    print_banner("LOADING MODELS")
    tokenizer = load_tokenizer(args.target_model)
    target_model = load_causal_lm(args.target_model, quantization_config=None, torch_dtype=torch.bfloat16, gradient_checkpointing=False)
    draft_model = load_causal_lm(args.draft_model, quantization_config=None, torch_dtype=torch.bfloat16, gradient_checkpointing=False)
    target_model.eval()
    draft_model.eval()

    if args.debug_first_batch:
        debug_prompts = prompts[:1]
        print_banner("DEBUG: plain generation")
        elapsed, tokens = generate_and_time(target_model, tokenizer, debug_prompts, args.max_new_tokens)
        print(f"{tokens} tokens in {elapsed:.2f}s")
        print_banner("DEBUG: speculative decoding")
        elapsed, tokens = generate_and_time(target_model, tokenizer, debug_prompts, args.max_new_tokens, assistant_model=draft_model)
        print(f"{tokens} tokens in {elapsed:.2f}s")
        print("--debug_first_batch set: exiting without the full benchmark.")
        return

    print_banner("PLAIN AUTOREGRESSIVE GENERATION")
    plain_elapsed, plain_tokens = generate_and_time(target_model, tokenizer, prompts, args.max_new_tokens)
    plain_throughput = plain_tokens / plain_elapsed
    print(f"{plain_tokens} tokens in {plain_elapsed:.2f}s -> {plain_throughput:.1f} tok/s")

    print_banner("SPECULATIVE DECODING (draft proposes, target verifies)")
    spec_elapsed, spec_tokens = generate_and_time(target_model, tokenizer, prompts, args.max_new_tokens, assistant_model=draft_model)
    spec_throughput = spec_tokens / spec_elapsed
    print(f"{spec_tokens} tokens in {spec_elapsed:.2f}s -> {spec_throughput:.1f} tok/s")

    print_banner("SUMMARY")
    speedup = plain_elapsed / spec_elapsed
    print(f"Speculative decoding speedup: {speedup:.2f}x")
    print(f"(Token counts may differ slightly between modes -- speculative decoding still samples from the TARGET's exact distribution, "
          f"but greedy argmax ties can resolve differently when verification batches multiple draft tokens at once.)")

    write_run_result(
        output_dir=args.output_dir,
        stage="optimization",
        task="speculative_decoding_benchmark",
        modality="text",
        architecture=ARCHITECTURE,
        model_name=args.target_model,
        dataset_name=f"{DATASET_NAME} ({DATASET_CONFIG})",
        hyperparameters=vars(args),
        metrics={
            "plain_elapsed_seconds": plain_elapsed,
            "plain_tokens": plain_tokens,
            "plain_throughput_tokens_per_sec": plain_throughput,
            "speculative_elapsed_seconds": spec_elapsed,
            "speculative_tokens": spec_tokens,
            "speculative_throughput_tokens_per_sec": spec_throughput,
            "speedup": speedup,
        },
        num_train_samples=0,
        num_eval_samples=len(prompts),
        train_runtime_seconds=plain_elapsed + spec_elapsed,
    )
    print("Done.")


if __name__ == "__main__":
    main()
