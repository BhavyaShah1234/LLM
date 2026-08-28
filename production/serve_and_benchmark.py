"""vLLM serving benchmark -- production/ stage.

Loads a model into vLLM's offline `LLM` engine (the same engine the
OpenAI-compatible HTTP server wraps -- using it directly here skips
subprocess/port-management complexity while exercising the identical
continuous-batching inference path) and, optionally, attaches a LoRA
adapter natively via vLLM's `LoRARequest` -- NO merge step needed. That
matters for this project specifically: `common/model_saving.py`'s
`adapter_only` strategy is what most `--lora` runs here actually use (it's
the cheapest way to keep every experiment's result around, see
model_saving.py's docstring), so a serving script that only accepted merged
checkpoints would leave most of this project's own trained artifacts
unservable. vLLM's native LoRA support serves the adapter directly against
the same base model instead.

Uses the exact same prompt format, dataset, and answer-extraction regex as
rlhf/grpo/grpo.py so this script can
do more than measure throughput: with --lora_path pointing at that script's
output, it also reports base-model vs LoRA-adapted accuracy on the same
prompts under vLLM -- a real check that the trained adapter's effect
survives being served through a different inference engine than it was
trained/evaluated with (trl's HF-based generation).

Usage:
    python serve_and_benchmark.py --debug_first_batch
    python serve_and_benchmark.py --model Qwen/Qwen3-1.7B --num_prompts 20
    python serve_and_benchmark.py --model Qwen/Qwen3-1.7B --lora_path ./output/rlhf/grpo --num_prompts 20
"""

import argparse
import re
import time

from datasets import load_dataset

from common.data_selection import select_samples
from common.logging_utils import print_banner, print_config
from common.run_results import write_run_result
from common.seeding import set_all_seeds

DATASET_NAME = "araag2/MedMCQA"
DATASET_CONFIG = "processed"
INSTRUCTION = "Answer the following medical multiple choice question by selecting the correct option (A, B, C, or D)."
ANSWER_LETTER_RE = re.compile(r"\b([A-D])\b")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Serve a model with vLLM and benchmark throughput/latency, optionally with a LoRA adapter attached natively.")
    p.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B", help="Base model to serve (HF Hub id or local path). Default matches rlhf/'s default.")
    p.add_argument("--lora_path", type=str, default=None, help="Path to a LoRA adapter directory (adapter_config.json + adapter_model.safetensors, e.g. this project's own adapter_only saves) to attach natively via vLLM's LoRARequest. Default: none (serve the base model only).")
    p.add_argument("--lora_rank", type=int, default=16, help="Rank of the LoRA adapter at --lora_path -- must be >= the adapter's actual r (this project's common/peft_setup.py default is 16). Only used if --lora_path is set.")
    p.add_argument("--quantization", type=str, default=None, choices=[None, "bitsandbytes"], help="vLLM inference-time quantization. Default: none (bf16 serving).")
    p.add_argument("--gpu_memory_utilization", type=float, default=0.7, help="Fraction of GPU memory vLLM is allowed to reserve (weights + KV cache). Default: 0.7 -- 0.5 works for base-model-only serving but underflows (negative KV cache budget) once --lora_path's Punica LoRA kernels and extra torch.compile artifacts are loaded; confirmed empirically on this 8GB card.")
    p.add_argument("--max_model_len", type=int, default=512, help="Max sequence length (prompt + generation). Default: 512.")
    p.add_argument("--num_prompts", type=int, default=20, help="Number of MedMCQA dev prompts to benchmark on. Default: 20.")
    p.add_argument("--max_tokens", type=int, default=64, help="Max generated tokens per prompt. Default: 64.")
    p.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature. Default: 0.0 (greedy, deterministic -- appropriate for a reproducible benchmark).")
    p.add_argument("--seed", type=int, default=42, help="Random seed for prompt selection. Default: 42.")
    p.add_argument("--output_dir", type=str, default="./output/production/vllm_serving", help="Where to write run_result.json.")
    p.add_argument("--debug_first_batch", action="store_true", default=False, help="Load the model, generate for 2 prompts, print outputs, then exit without the full benchmark.")
    return p


def format_question(row) -> str:
    return (
        f"{row['Question']}\n\n"
        f"A. {row['Option_A']}\n"
        f"B. {row['Option_B']}\n"
        f"C. {row['Option_C']}\n"
        f"D. {row['Option_D']}"
    )


def build_prompt(row) -> str:
    return f"### Instruction:\n{INSTRUCTION}\n\n### Input:\n{format_question(row)}\n\n### Response:\n"


def extract_answer_letter(completion: str):
    match = ANSWER_LETTER_RE.search(completion.strip().upper())
    return match.group(1) if match else None


def accuracy(completions, labels) -> float:
    correct = sum(1 for c, l in zip(completions, labels) if extract_answer_letter(c) == l)
    return correct / len(labels)


def main():
    args = build_arg_parser().parse_args()
    set_all_seeds(args.seed)
    print_config(args, "vLLM serving benchmark")

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    print_banner("LOADING DATASET (MedMCQA dev split -- same as rlhf/grpo/grpo.py)")
    eval_raw = load_dataset(DATASET_NAME, DATASET_CONFIG, split="dev")
    eval_raw = select_samples(eval_raw, args.num_prompts, "first", args.seed)
    prompts = [build_prompt(row) for row in eval_raw]
    labels = [row["Label"] for row in eval_raw]
    print(f"Prompts: {len(prompts)}")
    print(f"Sample prompt:\n{prompts[0]}")

    print_banner("STARTING vLLM ENGINE")
    llm_kwargs = dict(
        model=args.model,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        dtype="bfloat16",
        seed=args.seed,
    )
    if args.quantization:
        llm_kwargs["quantization"] = args.quantization
    if args.lora_path:
        llm_kwargs["enable_lora"] = True
        llm_kwargs["max_lora_rank"] = args.lora_rank
    llm = LLM(**llm_kwargs)

    sampling_params = SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens)
    lora_request = LoRARequest("adapter", 1, args.lora_path) if args.lora_path else None

    if args.debug_first_batch:
        debug_prompts = prompts[:2]
        outputs = llm.generate(debug_prompts, sampling_params, lora_request=lora_request)
        for prompt, output in zip(debug_prompts, outputs):
            print(f"--- Prompt ---\n{prompt}")
            print(f"--- Completion ---\n{output.outputs[0].text}\n")
        print("--debug_first_batch set: exiting without the full benchmark.")
        return

    print_banner("BASE MODEL: GENERATING + BENCHMARKING")
    start = time.time()
    base_outputs = llm.generate(prompts, sampling_params)
    base_runtime = time.time() - start
    base_completions = [o.outputs[0].text for o in base_outputs]
    base_num_tokens = sum(len(o.outputs[0].token_ids) for o in base_outputs)
    base_accuracy = accuracy(base_completions, labels)
    base_throughput = base_num_tokens / base_runtime
    print(f"Base model: {base_runtime:.2f}s for {len(prompts)} prompts, "
          f"{base_num_tokens} tokens generated, {base_throughput:.1f} tok/s, accuracy={base_accuracy:.3f}")

    lora_accuracy = None
    lora_throughput = None
    lora_runtime = None
    if args.lora_path:
        print_banner("LORA-ADAPTED MODEL: GENERATING + BENCHMARKING (same base engine, native vLLM LoRA)")
        start = time.time()
        lora_outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)
        lora_runtime = time.time() - start
        lora_completions = [o.outputs[0].text for o in lora_outputs]
        lora_num_tokens = sum(len(o.outputs[0].token_ids) for o in lora_outputs)
        lora_accuracy = accuracy(lora_completions, labels)
        lora_throughput = lora_num_tokens / lora_runtime
        print(f"LoRA-adapted: {lora_runtime:.2f}s for {len(prompts)} prompts, "
              f"{lora_num_tokens} tokens generated, {lora_throughput:.1f} tok/s, accuracy={lora_accuracy:.3f}")
        print(f"Accuracy delta (lora - base): {lora_accuracy - base_accuracy:+.3f}")

    metrics = {
        "base_throughput_tokens_per_sec": base_throughput,
        "base_runtime_seconds": base_runtime,
        "base_accuracy": base_accuracy,
        "lora_throughput_tokens_per_sec": lora_throughput,
        "lora_runtime_seconds": lora_runtime,
        "lora_accuracy": lora_accuracy,
        "accuracy_delta": (lora_accuracy - base_accuracy) if lora_accuracy is not None else None,
        "num_prompts": len(prompts),
        "max_tokens": args.max_tokens,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "quantization": args.quantization,
    }

    write_run_result(
        output_dir=args.output_dir,
        stage="production",
        task="vllm_serving",
        modality="text",
        architecture="decoder-only",
        model_name=args.model,
        dataset_name=f"{DATASET_NAME} ({DATASET_CONFIG})",
        hyperparameters=vars(args),
        metrics=metrics,
        num_train_samples=0,
        num_eval_samples=len(prompts),
        train_runtime_seconds=base_runtime + (lora_runtime or 0.0),
    )
    print("Done.")


if __name__ == "__main__":
    main()
