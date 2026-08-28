"""Direct Preference Optimization (DPO) -- decoder-only.

Aligns a model to human preferences using pairs of (chosen, rejected)
responses to the same prompt, without a separate reward model or online RL
rollout -- see root README's "RLHF algorithm family" section for the theory.

**Prerequisite (see root README's Model Selection Philosophy and this
project's rlhf/README.md)**: DPO
assumes the policy model already follows the target response format/
behavior -- running it on a raw pretrained base model doesn't work in
practice (there's nothing coherent yet to align). `--model` therefore
defaults to `Qwen/Qwen3-1.7B` (the vendor **instruction-tuned** sibling of
this project's usual `Qwen/Qwen3-1.7B-Base` default), NOT the base model.
This project's own supervised-finetuning/text/instruction-tuning/ or
chat-tuning/ scripts had only been smoke-tested (--debug_first_batch), not
actually trained to completion, at the time this script was written -- once
one of them has been trained for real, point --model at its output
directory instead (e.g. `--model ./output/supervised-finetuning/text/chat-tuning/decoder-only/standard`)
to see the "our own instruction-tuning" starting point instead of the vendor
one. That before/after comparison is exactly what
experiments/rlhf-pretrained-vs-sft-init/ is for.

Dataset: trl-lib/ultrafeedback_binarized (chosen/rejected conversational
pairs, the standard TRL example dataset for DPO -- 62k train / 1k test
rows, already in the exact {"role","content"} format DPOTrainer expects
natively, no reformatting needed).

Usage:
    python dpo.py --debug_first_batch --max_samples 20
    python dpo.py --lora --quantization 4bit --max_samples 2000

Memory note, confirmed empirically: DPOTrainer loads a policy model AND a
reference model, so even with --lora, a real training run of a ~4GB model
(Qwen/Qwen3-1.7B's fp16 weights) OOM'd on this project's 8GB GPU without
--quantization 4bit (two fp16 copies alone exceed 8GB before any
activations/optimizer state). Unlike the supervised-finetuning/ scripts,
where --quantization is optional headroom, it's effectively required here
for models at this scale on this hardware.
"""

import argparse

import torch
from datasets import load_dataset
from trl import DPOConfig, DPOTrainer

from common.data_selection import select_samples
from common.logging_utils import print_banner, print_config
from common.model_saving import save_model
from common.run_results import write_run_result
from common.seeding import set_all_seeds

DATASET_NAME = "trl-lib/ultrafeedback_binarized"
ARCHITECTURE = "decoder-only"


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Align a decoder-only model with DPO.")

    p.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B", help="SFT'd/instruction-tuned checkpoint to align (local path or HF Hub id) -- NOT a raw base model, see module docstring. Default: Qwen/Qwen3-1.7B (vendor instruct, fp16 ~4.1GB).")
    p.add_argument("--quantization", type=str, default="no", choices=["no", "4bit", "8bit"], help="Quantization for loading. Default: no.")

    p.add_argument("--lora", action="store_true", default=False, help="Enable LoRA. Recommended: full-parameter DPO of a 1.7-4B model needs a reference-model copy in memory too, doubling the footprint vs. SFT.")
    p.add_argument("--lora_r", type=int, default=16, help="LoRA rank. Default: 16.")
    p.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha. Default: 32.")
    p.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout. Default: 0.05.")
    p.add_argument("--lora_target_modules", type=str, default="q_proj,k_proj,v_proj,o_proj", help="Comma-separated LoRA target modules.")

    p.add_argument("--beta", type=float, default=0.1, help="DPO temperature (controls how strongly the policy is pulled from the reference model). Default: 0.1.")
    p.add_argument("--max_length", type=int, default=512, help="Max total sequence length (prompt + response combined -- this trl version's DPOConfig has no separate max_prompt_length field, confirmed via introspection). Default: 512.")

    p.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"], help="Mixed precision. Default: bf16.")
    p.add_argument("--batch_size", type=int, default=1, help="Per-device training batch size. Default: 1 (DPO keeps a reference-model copy resident -- see --lora note above).")
    p.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Gradient accumulation steps. Default: 8.")
    p.add_argument("--learning_rate", type=float, default=5e-6, help="Learning rate. Default: 5e-6 (DPO is typically trained with a much lower LR than SFT).")
    p.add_argument("--epochs", type=float, default=1.0, help="Training epochs. Default: 1.0.")
    p.add_argument("--gradient_checkpointing", action="store_true", default=True, help="Enable gradient checkpointing. Default: on.")

    p.add_argument("--max_samples", type=int, default=-1, help="Number of training rows to use. -1 (default) = full split.")
    p.add_argument("--sample_selection", type=str, default="random", choices=["random", "first", "last"], help="Which rows to pick when --max_samples is positive. Default: random.")
    p.add_argument("--max_eval_samples", type=int, default=200, help="Eval rows to use. Default: 200.")

    p.add_argument("--output_dir", type=str, default="./output/rlhf/dpo", help="Output directory.")
    p.add_argument("--logging_steps", type=int, default=10, help="Logging frequency. Default: 10.")
    p.add_argument("--eval_steps", type=int, default=100, help="Evaluation frequency. Default: 100.")
    p.add_argument("--save_steps", type=int, default=200, help="Checkpoint save frequency. Default: 200.")
    p.add_argument("--save_strategy", type=str, default="adapter_only", choices=["full", "adapter_only", "merged", "adapter_and_merged", "base_reference"], help="Model saving strategy. Default: adapter_only when --lora, otherwise 'full'.")

    p.add_argument("--seed", type=int, default=42, help="Random seed. Default: 42.")
    p.add_argument("--debug_first_batch", action="store_true", default=False, help="Print formatted preference pairs and exit without training.")

    return p


def verify_dataset() -> None:
    print_banner("VERIFYING DATASET")
    peek = load_dataset(DATASET_NAME, split="train", streaming=True)
    example = next(iter(peek))
    for field in ("chosen", "rejected"):
        assert field in example, f"Expected a {field!r} field in {DATASET_NAME}, got fields: {list(example.keys())}"
    print(f"Dataset: {DATASET_NAME}")
    print(f"Fields: {list(example.keys())}")
    print(f"Sample chosen[-1]: {example['chosen'][-1]['content'][:150]!r}")
    print(f"Sample rejected[-1]: {example['rejected'][-1]['content'][:150]!r}")
    print()


def load_and_prepare_data(args):
    print_banner("LOADING DATASET")
    train_dataset = load_dataset(DATASET_NAME, split="train")
    eval_dataset = load_dataset(DATASET_NAME, split="test")

    train_dataset = select_samples(train_dataset, args.max_samples, args.sample_selection, args.seed)
    eval_dataset = select_samples(eval_dataset, args.max_eval_samples, "first", args.seed)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Eval samples: {len(eval_dataset)}")
    print()
    return train_dataset, eval_dataset


def print_debug_examples(dataset, num_examples: int = 2) -> None:
    print_banner("FORMATTED PREFERENCE PAIRS")
    for i in range(min(num_examples, len(dataset))):
        example = dataset[i]
        prompt_text = example["chosen"][0]["content"] if isinstance(example["chosen"], list) else example.get("prompt", "")
        chosen_text = example["chosen"][-1]["content"]
        rejected_text = example["rejected"][-1]["content"]
        print(f"\n--- Example {i + 1} ---")
        print(f"Prompt: {prompt_text[:200]!r}")
        print(f"Chosen:   {chosen_text[:200]!r}")
        print(f"Rejected: {rejected_text[:200]!r}")
    print()


def main():
    args = build_arg_parser().parse_args()
    set_all_seeds(args.seed)
    print_config(args, "DPO alignment -- decoder-only")

    # verify_dataset()  # commented out: avoids loading the dataset twice (a streaming peek here, then a full load in load_and_prepare_data()) to cut memory overhead
    train_dataset, eval_dataset = load_and_prepare_data(args)

    if args.debug_first_batch:
        print_debug_examples(train_dataset, num_examples=2)
        print("--debug_first_batch set: exiting without training.")
        return

    quantization_config = None
    if args.quantization != "no":
        from common.quantization import build_quantization_config

        quantization_config = build_quantization_config(args.quantization, args.mixed_precision)

    peft_config = None
    if args.lora:
        from common.peft_setup import build_lora_config

        peft_config = build_lora_config(args.lora_r, args.lora_alpha, args.lora_dropout, args.lora_target_modules)

    training_args = DPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        beta=args.beta,
        max_length=args.max_length,
        bf16=(args.mixed_precision == "bf16"),
        fp16=(args.mixed_precision == "fp16"),
        gradient_checkpointing=args.gradient_checkpointing,
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        seed=args.seed,
        report_to=[],
    )

    trainer = DPOTrainer(
        model=args.model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        quantization_config=quantization_config,
        peft_config=peft_config,
    )

    print_banner("TRAINING")
    import time

    start = time.time()
    trainer.train()
    train_runtime_seconds = time.time() - start

    print_banner("FINAL EVALUATION")
    eval_metrics = trainer.evaluate()
    print(eval_metrics)

    save_strategy = args.save_strategy if args.lora else "full"
    save_model(trainer.model, trainer.processing_class, args.output_dir, strategy=save_strategy, base_model_name=args.model, is_lora=args.lora)

    write_run_result(
        output_dir=args.output_dir,
        stage="rlhf",
        task="dpo",
        modality="text",
        architecture=ARCHITECTURE,
        model_name=args.model,
        dataset_name=DATASET_NAME,
        save_strategy=save_strategy,
        hyperparameters=vars(args),
        metrics={k: v for k, v in eval_metrics.items() if isinstance(v, (int, float))},
        num_train_samples=len(train_dataset),
        num_eval_samples=len(eval_dataset),
        train_runtime_seconds=train_runtime_seconds,
    )
    print("Done.")


if __name__ == "__main__":
    main()
