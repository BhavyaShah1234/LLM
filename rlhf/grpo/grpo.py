"""Group Relative Policy Optimization (GRPO) -- decoder-only.

An RL method (used in DeepSeekMath / DeepSeek-R1) that estimates advantage
by sampling a *group* of completions per prompt and comparing them against
each other, rather than needing a learned value function like classic PPO --
simpler and more memory-efficient, which is presumably why `trl==1.9.2`
ships GRPOTrainer but no PPOTrainer. See root README's "RLHF algorithm
family" section for the theory.

This script uses a **verifiable, rule-based reward** (does the generated
answer letter match the ground-truth MedMCQA label?) rather than a learned
reward model -- this is RLVR (RL with Verifiable Rewards), the same
technique `ability-training/CoT/` uses to induce reasoning, and the reason
this script and that one share so much structure (see that folder's
README). No separate reward-modeling step is needed here at all.

**Prerequisite** (same reasoning as dpo.py in this stage): the policy needs
to already follow the target *format* (emit a recognizable answer letter)
for the reward function to give useful, non-zero signal -- a raw base model
tends to produce unstructured continuations the reward function can't
reliably parse. `--model` defaults to `Qwen/Qwen3-1.7B` (vendor instruct)
for the same reason dpo.py does.

Dataset: araag2/MedMCQA (config "processed") -- the same dataset
supervised-finetuning/text/mcq/decoder-only/mcq_standard.py uses, reused
here since its `Label` field (already a letter) is exactly the ground truth
a verifiable reward function needs, and it's already confirmed to load
correctly.

Usage:
    python grpo.py --debug_first_batch --max_samples 8
    python grpo.py --lora --quantization 4bit --max_samples 500

Note: this trl version's GRPOConfig (like DPOConfig, see dpo.py) has no
separate max_prompt_length field -- confirmed via introspection -- only
max_completion_length.
"""

import argparse
import re
import time

from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer

from common.data_selection import select_samples
from common.logging_utils import print_banner, print_config
from common.model_saving import save_model
from common.run_results import write_run_result
from common.seeding import set_all_seeds

DATASET_NAME = "araag2/MedMCQA"
DATASET_CONFIG = "processed"
ARCHITECTURE = "decoder-only"
INSTRUCTION = "Answer the following medical multiple choice question by selecting the correct option (A, B, C, or D)."
ANSWER_LETTER_RE = re.compile(r"\b([A-D])\b")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train a decoder-only model with GRPO against a verifiable MCQ-correctness reward.")

    p.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B", help="SFT'd/instruction-tuned checkpoint to train (local path or HF Hub id) -- NOT a raw base model, see module docstring. Default: Qwen/Qwen3-1.7B (vendor instruct).")
    p.add_argument("--quantization", type=str, default="no", choices=["no", "4bit", "8bit"], help="Quantization for loading. Default: no.")

    p.add_argument("--lora", action="store_true", default=False, help="Enable LoRA. Recommended: GRPO generates --num_generations completions per prompt during training, which is memory-hungry.")
    p.add_argument("--lora_r", type=int, default=16, help="LoRA rank. Default: 16.")
    p.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha. Default: 32.")
    p.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout. Default: 0.05.")
    p.add_argument("--lora_target_modules", type=str, default="q_proj,k_proj,v_proj,o_proj", help="Comma-separated LoRA target modules.")

    p.add_argument("--num_generations", type=int, default=4, help="Completions sampled per prompt per step (the 'group' in Group Relative Policy Optimization). Default: 4 (lower than TRL's default of 8 -- memory-bound on 8GB VRAM).")
    p.add_argument("--max_completion_length", type=int, default=16, help="Max new tokens generated per completion (a bare answer letter needs very few). Default: 16.")
    p.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature for the group of completions -- needs to be >0 for GRPO to see varied completions to compare. Default: 1.0.")
    p.add_argument("--beta", type=float, default=0.0, help="KL penalty coefficient against the reference model. Default: 0.0 (TRL's default -- no reference model needed, cheaper).")

    p.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"], help="Mixed precision. Default: bf16.")
    p.add_argument("--batch_size", type=int, default=4, help="Per-device training batch size (must be divisible by --num_generations). Default: 4.")
    p.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps. Default: 4.")
    p.add_argument("--learning_rate", type=float, default=1e-6, help="Learning rate. Default: 1e-6 (GRPO/RL is typically trained with a very low LR).")
    p.add_argument("--epochs", type=float, default=1.0, help="Training epochs. Default: 1.0.")
    p.add_argument("--gradient_checkpointing", action="store_true", default=True, help="Enable gradient checkpointing. Default: on.")

    p.add_argument("--max_samples", type=int, default=-1, help="Number of training rows to use. -1 (default) = full split.")
    p.add_argument("--sample_selection", type=str, default="random", choices=["random", "first", "last"], help="Which rows to pick when --max_samples is positive. Default: random.")
    p.add_argument("--max_eval_samples", type=int, default=100, help="Eval rows to use. Default: 100.")

    p.add_argument("--output_dir", type=str, default="./output/rlhf/grpo", help="Output directory.")
    p.add_argument("--logging_steps", type=int, default=5, help="Logging frequency. Default: 5.")
    p.add_argument("--eval_steps", type=int, default=50, help="Evaluation frequency. Default: 50.")
    p.add_argument("--save_steps", type=int, default=100, help="Checkpoint save frequency. Default: 100.")
    p.add_argument("--save_strategy", type=str, default="adapter_only", choices=["full", "adapter_only", "merged", "adapter_and_merged", "base_reference"], help="Model saving strategy. Default: adapter_only when --lora, otherwise 'full'.")

    p.add_argument("--seed", type=int, default=42, help="Random seed. Default: 42.")
    p.add_argument("--debug_first_batch", action="store_true", default=False, help="Print formatted prompts and a reward-function sanity check, then exit without training.")

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


def verify_dataset() -> None:
    print_banner("VERIFYING DATASET")
    peek = load_dataset(DATASET_NAME, DATASET_CONFIG, split="train", streaming=True)
    example = next(iter(peek))
    for field in ("Question", "Option_A", "Option_B", "Option_C", "Option_D", "Label"):
        assert field in example, f"Expected a {field!r} field in {DATASET_NAME}/{DATASET_CONFIG}, got fields: {list(example.keys())}"
    print(f"Dataset: {DATASET_NAME} (config={DATASET_CONFIG})")
    print(f"Fields: {list(example.keys())}")
    print(f"Sample prompt:\n{build_prompt(example)}")
    print()


def load_and_prepare_data(args):
    print_banner("LOADING DATASET")
    train_raw = load_dataset(DATASET_NAME, DATASET_CONFIG, split="train")
    # NOT split="test": confirmed empirically that araag2/MedMCQA's test split has Label=None
    # for every row (labels withheld, standard practice to prevent leaderboard cheating) --
    # every reward comparison against it would silently be 0. "dev" has real labels.
    eval_raw = load_dataset(DATASET_NAME, DATASET_CONFIG, split="dev")

    train_raw = select_samples(train_raw, args.max_samples, args.sample_selection, args.seed)
    eval_raw = select_samples(eval_raw, args.max_eval_samples, "first", args.seed)

    def to_prompt_dataset(examples):
        return {
            "prompt": [build_prompt({"Question": q, "Option_A": a, "Option_B": b, "Option_C": c, "Option_D": d}) for q, a, b, c, d in zip(examples["Question"], examples["Option_A"], examples["Option_B"], examples["Option_C"], examples["Option_D"])],
            "answer": examples["Label"],
        }

    train_dataset = train_raw.map(to_prompt_dataset, batched=True, remove_columns=train_raw.column_names, desc="Building prompts (train)")
    eval_dataset = eval_raw.map(to_prompt_dataset, batched=True, remove_columns=eval_raw.column_names, desc="Building prompts (eval)")

    print(f"Train samples: {len(train_dataset)}")
    print(f"Eval samples: {len(eval_dataset)}")
    print()
    return train_dataset, eval_dataset


def mcq_correctness_reward(prompts, completions, answer, **kwargs) -> list:
    """Verifiable, rule-based reward: 1.0 if the completion's first standalone
    A/B/C/D letter matches the ground-truth answer, else 0.0. No learned
    reward model, no human labels needed at training time -- this is RLVR."""
    rewards = []
    for completion, correct_letter in zip(completions, answer):
        match = ANSWER_LETTER_RE.search(completion.strip().upper())
        predicted = match.group(1) if match else None
        rewards.append(1.0 if predicted == correct_letter else 0.0)
    return rewards


def main():
    args = build_arg_parser().parse_args()
    set_all_seeds(args.seed)
    print_config(args, "GRPO -- decoder-only, verifiable MCQ-correctness reward (RLVR)")

    # verify_dataset()  # commented out: avoids loading the dataset twice (a streaming peek here, then a full load in load_and_prepare_data()) to cut memory overhead
    train_dataset, eval_dataset = load_and_prepare_data(args)

    if args.debug_first_batch:
        print_banner("REWARD FUNCTION SANITY CHECK")
        sample_answers = train_dataset["answer"][:4]
        fake_completions = [f"The answer is {a}." for a in sample_answers]
        fake_completions[0] = "I'm not sure, maybe B or C."
        rewards = mcq_correctness_reward(None, fake_completions, sample_answers)
        for ans, comp, r in zip(sample_answers, fake_completions, rewards):
            print(f"  ground truth={ans!r}  completion={comp!r}  reward={r}")
        print("\n--- Example prompt ---")
        print(train_dataset[0]["prompt"])
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

    training_args = GRPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        temperature=args.temperature,
        beta=args.beta,
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

    trainer = GRPOTrainer(
        model=args.model,
        reward_funcs=mcq_correctness_reward,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        quantization_config=quantization_config,
        peft_config=peft_config,
    )

    print_banner("TRAINING")
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
        task="grpo",
        modality="text",
        architecture=ARCHITECTURE,
        model_name=args.model,
        dataset_name=f"{DATASET_NAME} ({DATASET_CONFIG})",
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
