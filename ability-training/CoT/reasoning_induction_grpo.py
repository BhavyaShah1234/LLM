"""Inducing Chain-of-Thought reasoning via RL with Verifiable Rewards (RLVR).

This is this project's answer to "can we give a model CoT/reasoning ability
it doesn't have off-the-shelf?" -- via reinforcement learning, not
supervision. It reuses the exact GRPO mechanism from
rlhf/grpo/grpo.py (same trainer, same
dataset, same underlying correctness reward), but adds a **second reward
component** -- a format reward for producing well-formed `<think>...</think>`
structure -- and explicitly tracks whether `<think>` usage *emerges* over
training, starting from a model that was never shown a single example of
`<think>`-formatted reasoning. This is the DeepSeek-R1-Zero-style
phenomenon: structured reasoning emerging purely from an outcome-based
reward, not from imitating reasoning traces (contrast with this project's
CoT-variant supervised-finetuning/ scripts, which take the *other*
approach -- distillation from teacher-provided reasoning traces).

`Qwen/Qwen3-1.7B` (this script's default, same as grpo.py) is the plain
instruction-tuned Qwen3 variant, NOT `Qwen/Qwen3-1.7B-Thinking-2507` -- it
has no built-in habit of wrapping responses in `<think>` tags, which is
exactly what makes it a fair "model that doesn't have this ability
off-the-shelf" starting point for this experiment.

Dataset: araag2/MedMCQA (config "processed"), same as grpo.py, and for the
same reason (a verifiable letter-answer reward needs a dataset with an
unambiguous ground truth) -- reused here specifically so
`experiments/` can compare "GRPO for correctness alone" against
"GRPO for correctness + reasoning format" on the *same* task.

Usage:
    python reasoning_induction_grpo.py --debug_first_batch --max_samples 8
    python reasoning_induction_grpo.py --lora --quantization 4bit --max_samples 500
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
INSTRUCTION = (
    "Answer the following medical multiple choice question. First think through your "
    "reasoning inside <think></think> tags, then give your final answer as a single "
    "letter (A, B, C, or D)."
)
THINK_BLOCK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
ANSWER_LETTER_RE = re.compile(r"\b([A-D])\b")


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for this script.

    Returns:
        argparse.ArgumentParser: Parser covering model/quantization, LoRA,
        GRPO reward and generation, training, sampling, output, and
        debug/seed options.
    """
    p = argparse.ArgumentParser(description="Induce CoT reasoning via GRPO with a correctness + format reward.")

    p.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B", help="Instruction-tuned checkpoint that does NOT already habitually use <think> tags (local path or HF Hub id). Default: Qwen/Qwen3-1.7B (plain instruct, not the -Thinking variant).")
    p.add_argument("--quantization", type=str, default="no", choices=["no", "4bit", "8bit"], help="Quantization for loading. Default: no (use 4bit alongside --lora for a real run, see rlhf/README.md).")

    p.add_argument("--lora", action="store_true", default=False, help="Enable LoRA.")
    p.add_argument("--lora_r", type=int, default=16, help="LoRA rank. Default: 16.")
    p.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha. Default: 32.")
    p.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout. Default: 0.05.")
    p.add_argument("--lora_target_modules", type=str, default="q_proj,k_proj,v_proj,o_proj", help="Comma-separated LoRA target modules.")

    p.add_argument("--format_reward_weight", type=float, default=0.3, help="Weight of the <think>-format reward relative to the correctness reward (correctness has weight 1.0). Default: 0.3 -- kept lower than correctness so the model can't game the reward by wrapping garbage in <think> tags without also getting answers right.")
    p.add_argument("--num_generations", type=int, default=4, help="Completions sampled per prompt per step. Default: 4.")
    p.add_argument("--max_completion_length", type=int, default=200, help="Max new tokens per completion -- needs real headroom for a reasoning block plus the answer, unlike grpo.py's bare-letter default. Default: 200.")
    p.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature. Default: 1.0.")
    p.add_argument("--beta", type=float, default=0.0, help="KL penalty coefficient against the reference model. Default: 0.0.")

    p.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"], help="Mixed precision. Default: bf16.")
    p.add_argument("--batch_size", type=int, default=4, help="Per-device training batch size (must be divisible by --num_generations). Default: 4.")
    p.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps. Default: 4.")
    p.add_argument("--learning_rate", type=float, default=1e-6, help="Learning rate. Default: 1e-6.")
    p.add_argument("--epochs", type=float, default=1.0, help="Training epochs. Default: 1.0.")
    p.add_argument("--gradient_checkpointing", action="store_true", default=True, help="Enable gradient checkpointing. Default: on.")

    p.add_argument("--max_samples", type=int, default=-1, help="Number of training rows to use. -1 (default) = full split.")
    p.add_argument("--sample_selection", type=str, default="random", choices=["random", "first", "last"], help="Which rows to pick when --max_samples is positive. Default: random.")
    p.add_argument("--max_eval_samples", type=int, default=100, help="Eval rows to use. Default: 100.")

    p.add_argument("--output_dir", type=str, default="./output/ability-training/CoT", help="Output directory.")
    p.add_argument("--logging_steps", type=int, default=5, help="Logging frequency. Default: 5.")
    p.add_argument("--eval_steps", type=int, default=50, help="Evaluation frequency. Default: 50.")
    p.add_argument("--save_steps", type=int, default=100, help="Checkpoint save frequency. Default: 100.")
    p.add_argument("--save_strategy", type=str, default="adapter_only", choices=["full", "adapter_only", "merged", "adapter_and_merged", "base_reference"], help="Model saving strategy. Default: adapter_only when --lora, otherwise 'full'.")

    p.add_argument("--seed", type=int, default=42, help="Random seed. Default: 42.")
    p.add_argument("--debug_first_batch", action="store_true", default=False, help="Print formatted prompts and a reward-function sanity check, then exit without training.")

    return p


def format_question(row) -> str:
    """Render a MedMCQA row's question and four options as plain text.

    Args:
        row: Mapping with "Question", "Option_A", "Option_B", "Option_C",
            and "Option_D" keys.

    Returns:
        str: The question followed by lettered options A-D, one per line.
    """
    return (
        f"{row['Question']}\n\n"
        f"A. {row['Option_A']}\n"
        f"B. {row['Option_B']}\n"
        f"C. {row['Option_C']}\n"
        f"D. {row['Option_D']}"
    )


def build_prompt(row) -> str:
    """Wrap a formatted question in the instruction/input/response template.

    Args:
        row: Mapping with "Question", "Option_A", "Option_B", "Option_C",
            and "Option_D" keys.

    Returns:
        str: Full prompt text ending at "### Response:\\n", ready for
        generation.
    """
    return f"### Instruction:\n{INSTRUCTION}\n\n### Input:\n{format_question(row)}\n\n### Response:\n"


def verify_dataset() -> None:
    """Peek at the training split via streaming and assert expected fields exist.

    Raises:
        AssertionError: If any of the expected MedMCQA fields is missing
            from the first streamed example.
    """
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
    """Load the MedMCQA train/dev splits and convert rows into GRPO prompts.

    Args:
        args (argparse.Namespace): Parsed CLI args; uses max_samples,
            sample_selection, max_eval_samples, and seed.

    Returns:
        tuple[datasets.Dataset, datasets.Dataset]: Train and eval datasets,
        each with "prompt" and "answer" columns.
    """
    print_banner("LOADING DATASET")
    train_raw = load_dataset(DATASET_NAME, DATASET_CONFIG, split="train")
    # Not split="test": its Label field is None for every row (found while building
    # rlhf/grpo/grpo.py, which shares this dataset).
    eval_raw = load_dataset(DATASET_NAME, DATASET_CONFIG, split="dev")

    train_raw = select_samples(train_raw, args.max_samples, args.sample_selection, args.seed)
    eval_raw = select_samples(eval_raw, args.max_eval_samples, "first", args.seed)

    def to_prompt_dataset(examples):
        """Map a batch of raw MedMCQA rows to {"prompt", "answer"} columns.

        Args:
            examples (dict): Batch with `Question`, `Option_A`-`Option_D`,
                and `Label` columns.

        Returns:
            dict: `{"prompt", "answer"}` — one formatted prompt and ground-truth
            answer letter per input row.
        """
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


def extract_answer_letter(completion: str):
    """Extract the model's final answer letter from a completion.

    Prefers the letter after </think> if a think block is present (the
    "final answer" should follow the reasoning); falls back to the first
    standalone letter anywhere in the completion otherwise.

    Args:
        completion (str): Raw model completion text.

    Returns:
        str | None: The matched letter (A-D), or None if no letter was found.
    """
    think_match = THINK_BLOCK_RE.search(completion)
    search_region = completion[think_match.end():] if think_match else completion
    match = ANSWER_LETTER_RE.search(search_region.strip().upper())
    return match.group(1) if match else None


def correctness_reward(prompts, completions, answer, **kwargs) -> list:
    """Verifiable GRPO reward: 1.0 if the final answer letter is correct, else 0.0.

    Args:
        prompts: Unused; accepted because TRL's GRPOTrainer always passes it
            to reward functions.
        completions (list[str]): Model completions for this batch.
        answer (list[str]): Ground-truth answer letters, aligned with
            completions.
        **kwargs: Additional TRL-supplied fields, unused.

    Returns:
        list[float]: One reward per completion, 1.0 or 0.0.
    """
    return [1.0 if extract_answer_letter(c) == a else 0.0 for c, a in zip(completions, answer)]


def format_reward(prompts, completions, **kwargs) -> list:
    """GRPO reward for well-formed, non-trivial <think>...</think> structure.

    Independent of whether the final answer is correct -- this is what
    actually induces the *habit* of producing structured reasoning, separate
    from just getting answers right.

    Args:
        prompts: Unused; accepted because TRL's GRPOTrainer always passes it
            to reward functions.
        completions (list[str]): Model completions for this batch.
        **kwargs: Additional TRL-supplied fields, unused.

    Returns:
        list[float]: One reward per completion -- 1.0 if it contains a
        <think> block with at least 3 words, else 0.0.
    """
    rewards = []
    for completion in completions:
        match = THINK_BLOCK_RE.search(completion)
        if match and len(match.group(1).strip().split()) >= 3:
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards


def print_debug_examples(train_dataset) -> None:
    """Print reward values for hand-built fake completions plus a sample prompt.

    Used by --debug_first_batch to sanity-check correctness_reward and
    format_reward before spending compute on real training.

    Args:
        train_dataset (datasets.Dataset): Training dataset with "prompt" and
            "answer" columns.
    """
    print_banner("REWARD FUNCTION SANITY CHECK")
    sample_answers = train_dataset["answer"][:3]
    fake_completions = [
        f"<think>Let's think about this. Option {sample_answers[0]} matches the classic textbook description.</think>{sample_answers[0]}",
        "The answer is probably B, not fully sure.",
        f"<think>hmm</think>{sample_answers[2] if len(sample_answers) > 2 else 'A'}",
    ]
    c_rewards = correctness_reward(None, fake_completions, sample_answers[: len(fake_completions)])
    f_rewards = format_reward(None, fake_completions)
    for ans, comp, cr, fr in zip(sample_answers, fake_completions, c_rewards, f_rewards):
        print(f"  ground truth={ans!r}")
        print(f"  completion={comp!r}")
        print(f"  correctness_reward={cr}  format_reward={fr}\n")
    print("--- Example prompt ---")
    print(train_dataset[0]["prompt"])


def evaluate_cot_usage_rate(model, tokenizer, eval_dataset, args, num_samples: int = 20) -> float:
    """Measure the fraction of greedy eval completions that use <think> tags.

    Separate from GRPOTrainer's own eval loop (which reports mean reward but
    not this specific interpretable metric): what fraction of eval
    completions spontaneously use well-formed <think> tags after training?

    Args:
        model: Trained policy model to generate from.
        tokenizer: Tokenizer/processor matching model.
        eval_dataset (datasets.Dataset): Eval dataset with a "prompt" column.
        args (argparse.Namespace): Parsed CLI args; uses max_completion_length.
        num_samples (int): Number of eval rows to sample from, capped at the
            dataset size. Defaults to 20.

    Returns:
        float: Fraction (0.0-1.0) of sampled completions containing a
        well-formed <think> block, or 0.0 if no rows were sampled.
    """
    import torch

    model.eval()
    n = min(num_samples, len(eval_dataset))
    with_think = 0
    with torch.no_grad():
        for i in range(n):
            prompt = eval_dataset[i]["prompt"]
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            output = model.generate(**inputs, max_new_tokens=args.max_completion_length, do_sample=False, pad_token_id=tokenizer.pad_token_id)
            generated = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            if format_reward(None, [generated])[0] == 1.0:
                with_think += 1
    return with_think / n if n else 0.0


def main():
    """Parse CLI args, run GRPO CoT-induction training, and write results.

    Loads data, optionally exits early for --debug_first_batch, otherwise
    builds quantization/LoRA configs, trains with GRPOTrainer, evaluates,
    measures CoT usage rate, saves the model, and writes a run_result.json.
    """
    args = build_arg_parser().parse_args()
    set_all_seeds(args.seed)
    print_config(args, "CoT/reasoning induction via GRPO (RLVR: correctness + format reward)")

    # verify_dataset()  # commented out: avoids loading the dataset twice (a streaming peek here, then a full load in load_and_prepare_data()) to cut memory overhead
    train_dataset, eval_dataset = load_and_prepare_data(args)

    if args.debug_first_batch:
        print_debug_examples(train_dataset)
        print("\n--debug_first_batch set: exiting without training.")
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
        reward_weights=[1.0, args.format_reward_weight],
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
        reward_funcs=[correctness_reward, format_reward],
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

    cot_usage_rate = evaluate_cot_usage_rate(trainer.model, trainer.processing_class, eval_dataset, args)
    print(f"\nCoT (<think>) usage rate after training: {cot_usage_rate:.2%}")

    save_strategy = args.save_strategy if args.lora else "full"
    save_model(trainer.model, trainer.processing_class, args.output_dir, strategy=save_strategy, base_model_name=args.model, is_lora=args.lora)

    metrics = {k: v for k, v in eval_metrics.items() if isinstance(v, (int, float))}
    metrics["cot_usage_rate"] = cot_usage_rate

    write_run_result(
        output_dir=args.output_dir,
        stage="ability-training",
        task="cot_induction_grpo",
        modality="text",
        architecture=ARCHITECTURE,
        cot_enabled=True,
        model_name=args.model,
        dataset_name=f"{DATASET_NAME} ({DATASET_CONFIG})",
        save_strategy=save_strategy,
        hyperparameters=vars(args),
        metrics=metrics,
        num_train_samples=len(train_dataset),
        num_eval_samples=len(eval_dataset),
        train_runtime_seconds=train_runtime_seconds,
    )
    print("Done.")


if __name__ == "__main__":
    main()
