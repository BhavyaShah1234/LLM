"""Reward Model Training -- decoder-only.

Trains a scalar reward model: given a (chosen, rejected) preference pair,
learn to score `chosen` higher than `rejected` via a Bradley-Terry pairwise
loss (`-logsigmoid(reward_chosen - reward_rejected)`). This is the
component classic RLHF (reward-model-then-PPO) needs and DPO/KTO are
specifically designed to skip -- see root README's "RLHF algorithm family"
for how these fit together. Not currently consumed by another script in
this project (no PPO trainer built yet), but useful standalone: the trained
reward model's scores are a general-purpose "how good is this response"
signal, and this project's own `reward-modeling/` output could later serve
as a real (non-verifiable) reward source for a GRPO run, in contrast to
`rlhf/grpo/grpo.py`'s hand-written
rule-based reward.

**No RLHF prerequisite here, unlike DPO/GRPO/KTO**: this trains a
classifier, not the policy itself, so there's no "policy must already
follow the target format" concern -- `--model` defaults to
`Qwen/Qwen3-1.7B-Base` (this project's usual base-model default), not an
instruction-tuned checkpoint. `trl`'s `RewardTrainer` swaps in
`AutoModelForSequenceClassification` with `num_labels=1` automatically when
given a model name string (confirmed via source inspection) -- this is
architecturally a different model than the causal LM in
`--model`'s directory (a randomly-initialized scalar head is added on top),
so `--model` really means "which pretrained backbone to attach the reward
head to," not "resume this exact checkpoint."

Dataset: trl-lib/ultrafeedback_binarized -- same dataset dpo.py uses.
Reused deliberately (see root README's dataset-reuse philosophy): reward
modeling needs the identical (chosen, rejected) pair shape DPO does, and
RewardTrainer consumes it in the same conversational format
DPOTrainer does, no reformatting needed -- confirmed via source inspection
of RewardTrainer's tokenization path.

Usage:
    python reward_modeling.py --debug_first_batch --max_samples 20
    python reward_modeling.py --lora --quantization 4bit --max_samples 2000

Memory note: unlike DPO/GRPO/KTO, RewardTrainer keeps only ONE model
resident (no reference-model copy -- there's no policy being regularized
against a reference here, just a classifier being trained), so this script
needs less headroom than the alignment scripts at the same model scale;
--quantization is optional headroom rather than effectively required.
"""

import argparse
import time

from datasets import load_dataset
from trl import RewardConfig, RewardTrainer

from common.data_selection import select_samples
from common.logging_utils import print_banner, print_config
from common.model_saving import save_model
from common.run_results import write_run_result
from common.seeding import set_all_seeds

DATASET_NAME = "trl-lib/ultrafeedback_binarized"
ARCHITECTURE = "decoder-only"


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for this script's model/LoRA/reward-modeling/training options.

    Returns:
        argparse.ArgumentParser: Parser covering model/quantization, LoRA,
        reward-modeling options (`--max_length`,
        `--center_rewards_coefficient`), training hyperparameters, data
        selection, output/checkpointing, and `--seed`/`--debug_first_batch`.
    """
    p = argparse.ArgumentParser(description="Train a scalar reward model on (chosen, rejected) preference pairs.")

    p.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B-Base", help="Backbone to attach a randomly-initialized scalar reward head to (local path or HF Hub id). No instruction-tuning prerequisite -- see module docstring. Default: Qwen/Qwen3-1.7B-Base (this project's usual base-model default).")
    p.add_argument("--quantization", type=str, default="no", choices=["no", "4bit", "8bit"], help="Quantization for loading. Default: no.")

    p.add_argument("--lora", action="store_true", default=False, help="Enable LoRA on the backbone (the reward head itself is always trained in full precision).")
    p.add_argument("--lora_r", type=int, default=16, help="LoRA rank. Default: 16.")
    p.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha. Default: 32.")
    p.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout. Default: 0.05.")
    p.add_argument("--lora_target_modules", type=str, default="q_proj,k_proj,v_proj,o_proj", help="Comma-separated LoRA target modules.")

    p.add_argument("--max_length", type=int, default=512, help="Max sequence length per (prompt+response) side. Default: 512.")
    p.add_argument("--center_rewards_coefficient", type=float, default=None, help="Optional auxiliary loss coefficient pulling (reward_chosen + reward_rejected) toward 0, to prevent reward-score drift. Default: none (off).")

    p.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"], help="Mixed precision. Default: bf16.")
    p.add_argument("--batch_size", type=int, default=4, help="Per-device training batch size. Default: 4 (no reference-model copy resident, unlike dpo.py/kto.py, so more headroom is available at the same model scale).")
    p.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps. Default: 4.")
    p.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate. Default: 1e-5 (typical for reward-model training -- between SFT and DPO's LR regimes).")
    p.add_argument("--epochs", type=float, default=1.0, help="Training epochs. Default: 1.0.")
    p.add_argument("--gradient_checkpointing", action="store_true", default=True, help="Enable gradient checkpointing. Default: on.")

    p.add_argument("--max_samples", type=int, default=-1, help="Number of training rows to use. -1 (default) = full split.")
    p.add_argument("--sample_selection", type=str, default="random", choices=["random", "first", "last"], help="Which rows to pick when --max_samples is positive. Default: random.")
    p.add_argument("--max_eval_samples", type=int, default=200, help="Eval rows to use. Default: 200.")

    p.add_argument("--output_dir", type=str, default="./output/rlhf/reward-modeling", help="Output directory.")
    p.add_argument("--logging_steps", type=int, default=10, help="Logging frequency. Default: 10.")
    p.add_argument("--eval_steps", type=int, default=100, help="Evaluation frequency. Default: 100.")
    p.add_argument("--save_steps", type=int, default=200, help="Checkpoint save frequency. Default: 200.")
    p.add_argument("--save_strategy", type=str, default="adapter_only", choices=["full", "adapter_only", "merged", "adapter_and_merged", "base_reference"], help="Model saving strategy. Default: adapter_only when --lora, otherwise 'full'.")

    p.add_argument("--seed", type=int, default=42, help="Random seed. Default: 42.")
    p.add_argument("--debug_first_batch", action="store_true", default=False, help="Print formatted preference pairs and exit without training.")

    return p


def verify_dataset() -> None:
    """Stream-peek one row of the dataset and assert the expected preference fields are present."""
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
    """Load the train/test splits and subsample them per the CLI args.

    Args:
        args (argparse.Namespace): Parsed CLI args; uses `max_samples`,
            `sample_selection`, `max_eval_samples`, and `seed`.

    Returns:
        tuple: `(train_dataset, eval_dataset)`, each a subsampled `Dataset`
        of chosen/rejected preference pairs.
    """
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
    """Print a few formatted prompt/chosen/rejected triples for a sanity check.

    Args:
        dataset: A chosen/rejected preference dataset to sample from.
        num_examples (int): Number of rows to print. Defaults to 2.
    """
    print_banner("FORMATTED PREFERENCE PAIRS")
    for i in range(min(num_examples, len(dataset))):
        example = dataset[i]
        prompt_text = example["chosen"][0]["content"]
        chosen_text = example["chosen"][-1]["content"]
        rejected_text = example["rejected"][-1]["content"]
        print(f"\n--- Example {i + 1} ---")
        print(f"Prompt: {prompt_text[:200]!r}")
        print(f"Chosen:   {chosen_text[:200]!r}")
        print(f"Rejected: {rejected_text[:200]!r}")
    print()


def main():
    """Run the reward-model training pipeline: load data, build the trainer, train, evaluate, save, and record results."""
    args = build_arg_parser().parse_args()
    set_all_seeds(args.seed)
    print_config(args, "Reward model training -- decoder-only")

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

        # task_type="SEQ_CLS", not the common/ default "CAUSAL_LM": the backbone here is
        # AutoModelForSequenceClassification (RewardTrainer's own model swap, see module
        # docstring), which has no .generate()/prepare_inputs_for_generation. peft's
        # get_peft_model() assumes CAUSAL_LM models are generation-capable and crashes with
        # AttributeError: 'Qwen3ForSequenceClassification' object has no attribute
        # 'prepare_inputs_for_generation' otherwise -- confirmed via a live traceback.
        peft_config = build_lora_config(args.lora_r, args.lora_alpha, args.lora_dropout, args.lora_target_modules, task_type="SEQ_CLS")

    training_args = RewardConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        center_rewards_coefficient=args.center_rewards_coefficient,
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

    trainer = RewardTrainer(
        model=args.model,
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
        task="reward_modeling",
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
