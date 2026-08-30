"""Kahneman-Tversky Optimization (KTO) -- decoder-only.

Aligns a model to human preferences using UNPAIRED binary feedback --
each example is a single (prompt, completion, label) row where label is
just "desirable" or "undesirable", not a (chosen, rejected) pair for the
same prompt like DPO needs. This is the practical advantage KTO has over
DPO: unpaired thumbs-up/thumbs-down feedback (e.g. real product usage logs)
is far easier to collect than matched preference pairs -- see root README's
"RLHF algorithm family" section for the theory (Kahneman-Tversky prospect
theory: humans weigh losses more heavily than equivalent gains, and KTO's
loss function is derived directly from that asymmetry via
--desirable_weight / --undesirable_weight, rather than from a Bradley-Terry
pairwise-comparison model like DPO/reward modeling use).

**Prerequisite** (same as dpo.py/grpo.py -- see
rlhf/README.md): KTO assumes the
policy model already follows the target response format/behavior.
`--model` defaults to `Qwen/Qwen3-1.7B` (vendor instruction-tuned), NOT the
base model.

Dataset: trl-lib/kto-mix-14k -- the standard TRL example dataset for KTO.
Fields: prompt (conversational list), completion (conversational list),
label (bool: True=desirable, False=undesirable). 13,500 train / 1,500 test
rows, balanced 50/50 on label -- verified live before writing this script
(trl-lib/ultrafeedback_binarized, used for dpo.py, is NOT reusable here:
KTO needs a label field per (prompt, completion) row, not chosen/rejected
pairs).

Usage:
    python kto.py --debug_first_batch --max_samples 20
    python kto.py --lora --quantization 4bit --max_samples 2000

Memory note (same finding as dpo.py, confirmed empirically): KTOTrainer
loads a policy model AND a reference model, so --quantization 4bit is
effectively required at this model scale on an 8GB GPU, not just optional
headroom.
"""

import argparse
import time

from datasets import load_dataset
from trl import KTOConfig, KTOTrainer

from common.data_selection import select_samples
from common.logging_utils import print_banner, print_config
from common.model_saving import save_model
from common.run_results import write_run_result
from common.seeding import set_all_seeds

DATASET_NAME = "trl-lib/kto-mix-14k"
ARCHITECTURE = "decoder-only"


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for this script's model/LoRA/KTO/training options.

    Returns:
        argparse.ArgumentParser: Parser covering model/quantization, LoRA,
        KTO (`--beta`, `--desirable_weight`, `--undesirable_weight`,
        `--max_length`), training hyperparameters, data selection,
        output/checkpointing, and `--seed`/`--debug_first_batch`.
    """
    p = argparse.ArgumentParser(description="Align a decoder-only model with KTO (unpaired binary feedback).")

    p.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B", help="SFT'd/instruction-tuned checkpoint to align (local path or HF Hub id) -- NOT a raw base model, see module docstring. Default: Qwen/Qwen3-1.7B (vendor instruct, fp16 ~4.1GB).")
    p.add_argument("--quantization", type=str, default="no", choices=["no", "4bit", "8bit"], help="Quantization for loading. Default: no.")

    p.add_argument("--lora", action="store_true", default=False, help="Enable LoRA. Recommended: full-parameter KTO of a 1.7-4B model needs a reference-model copy in memory too, doubling the footprint vs. SFT.")
    p.add_argument("--lora_r", type=int, default=16, help="LoRA rank. Default: 16.")
    p.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha. Default: 32.")
    p.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout. Default: 0.05.")
    p.add_argument("--lora_target_modules", type=str, default="q_proj,k_proj,v_proj,o_proj", help="Comma-separated LoRA target modules.")

    p.add_argument("--beta", type=float, default=0.1, help="KTO temperature (controls how strongly the policy is pulled from the reference model, same role as DPO's beta). Default: 0.1.")
    p.add_argument("--desirable_weight", type=float, default=1.0, help="Loss weight on desirable (label=True) examples -- KTOConfig's mechanism for the Kahneman-Tversky loss-aversion asymmetry (raise --undesirable_weight relative to this to penalize bad completions more than an equivalent DPO-style comparison would). Default: 1.0.")
    p.add_argument("--undesirable_weight", type=float, default=1.0, help="Loss weight on undesirable (label=False) examples. Default: 1.0.")
    p.add_argument("--max_length", type=int, default=512, help="Max total sequence length (prompt + completion combined -- KTOConfig has no separate max_prompt_length field, confirmed via introspection, same as DPOConfig/GRPOConfig in this trl version). Default: 512.")

    p.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"], help="Mixed precision. Default: bf16.")
    p.add_argument("--batch_size", type=int, default=1, help="Per-device training batch size. Default: 1 (KTO keeps a reference-model copy resident -- see --lora note above).")
    p.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Gradient accumulation steps. Default: 8.")
    p.add_argument("--learning_rate", type=float, default=5e-6, help="Learning rate. Default: 5e-6 (same low LR regime as DPO).")
    p.add_argument("--epochs", type=float, default=1.0, help="Training epochs. Default: 1.0.")
    p.add_argument("--gradient_checkpointing", action="store_true", default=True, help="Enable gradient checkpointing. Default: on.")

    p.add_argument("--max_samples", type=int, default=-1, help="Number of training rows to use. -1 (default) = full split.")
    p.add_argument("--sample_selection", type=str, default="random", choices=["random", "first", "last"], help="Which rows to pick when --max_samples is positive. Default: random.")
    p.add_argument("--max_eval_samples", type=int, default=200, help="Eval rows to use. Default: 200.")

    p.add_argument("--output_dir", type=str, default="./output/rlhf/kto", help="Output directory.")
    p.add_argument("--logging_steps", type=int, default=10, help="Logging frequency. Default: 10.")
    p.add_argument("--eval_steps", type=int, default=100, help="Evaluation frequency. Default: 100.")
    p.add_argument("--save_steps", type=int, default=200, help="Checkpoint save frequency. Default: 200.")
    p.add_argument("--save_strategy", type=str, default="adapter_only", choices=["full", "adapter_only", "merged", "adapter_and_merged", "base_reference"], help="Model saving strategy. Default: adapter_only when --lora, otherwise 'full'.")

    p.add_argument("--seed", type=int, default=42, help="Random seed. Default: 42.")
    p.add_argument("--debug_first_batch", action="store_true", default=False, help="Print formatted (prompt, completion, label) examples and exit without training.")

    return p


def verify_dataset() -> None:
    """Stream-peek one row of the dataset and assert the expected prompt/completion/label fields are present."""
    print_banner("VERIFYING DATASET")
    peek = load_dataset(DATASET_NAME, split="train", streaming=True)
    example = next(iter(peek))
    for field in ("prompt", "completion", "label"):
        assert field in example, f"Expected a {field!r} field in {DATASET_NAME}, got fields: {list(example.keys())}"
    print(f"Dataset: {DATASET_NAME}")
    print(f"Fields: {list(example.keys())}")
    print(f"Sample prompt[-1]: {example['prompt'][-1]['content'][:150]!r}")
    print(f"Sample completion[-1]: {example['completion'][-1]['content'][:150]!r}")
    print(f"Sample label: {example['label']!r}")
    print()


def load_and_prepare_data(args):
    """Load the train/test splits and subsample them per the CLI args.

    Args:
        args (argparse.Namespace): Parsed CLI args; uses `max_samples`,
            `sample_selection`, `max_eval_samples`, and `seed`.

    Returns:
        tuple: `(train_dataset, eval_dataset)`, each a subsampled `Dataset`
        of (prompt, completion, label) rows.
    """
    print_banner("LOADING DATASET")
    train_dataset = load_dataset(DATASET_NAME, split="train")
    eval_dataset = load_dataset(DATASET_NAME, split="test")

    train_dataset = select_samples(train_dataset, args.max_samples, args.sample_selection, args.seed)
    eval_dataset = select_samples(eval_dataset, args.max_eval_samples, "first", args.seed)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Eval samples: {len(eval_dataset)}")
    n_desirable = sum(train_dataset["label"])
    print(f"Train label balance: {n_desirable} desirable / {len(train_dataset) - n_desirable} undesirable")
    print()
    return train_dataset, eval_dataset


def print_debug_examples(dataset, num_examples: int = 2) -> None:
    """Print a few formatted (prompt, completion, label) rows for a sanity check.

    Args:
        dataset: A (prompt, completion, label) dataset to sample from.
        num_examples (int): Number of rows to print. Defaults to 2.
    """
    print_banner("FORMATTED (PROMPT, COMPLETION, LABEL) EXAMPLES")
    for i in range(min(num_examples, len(dataset))):
        example = dataset[i]
        prompt_text = example["prompt"][-1]["content"]
        completion_text = example["completion"][-1]["content"]
        print(f"\n--- Example {i + 1} ---")
        print(f"Prompt:     {prompt_text[:200]!r}")
        print(f"Completion: {completion_text[:200]!r}")
        print(f"Label (desirable={example['label']})")
    print()


def main():
    """Run the KTO alignment pipeline: load data, build the trainer, train, evaluate, save, and record results."""
    args = build_arg_parser().parse_args()
    set_all_seeds(args.seed)
    print_config(args, "KTO alignment -- decoder-only")

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

    training_args = KTOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        beta=args.beta,
        desirable_weight=args.desirable_weight,
        undesirable_weight=args.undesirable_weight,
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

    trainer = KTOTrainer(
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
        task="kto",
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
