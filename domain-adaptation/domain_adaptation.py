"""Domain adaptation -- decoder-only (CLM), specializing a real pretrained
model to a specific domain's text distribution.

Distinct from continued-pretraining/continued_pretraining.py (this project's
other Wave-3 script), which continues a model on *more general* text of the
same kind it was already trained on. This script instead specializes a
capable, real pretrained model (`Qwen/Qwen3-1.7B-Base` by default) to a
*narrow domain* -- medical text -- via the same CLM objective, on the theory
that a downstream medical task (e.g. supervised-finetuning/text/mcq/'s
MedMCQA scripts) will do better starting from a base model that has already
seen a lot of in-domain language, not just general web/book text.

Dataset: the `Explanation` field of araag2/MedMCQA (config "processed") --
the same dataset supervised-finetuning/text/mcq/decoder-only/mcq_standard.py
uses, reused here as a plain medical-text corpus (its long clinical-reasoning
explanations, not the question/option/answer structure). About 11% of rows
have an empty Explanation (confirmed empirically), filtered out during
ingestion.

Since this project's model-selection philosophy defaults to a small-but-real
1.7B model here (unlike continued_pretraining.py's toy from-scratch
checkpoint), full-parameter training does not fit in 8GB VRAM -- confirmed
empirically while building supervised-finetuning/ (see its README)  --
`--lora` (optionally with `--quantization 4bit`) is the realistic default
for this script, same as every supervised-finetuning/ decoder-only script.

Usage:
    python domain_adaptation.py --debug_first_batch --max_samples 20
    python domain_adaptation.py --lora --max_samples 2000
"""

import argparse
import math
import time

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments

from common.data_selection import select_samples
from common.logging_utils import print_banner, print_config, print_formatted_examples
from common.model_saving import save_model
from common.quantization import build_quantization_config
from common.run_results import write_run_result
from common.seeding import set_all_seeds

DATASET_NAME = "araag2/MedMCQA"
DATASET_CONFIG = "processed"
ARCHITECTURE = "decoder-only"


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for this script.

    Returns:
        argparse.ArgumentParser: Parser covering model/quantization, LoRA,
        packing/sampling, training, and output/debug options.
    """
    p = argparse.ArgumentParser(description="Domain-adapt a decoder-only checkpoint (CLM objective) to medical text.")

    p.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B-Base", help="Base checkpoint to domain-adapt. Default: Qwen/Qwen3-1.7B-Base.")
    p.add_argument("--quantization", type=str, default="no", choices=["no", "4bit", "8bit"], help="Quantization for loading. Default: no (use 4bit alongside --lora if VRAM is tight).")

    p.add_argument("--lora", action="store_true", default=False, help="Enable LoRA. Recommended: full-parameter CLM training of this model doesn't fit in 8GB (confirmed empirically).")
    p.add_argument("--lora_r", type=int, default=16, help="LoRA rank. Default: 16.")
    p.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha. Default: 32.")
    p.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout. Default: 0.05.")
    p.add_argument("--lora_target_modules", type=str, default="q_proj,k_proj,v_proj,o_proj", help="Comma-separated LoRA target modules.")

    p.add_argument("--block_size", type=int, default=256, help="Context length in tokens per packed training example. Default: 256.")
    p.add_argument("--max_samples", type=int, default=-1, help="Number of raw dataset rows (explanations) to use before packing. -1 (default) = use the full split.")
    p.add_argument("--sample_selection", type=str, default="random", choices=["random", "first", "last"], help="Which rows to pick when --max_samples is positive. Default: random.")
    p.add_argument("--max_eval_samples", type=int, default=500, help="Number of dev-split rows to use for evaluation. Default: 500.")

    p.add_argument("--batch_size", type=int, default=2, help="Per-device training batch size. Default: 2 (kept small -- see supervised-finetuning/README.md's OOM notes for this model size on 8GB VRAM).")
    p.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Gradient accumulation steps. Default: 8.")
    p.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate. Default: 1e-4.")
    p.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay. Default: 0.01.")
    p.add_argument("--warmup_steps", type=int, default=100, help="LR warmup steps. Default: 100.")
    p.add_argument("--epochs", type=float, default=1.0, help="Training epochs over the (possibly truncated) train split. Ignored if --max_steps is set. Default: 1.0.")
    p.add_argument("--max_steps", type=int, default=-1, help="If set (>0), overrides --epochs. Default: -1 (unset).")

    p.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"], help="Mixed precision mode. Default: bf16.")
    p.add_argument("--gradient_checkpointing", action="store_true", default=True, help="Enable gradient checkpointing. Default: on.")
    p.add_argument("--seed", type=int, default=42, help="Random seed. Default: 42.")
    p.add_argument("--output_dir", type=str, default="./output/domain-adaptation", help="Where to save the domain-adapted model, tokenizer, and run_result.json.")
    p.add_argument("--logging_steps", type=int, default=10, help="Logging frequency. Default: 10.")
    p.add_argument("--eval_steps", type=int, default=100, help="Evaluation frequency. Default: 100.")
    p.add_argument("--save_steps", type=int, default=200, help="Checkpoint save frequency. Default: 200.")
    p.add_argument("--save_strategy", type=str, default="adapter_only", choices=["full", "adapter_only", "merged", "adapter_and_merged", "base_reference"], help="Model saving strategy (see common/model_saving.py). Default: adapter_only when --lora, otherwise 'full'.")

    p.add_argument("--debug_first_batch", action="store_true", default=False, help="Print formatted examples and exit without training.")

    return p


def verify_dataset() -> None:
    """Peek at the training split via streaming and assert an "Explanation" field exists.

    Raises:
        AssertionError: If the first streamed example has no "Explanation"
            field.
    """
    print_banner("VERIFYING DATASET")
    peek = load_dataset(DATASET_NAME, DATASET_CONFIG, split="train", streaming=True)
    example = next(iter(peek))
    assert "Explanation" in example, f"Expected an 'Explanation' field in {DATASET_NAME}/{DATASET_CONFIG}, got fields: {list(example.keys())}"
    print(f"Dataset: {DATASET_NAME} (config={DATASET_CONFIG})")
    print(f"Fields: {list(example.keys())}")
    print(f"Sample explanation: {example['Explanation'][:150]!r}")
    print()


def tokenize_and_pack(dataset, tokenizer, block_size: int, desc: str):
    """Tokenize MedMCQA explanations and pack them into fixed-length CLM blocks.

    Args:
        dataset (datasets.Dataset): Raw dataset with an "Explanation" column.
        tokenizer: Tokenizer to encode text with.
        block_size (int): Number of tokens per packed training block.
        desc (str): Short label used in progress-bar descriptions (e.g.
            "train" or "eval").

    Returns:
        datasets.Dataset: Dataset of fixed-length "input_ids"/"labels"
        blocks, with any trailing tokens shorter than block_size dropped.
    """
    def tokenize_fn(examples):
        """Tokenize a batch, dropping empty explanations and appending EOS."""
        non_empty = [t + tokenizer.eos_token for t in examples["Explanation"] if t and t.strip()]
        return tokenizer(non_empty)

    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=dataset.column_names, desc=f"Tokenizing ({desc})")

    def group_texts(examples):
        """Concatenate a tokenized batch and split it into block_size chunks."""
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = (len(concatenated["input_ids"]) // block_size) * block_size
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    return tokenized.map(group_texts, batched=True, desc=f"Packing into {block_size}-token blocks ({desc})")


def load_and_prepare_data(args, tokenizer):
    """Load the MedMCQA train/dev splits and tokenize+pack their explanations.

    Args:
        args (argparse.Namespace): Parsed CLI args; uses max_samples,
            sample_selection, max_eval_samples, seed, and block_size.
        tokenizer: Tokenizer to encode text with.

    Returns:
        tuple[datasets.Dataset, datasets.Dataset]: Packed train and eval
        datasets of fixed-length blocks.
    """
    print_banner("LOADING DATASET")
    raw_train = load_dataset(DATASET_NAME, DATASET_CONFIG, split="train")
    raw_eval = load_dataset(DATASET_NAME, DATASET_CONFIG, split="dev")

    raw_train = select_samples(raw_train, args.max_samples, args.sample_selection, args.seed)
    raw_eval = select_samples(raw_eval, args.max_eval_samples, "first", args.seed)

    print(f"Train rows (before packing, before empty-explanation filtering): {len(raw_train)}")
    print(f"Eval rows (before packing, before empty-explanation filtering): {len(raw_eval)}")

    train_dataset = tokenize_and_pack(raw_train, tokenizer, args.block_size, "train")
    eval_dataset = tokenize_and_pack(raw_eval, tokenizer, args.block_size, "eval")

    print(f"Train blocks (after packing to {args.block_size} tokens): {len(train_dataset)}")
    print(f"Eval blocks (after packing to {args.block_size} tokens): {len(eval_dataset)}")
    print()
    return train_dataset, eval_dataset


def decode_example(example, index, tokenizer):
    """Decode a single packed block back to text for --debug_first_batch display.

    Args:
        example (dict): One packed dataset row with an "input_ids" field.
        index (int): Row index; unused, present for print_formatted_examples'
            decode_fn signature.
        tokenizer: Tokenizer to decode input_ids with.

    Returns:
        str: Human-readable summary of the block's token count and text.
    """
    text = tokenizer.decode(example["input_ids"])
    return f"Packed block ({len(example['input_ids'])} tokens):\n{text}"


def main():
    """Parse CLI args, run domain-adaptation training, and write results.

    Loads the tokenizer/checkpoint, prepares packed data, optionally exits
    early for --debug_first_batch, otherwise applies quantization/LoRA,
    trains with Trainer, evaluates perplexity, saves the model, and writes a
    run_result.json.
    """
    args = build_arg_parser().parse_args()
    set_all_seeds(args.seed)
    print_config(args, "Domain adaptation (decoder-only, CLM) -- specializing to medical text")

    # verify_dataset()  # commented out: avoids loading the dataset twice (a streaming peek here, then a full load in load_and_prepare_data()) to cut memory overhead

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset, eval_dataset = load_and_prepare_data(args, tokenizer)

    quant_config = build_quantization_config(args.quantization, args.mixed_precision) if args.quantization != "no" else None
    torch_dtype = torch.bfloat16 if args.mixed_precision == "bf16" else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=quant_config,
        dtype=torch_dtype,
        device_map="auto" if quant_config is not None else None,
    )
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    if args.lora:
        from common.peft_setup import apply_lora, build_lora_config

        lora_config = build_lora_config(args.lora_r, args.lora_alpha, args.lora_dropout, args.lora_target_modules)
        model = apply_lora(model, lora_config, prepare_for_kbit=(quant_config is not None))

    total_params = sum(p.numel() for p in model.parameters())
    print_banner("MODEL")
    print(f"Domain-adapting: {args.model}")
    print(f"Total parameters: {total_params:,}")
    print()

    if args.debug_first_batch:
        print_formatted_examples(train_dataset, tokenizer, num_examples=2, decode_fn=decode_example)
        print("--debug_first_batch set: exiting without training.")
        return

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
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
        ddp_find_unused_parameters=(True if args.lora else None),
    )

    trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=data_collator)

    print_banner("TRAINING")
    start = time.time()
    trainer.train()
    train_runtime_seconds = time.time() - start

    print_banner("FINAL EVALUATION")
    eval_metrics = trainer.evaluate()
    eval_loss = eval_metrics["eval_loss"]
    perplexity = math.exp(eval_loss) if eval_loss < 20 else float("inf")
    print(f"Eval loss: {eval_loss:.4f}  |  Perplexity: {perplexity:.2f}")

    save_strategy = args.save_strategy if args.lora else "full"
    save_model(model, tokenizer, args.output_dir, strategy=save_strategy, base_model_name=args.model, is_lora=args.lora)

    write_run_result(
        output_dir=args.output_dir,
        stage="domain-adaptation",
        task="clm",
        modality="text",
        architecture=ARCHITECTURE,
        model_name=args.model,
        dataset_name=f"{DATASET_NAME} ({DATASET_CONFIG}, Explanation field)",
        save_strategy=save_strategy,
        hyperparameters=vars(args),
        metrics={"eval_loss": eval_loss, "perplexity": perplexity, "total_parameters": total_params},
        num_train_samples=len(train_dataset),
        num_eval_samples=len(eval_dataset),
        train_runtime_seconds=train_runtime_seconds,
    )
    print("Done.")


if __name__ == "__main__":
    main()
