"""Causal Language Modelling (CLM) pretraining -- decoder-only architecture.

Trains a small, GPT-style transformer *from scratch* (random initialization,
no pretrained weights) on next-token prediction with a causal attention mask.
This is the objective decoder-only models (GPT, Llama, Qwen, ...) are
pretrained with.

Toy-scale by design: on a single 8GB GPU, a from-scratch run here is meant to
teach the CLM mechanics and provide one point of comparison in
experiments/pretraining-objective-comparison/, not to produce a competitive
language model -- see pretraining/README.md.

Usage:
    python clm.py --debug_first_batch --max_samples 20
    python clm.py --max_steps 500 --output_dir ./output/pretraining/clm
"""

import argparse
import math
import time

from datasets import load_dataset
from transformers import (
    DataCollatorForLanguageModeling,
    GPT2Config,
    Trainer,
    TrainingArguments,
)

from common.data_selection import select_samples
from common.logging_utils import print_banner, print_config, print_formatted_examples
from common.model_loading import build_model_from_scratch, load_tokenizer
from common.model_saving import save_model
from common.run_results import write_run_result
from common.seeding import set_all_seeds

DATASET_NAME = "roneneldan/TinyStories"
TOKENIZER_NAME = "gpt2"
ARCHITECTURE = "decoder-only"


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for this script.

    Returns:
        argparse.ArgumentParser: Parser covering architecture sizing, data,
        training hyperparameters, system, and debug options.
    """
    p = argparse.ArgumentParser(description="Pretrain a small decoder-only model from scratch (CLM objective).")

    # Architecture sizing (from-scratch config, not a pretrained checkpoint choice)
    p.add_argument("--hidden_size", type=int, default=512, help="Transformer hidden size (n_embd). Default: 512.")
    p.add_argument("--num_layers", type=int, default=8, help="Number of transformer blocks (n_layer). Default: 8.")
    p.add_argument("--num_attention_heads", type=int, default=8, help="Number of attention heads (n_head). Default: 8.")
    p.add_argument("--block_size", type=int, default=256, help="Context length in tokens per packed training example. Default: 256.")

    # Data
    p.add_argument("--max_samples", type=int, default=-1, help="Number of raw dataset rows (stories) to use before tokenization/packing. -1 (default) = use the full split.")
    p.add_argument("--sample_selection", type=str, default="random", choices=["random", "first", "last"], help="Which rows to pick when --max_samples is positive. Default: random.")
    p.add_argument("--max_eval_samples", type=int, default=2000, help="Number of validation rows to use for evaluation (kept small for speed). Default: 2000.")

    # Training hyperparameters
    p.add_argument("--batch_size", type=int, default=16, help="Per-device training batch size. Default: 16.")
    p.add_argument("--gradient_accumulation_steps", type=int, default=2, help="Gradient accumulation steps. Default: 2.")
    p.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate. Default: 3e-4 (typical for from-scratch small-model pretraining, higher than SFT LRs).")
    p.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay. Default: 0.01.")
    p.add_argument("--warmup_steps", type=int, default=200, help="LR warmup steps. Default: 200.")
    p.add_argument("--epochs", type=float, default=1.0, help="Number of training epochs over the (possibly truncated) train split. Ignored if --max_steps is set. Default: 1.0.")
    p.add_argument("--max_steps", type=int, default=-1, help="If set (>0), overrides --epochs and stops after this many optimizer steps. Default: -1 (unset, use --epochs).")

    # System
    p.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"], help="Mixed precision mode. Default: bf16.")
    p.add_argument("--gradient_checkpointing", action="store_true", default=False, help="Enable gradient checkpointing. Default: off (small model, not usually needed).")
    p.add_argument("--seed", type=int, default=42, help="Random seed. Default: 42.")
    p.add_argument("--output_dir", type=str, default="./output/pretraining/clm", help="Where to save the trained model, tokenizer, and run_result.json.")
    p.add_argument("--logging_steps", type=int, default=25, help="Logging frequency. Default: 25.")
    p.add_argument("--eval_steps", type=int, default=250, help="Evaluation frequency. Default: 250.")
    p.add_argument("--save_steps", type=int, default=500, help="Checkpoint save frequency. Default: 500.")

    # Debugging
    p.add_argument("--debug_first_batch", action="store_true", default=False, help="Load data, build the model, print formatted examples and the model's real parameter count, then exit without training.")

    return p


def verify_dataset() -> None:
    """Peek one streamed example before committing to a full (non-streaming) load.

    Raises:
        AssertionError: If the first streamed example has no "text" field.
    """
    print_banner("VERIFYING DATASET")
    peek = load_dataset(DATASET_NAME, split="train", streaming=True)
    example = next(iter(peek))
    assert "text" in example, f"Expected a 'text' field in {DATASET_NAME}, got fields: {list(example.keys())}"
    print(f"Dataset: {DATASET_NAME}")
    print(f"Fields: {list(example.keys())}")
    print(f"Sample text: {example['text'][:200]!r}")
    print()


def tokenize_and_pack(dataset, tokenizer, block_size: int, desc: str):
    """Tokenize raw story text and pack it into fixed-length CLM blocks.

    Args:
        dataset (datasets.Dataset): Raw dataset with a "text" column.
        tokenizer: Tokenizer to encode text with.
        block_size (int): Number of tokens per packed training block.
        desc (str): Short label used in progress-bar descriptions (e.g.
            "train" or "eval").

    Returns:
        datasets.Dataset: Dataset of fixed-length "input_ids"/"labels"
        blocks, with any trailing tokens shorter than block_size dropped.
    """
    def tokenize_fn(examples):
        """Append EOS to each story and tokenize the batch.

        Args:
            examples (dict): Batch with a `text` column.

        Returns:
            dict: Tokenizer output (`input_ids`, `attention_mask`, ...).
        """
        texts_with_eos = [t + tokenizer.eos_token for t in examples["text"]]
        return tokenizer(texts_with_eos)

    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=dataset.column_names, desc=f"Tokenizing ({desc})")

    def group_texts(examples):
        """Concatenate all tokenized examples in the batch and split into fixed blocks.

        Args:
            examples (dict): Batch of tokenizer output columns.

        Returns:
            dict: Same columns rechunked to `block_size`, plus `labels` (a
            copy of `input_ids`); any remainder shorter than `block_size`
            is dropped.
        """
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
    """Load the TinyStories train/validation splits and tokenize+pack them.

    Args:
        args (argparse.Namespace): Parsed CLI args; uses max_samples,
            sample_selection, max_eval_samples, seed, and block_size.
        tokenizer: Tokenizer to encode text with.

    Returns:
        tuple[datasets.Dataset, datasets.Dataset]: Packed train and eval
        datasets of fixed-length blocks.
    """
    print_banner("LOADING DATASET")
    raw_train = load_dataset(DATASET_NAME, split="train")
    raw_eval = load_dataset(DATASET_NAME, split="validation")

    raw_train = select_samples(raw_train, args.max_samples, args.sample_selection, args.seed)
    raw_eval = select_samples(raw_eval, args.max_eval_samples, "first", args.seed)

    print(f"Train rows (before packing): {len(raw_train)}")
    print(f"Eval rows (before packing): {len(raw_eval)}")

    train_dataset = tokenize_and_pack(raw_train, tokenizer, args.block_size, "train")
    eval_dataset = tokenize_and_pack(raw_eval, tokenizer, args.block_size, "eval")

    print(f"Train blocks (after packing to {args.block_size} tokens): {len(train_dataset)}")
    print(f"Eval blocks (after packing to {args.block_size} tokens): {len(eval_dataset)}")
    print()
    return train_dataset, eval_dataset


def build_model(args, tokenizer):
    """Construct a randomly initialized GPT2-style model sized by CLI args.

    Args:
        args (argparse.Namespace): Parsed CLI args; uses hidden_size,
            num_layers, num_attention_heads, block_size, and
            gradient_checkpointing.
        tokenizer: Tokenizer used to size the vocabulary and set special
            token ids.

    Returns:
        transformers.PreTrainedModel: Freshly initialized decoder-only model.
    """
    config = GPT2Config(
        vocab_size=len(tokenizer),
        n_positions=args.block_size,
        n_embd=args.hidden_size,
        n_layer=args.num_layers,
        n_head=args.num_attention_heads,
        bos_token_id=tokenizer.bos_token_id or tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    model = build_model_from_scratch(ARCHITECTURE, config, gradient_checkpointing=args.gradient_checkpointing)

    total_params = sum(p.numel() for p in model.parameters())
    embedding_params = model.transformer.wte.weight.numel() + model.transformer.wpe.weight.numel()
    print_banner("MODEL (FROM SCRATCH, RANDOM INIT)")
    print(f"Architecture: {ARCHITECTURE} (GPT2-style), vocab_size={config.vocab_size}, "
          f"hidden_size={config.n_embd}, num_layers={config.n_layer}, num_heads={config.n_head}")
    print(f"Total parameters:      {total_params:,}")
    print(f"Embedding parameters:  {embedding_params:,} ({100 * embedding_params / total_params:.1f}% of total)")
    print(f"Non-embedding params:  {total_params - embedding_params:,}")
    print()
    return model


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
    """Parse CLI args, run from-scratch CLM pretraining, and write results.

    Loads the tokenizer and data, builds a randomly initialized model,
    optionally exits early for --debug_first_batch, otherwise trains with
    Trainer, evaluates perplexity, saves the model, and writes a
    run_result.json.
    """
    args = build_arg_parser().parse_args()
    set_all_seeds(args.seed)
    print_config(args, "Causal Language Modelling pretraining (decoder-only, from scratch)")

    # verify_dataset()  # commented out: avoids loading the dataset twice (a streaming peek here, then a full load in load_and_prepare_data()) to cut memory overhead
    tokenizer = load_tokenizer(TOKENIZER_NAME)

    train_dataset, eval_dataset = load_and_prepare_data(args, tokenizer)
    model = build_model(args, tokenizer)

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
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    print_banner("TRAINING")
    start = time.time()
    trainer.train()
    train_runtime_seconds = time.time() - start

    print_banner("FINAL EVALUATION")
    eval_metrics = trainer.evaluate()
    eval_loss = eval_metrics["eval_loss"]
    perplexity = math.exp(eval_loss) if eval_loss < 20 else float("inf")
    print(f"Eval loss: {eval_loss:.4f}  |  Perplexity: {perplexity:.2f}")

    save_model(model, tokenizer, args.output_dir, strategy="full")

    total_params = sum(p.numel() for p in model.parameters())
    write_run_result(
        output_dir=args.output_dir,
        stage="pretraining",
        task="clm",
        modality="text",
        architecture=ARCHITECTURE,
        model_name=f"from-scratch-gpt2-style-{total_params}params",
        dataset_name=DATASET_NAME,
        hyperparameters=vars(args),
        metrics={"eval_loss": eval_loss, "perplexity": perplexity, "total_parameters": total_params},
        num_train_samples=len(train_dataset),
        num_eval_samples=len(eval_dataset),
        train_runtime_seconds=train_runtime_seconds,
    )
    print("Done.")


if __name__ == "__main__":
    main()
