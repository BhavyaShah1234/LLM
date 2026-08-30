"""Masked Language Modelling (MLM) pretraining -- encoder-only architecture.

Trains a small, BERT-style transformer *from scratch* (random initialization,
no pretrained weights) on masked-token prediction with bidirectional
attention (no causal mask -- every position can attend to every other
position, which is exactly what makes this family unsuitable for open-ended
generation but strong at classification/embedding tasks downstream).

Simplification vs. the original BERT recipe: no Next-Sentence-Prediction
objective and no explicit [CLS]/[SEP]-wrapped sentence pairs -- just packed,
single-stream masked-token prediction. This mirrors RoBERTa's finding that
NSP isn't necessary, not a corner cut for convenience.

Toy-scale by design -- see pretraining/README.md.

Usage:
    python mlm.py --debug_first_batch --max_samples 20
    python mlm.py --max_steps 500 --output_dir ./output/pretraining/mlm
"""

import argparse
import math
import time

from datasets import load_dataset
from transformers import BertConfig, DataCollatorForLanguageModeling, Trainer, TrainingArguments

from common.data_selection import select_samples
from common.logging_utils import print_banner, print_config, print_formatted_examples
from common.model_loading import build_model_from_scratch, load_tokenizer
from common.model_saving import save_model
from common.run_results import write_run_result
from common.seeding import set_all_seeds

DATASET_NAME = "roneneldan/TinyStories"
TOKENIZER_NAME = "bert-base-uncased"
ARCHITECTURE = "encoder-only"


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for this script.

    Returns:
        argparse.ArgumentParser: Parser covering architecture sizing, data,
        training hyperparameters, system, and debug options.
    """
    p = argparse.ArgumentParser(description="Pretrain a small encoder-only model from scratch (MLM objective).")

    p.add_argument("--hidden_size", type=int, default=512, help="Transformer hidden size. Default: 512.")
    p.add_argument("--num_layers", type=int, default=8, help="Number of transformer layers. Default: 8.")
    p.add_argument("--num_attention_heads", type=int, default=8, help="Number of attention heads. Default: 8.")
    p.add_argument("--block_size", type=int, default=256, help="Sequence length in tokens per packed training example. Default: 256.")
    p.add_argument("--mlm_probability", type=float, default=0.15, help="Fraction of tokens masked per example. Default: 0.15 (the original BERT value).")

    p.add_argument("--max_samples", type=int, default=-1, help="Number of raw dataset rows (stories) to use before tokenization/packing. -1 (default) = use the full split.")
    p.add_argument("--sample_selection", type=str, default="random", choices=["random", "first", "last"], help="Which rows to pick when --max_samples is positive. Default: random.")
    p.add_argument("--max_eval_samples", type=int, default=2000, help="Number of validation rows to use for evaluation. Default: 2000.")

    p.add_argument("--batch_size", type=int, default=16, help="Per-device training batch size. Default: 16.")
    p.add_argument("--gradient_accumulation_steps", type=int, default=2, help="Gradient accumulation steps. Default: 2.")
    p.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate. Default: 3e-4.")
    p.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay. Default: 0.01.")
    p.add_argument("--warmup_steps", type=int, default=200, help="LR warmup steps. Default: 200.")
    p.add_argument("--epochs", type=float, default=1.0, help="Training epochs over the (possibly truncated) train split. Ignored if --max_steps is set. Default: 1.0.")
    p.add_argument("--max_steps", type=int, default=-1, help="If set (>0), overrides --epochs. Default: -1 (unset).")

    p.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"], help="Mixed precision mode. Default: bf16.")
    p.add_argument("--gradient_checkpointing", action="store_true", default=False, help="Enable gradient checkpointing. Default: off.")
    p.add_argument("--seed", type=int, default=42, help="Random seed. Default: 42.")
    p.add_argument("--output_dir", type=str, default="./output/pretraining/mlm", help="Where to save the trained model, tokenizer, and run_result.json.")
    p.add_argument("--logging_steps", type=int, default=25, help="Logging frequency. Default: 25.")
    p.add_argument("--eval_steps", type=int, default=250, help="Evaluation frequency. Default: 250.")
    p.add_argument("--save_steps", type=int, default=500, help="Checkpoint save frequency. Default: 500.")

    p.add_argument("--debug_first_batch", action="store_true", default=False, help="Load data, build the model, print formatted examples and the model's real parameter count, then exit without training.")

    return p


def verify_dataset() -> None:
    """Peek at the training split via streaming and assert a "text" field exists.

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
    """Tokenize raw story text and pack it into fixed-length MLM blocks.

    Masking itself is not applied here -- it happens per-batch at train time
    via DataCollatorForLanguageModeling(mlm=True).

    Args:
        dataset (datasets.Dataset): Raw dataset with a "text" column.
        tokenizer: Tokenizer to encode text with.
        block_size (int): Number of tokens per packed training block.
        desc (str): Short label used in progress-bar descriptions (e.g.
            "train" or "eval").

    Returns:
        datasets.Dataset: Dataset of fixed-length "input_ids" blocks, with
        any trailing tokens shorter than block_size dropped.
    """
    def tokenize_fn(examples):
        """Tokenize a batch of stories without adding special tokens."""
        return tokenizer(examples["text"], add_special_tokens=False)

    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=dataset.column_names, desc=f"Tokenizing ({desc})")

    def group_texts(examples):
        """Concatenate a tokenized batch and split it into block_size chunks."""
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = (len(concatenated["input_ids"]) // block_size) * block_size
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated.items()
        }
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
    """Construct a randomly initialized BERT-style model sized by CLI args.

    Args:
        args (argparse.Namespace): Parsed CLI args; uses hidden_size,
            num_layers, num_attention_heads, block_size, and
            gradient_checkpointing.
        tokenizer: Tokenizer used to size the vocabulary and set the pad
            token id.

    Returns:
        transformers.PreTrainedModel: Freshly initialized encoder-only model.
    """
    config = BertConfig(
        vocab_size=len(tokenizer),
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_layers,
        num_attention_heads=args.num_attention_heads,
        intermediate_size=4 * args.hidden_size,
        max_position_embeddings=args.block_size,
        pad_token_id=tokenizer.pad_token_id,
    )
    model = build_model_from_scratch(ARCHITECTURE, config, gradient_checkpointing=args.gradient_checkpointing)

    total_params = sum(p.numel() for p in model.parameters())
    emb = model.bert.embeddings
    embedding_params = (
        emb.word_embeddings.weight.numel()
        + emb.position_embeddings.weight.numel()
        + emb.token_type_embeddings.weight.numel()
    )
    print_banner("MODEL (FROM SCRATCH, RANDOM INIT)")
    print(f"Architecture: {ARCHITECTURE} (BERT-style), vocab_size={config.vocab_size}, "
          f"hidden_size={config.hidden_size}, num_layers={config.num_hidden_layers}, num_heads={config.num_attention_heads}")
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
        str: Human-readable summary of the block's token count and
        (pre-masking) text.
    """
    text = tokenizer.decode(example["input_ids"])
    return f"Packed block ({len(example['input_ids'])} tokens, unmasked -- masking is applied per-batch at train time):\n{text}"


def main():
    """Parse CLI args, run from-scratch MLM pretraining, and write results.

    Loads the tokenizer and data, builds a randomly initialized model,
    optionally exits early for --debug_first_batch, otherwise trains with
    Trainer using a masking data collator, evaluates pseudo-perplexity,
    saves the model, and writes a run_result.json.
    """
    args = build_arg_parser().parse_args()
    set_all_seeds(args.seed)
    print_config(args, "Masked Language Modelling pretraining (encoder-only, from scratch)")

    # verify_dataset()  # commented out: avoids loading the dataset twice (a streaming peek here, then a full load in load_and_prepare_data()) to cut memory overhead
    tokenizer = load_tokenizer(TOKENIZER_NAME)

    train_dataset, eval_dataset = load_and_prepare_data(args, tokenizer)
    model = build_model(args, tokenizer)

    if args.debug_first_batch:
        print_formatted_examples(train_dataset, tokenizer, num_examples=2, decode_fn=decode_example)
        print("--debug_first_batch set: exiting without training.")
        return

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=args.mlm_probability)

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
    # Conventional approximation (loss is only over masked positions, not every
    # token, so this isn't a true sequence perplexity -- but it's the standard
    # proxy reported in BERT-style pretraining logs).
    perplexity = math.exp(eval_loss) if eval_loss < 20 else float("inf")
    print(f"Eval (masked-token) loss: {eval_loss:.4f}  |  Pseudo-perplexity: {perplexity:.2f}")

    save_model(model, tokenizer, args.output_dir, strategy="full")

    total_params = sum(p.numel() for p in model.parameters())
    write_run_result(
        output_dir=args.output_dir,
        stage="pretraining",
        task="mlm",
        modality="text",
        architecture=ARCHITECTURE,
        model_name=f"from-scratch-bert-style-{total_params}params",
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
