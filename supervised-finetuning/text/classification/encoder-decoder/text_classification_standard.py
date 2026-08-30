"""Text classification SFT -- encoder-decoder, text-to-text formulation.

Dataset: Tohrumi/glue_sst2_10k (sentence -> sentiment label), same corpus as
the decoder-only/encoder-only classification scripts, so results are
directly comparable across architecture families (see
experiments/architecture-family-classification/).

Uses raw pretrained `t5-base`, NOT `google/flan-t5-base` -- FLAN-T5 is
already instruction-tuned, which would violate this project's
prefer-base-checkpoints rule (see root README's Model Selection Philosophy).
Formulated as text-to-text: `"classify sentiment: {sentence}" -> "{label}"`,
trained with the standard seq2seq (span/sequence generation) loss via
Seq2SeqTrainer + DataCollatorForSeq2Seq. This establishes the pattern for a
future CoT-capable encoder-decoder script -- T5-family models can generate a
reasoning span before the label, unlike the encoder-only classifier.

Usage:
    python text_classification_standard.py --debug_first_batch --max_samples 20
"""

import argparse
import time

import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from common.data_selection import select_samples
from common.logging_utils import print_banner, print_config
from common.model_saving import save_model
from common.run_results import write_run_result
from common.seeding import set_all_seeds

DATASET_NAME = "Tohrumi/glue_sst2_10k"
ARCHITECTURE = "encoder-decoder"
LABEL_MAP = {0: "negative", 1: "positive"}
PREFIX = "classify sentiment: "


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for the encoder-decoder classification run.

    Returns:
        argparse.ArgumentParser: Parser covering model/precision, optimization,
            data-selection, output/checkpointing, and seed/debug flags for
            this script.
    """
    p = argparse.ArgumentParser(description="SFT an encoder-decoder model for text classification (text-to-text).")

    p.add_argument("--model", type=str, default="t5-base", help="Encoder-decoder base checkpoint (raw, non-instruction-tuned). Default: t5-base (fp16 ~0.9GB).")
    p.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"], help="Mixed precision. Default: bf16.")
    p.add_argument("--batch_size", type=int, default=16, help="Per-device training batch size. Default: 16.")
    p.add_argument("--eval_batch_size", type=int, default=16, help="Per-device eval batch size. Default: 16.")
    p.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps. Default: 1.")
    p.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate. Default: 3e-4 (typical for full-parameter T5 finetuning).")
    p.add_argument("--epochs", type=int, default=3, help="Training epochs. Default: 3.")
    p.add_argument("--max_length", type=int, default=256, help="Max input sequence length. Default: 256.")
    p.add_argument("--max_target_length", type=int, default=8, help="Max target sequence length (the label is one word). Default: 8.")
    p.add_argument("--warmup_steps", type=int, default=100, help="LR warmup steps. Default: 100.")
    p.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay. Default: 0.01.")
    p.add_argument("--gradient_checkpointing", action="store_true", default=False, help="Enable gradient checkpointing. Default: off.")

    p.add_argument("--max_samples", type=int, default=-1, help="Number of training rows to use. -1 (default) = full split.")
    p.add_argument("--sample_selection", type=str, default="random", choices=["random", "first", "last"], help="Which rows to pick when --max_samples is positive. Default: random.")
    p.add_argument("--max_eval_samples", type=int, default=200, help="Eval rows to use (generation-based eval). Default: 200.")

    p.add_argument("--output_dir", type=str, default="./output/supervised-finetuning/text/classification/encoder-decoder/standard", help="Output directory.")
    p.add_argument("--logging_steps", type=int, default=10, help="Logging frequency. Default: 10.")
    p.add_argument("--eval_steps", type=int, default=200, help="Evaluation frequency. Default: 200.")
    p.add_argument("--save_steps", type=int, default=200, help="Checkpoint save frequency. Default: 200.")

    p.add_argument("--seed", type=int, default=42, help="Random seed. Default: 42.")
    p.add_argument("--debug_first_batch", action="store_true", default=False, help="Print formatted examples and exit without training.")

    return p


def verify_dataset() -> None:
    """Peek the dataset (streaming) and assert the expected fields exist.

    Raises:
        AssertionError: If `sentence` or `label` is missing from the first
            streamed example.
    """
    print_banner("VERIFYING DATASET")
    peek = load_dataset(DATASET_NAME, split="train", streaming=True)
    example = next(iter(peek))
    for field in ("sentence", "label"):
        assert field in example, f"Expected a {field!r} field in {DATASET_NAME}, got fields: {list(example.keys())}"
    print(f"Dataset: {DATASET_NAME}")
    print(f"Fields: {list(example.keys())}")
    print(f"Sample: sentence={example['sentence']!r}, label={example['label']!r}")
    print()


def load_and_prepare_data(args, tokenizer):
    """Load the train/eval splits and tokenize them as text-to-text pairs.

    Prefixes each sentence with the `"classify sentiment: "` task prefix and
    tokenizes the label word as the decoder target.

    Args:
        args (argparse.Namespace): Parsed CLI args; uses `max_samples`,
            `sample_selection`, `max_eval_samples`, `seed`, `max_length`, and
            `max_target_length`.
        tokenizer: Tokenizer used to encode inputs and targets.

    Returns:
        tuple: `(train_dataset, eval_dataset)`, HF `Dataset` objects with
            `input_ids`, `attention_mask`, and `labels` columns.
    """
    print_banner("LOADING DATASET")
    train_raw = load_dataset(DATASET_NAME, split="train")
    eval_raw = load_dataset(DATASET_NAME, split="eval")

    train_raw = select_samples(train_raw, args.max_samples, args.sample_selection, args.seed)
    eval_raw = select_samples(eval_raw, args.max_eval_samples, "first", args.seed)

    def tokenize_fn(examples):
        """Tokenize a batch of raw examples into text-to-text model inputs.

        Args:
            examples (dict): Batched raw rows with `sentence` and `label`
                columns (as produced by `Dataset.map(batched=True)`).

        Returns:
            dict: Tokenized `input_ids`/`attention_mask` for the prefixed
                sentence plus tokenized `labels` for the target label word.
        """
        inputs = [PREFIX + s for s in examples["sentence"]]
        model_inputs = tokenizer(inputs, truncation=True, max_length=args.max_length)
        targets = [LABEL_MAP[l] for l in examples["label"]]
        labels = tokenizer(text_target=targets, truncation=True, max_length=args.max_target_length)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    train_dataset = train_raw.map(tokenize_fn, batched=True, remove_columns=train_raw.column_names, desc="Tokenizing (train)")
    eval_dataset = eval_raw.map(tokenize_fn, batched=True, remove_columns=eval_raw.column_names, desc="Tokenizing (eval)")

    print(f"Train samples: {len(train_dataset)}")
    print(f"Eval samples: {len(eval_dataset)}")
    print()
    return train_dataset, eval_dataset


def print_formatted_examples_t5(dataset, tokenizer, num_examples=2):
    """Print a few tokenized encoder-input/decoder-target pairs for debugging.

    Args:
        dataset: Tokenized HF `Dataset` with `input_ids` and `labels` columns.
        tokenizer: Tokenizer used to decode the token ids back to text.
        num_examples (int): Number of examples to print. Default: 2.
    """
    print_banner("FORMATTED EXAMPLES")
    for i in range(min(num_examples, len(dataset))):
        example = dataset[i]
        print(f"\n--- Example {i + 1} ---")
        print(f"Encoder input: {tokenizer.decode(example['input_ids'], skip_special_tokens=False)}")
        print(f"Decoder target: {tokenizer.decode(example['labels'], skip_special_tokens=False)}")
    print()


def build_compute_metrics(tokenizer):
    """Build a `Seq2SeqTrainer`-compatible `compute_metrics` callback.

    Args:
        tokenizer: Tokenizer used to decode generated/target token ids back
            to label text.

    Returns:
        Callable: A `compute_metrics(eval_pred)` function suitable for
            `Seq2SeqTrainer`.
    """
    def compute_metrics(eval_pred):
        """Decode generated and target sequences and score them as labels.

        Args:
            eval_pred: `(predictions, labels)` tuple from `Seq2SeqTrainer`
                (token ids, with -100 marking padding in `labels`).

        Returns:
            dict: Accuracy, macro/weighted F1, precision, and recall,
                computed after mapping decoded text back to label ids
                (defaulting unrecognized text to the "positive" class).
        """
        predictions, labels = eval_pred
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_preds = [t.strip().lower() for t in tokenizer.batch_decode(predictions, skip_special_tokens=True)]
        decoded_labels = [t.strip().lower() for t in tokenizer.batch_decode(labels, skip_special_tokens=True)]

        reverse_map = {"negative": 0, "positive": 1}
        pred_ids = [reverse_map.get(p, 1) for p in decoded_preds]
        label_ids = [reverse_map.get(l, 1) for l in decoded_labels]

        return {
            "accuracy": accuracy_score(label_ids, pred_ids),
            "f1_macro": f1_score(label_ids, pred_ids, average="macro"),
            "f1_weighted": f1_score(label_ids, pred_ids, average="weighted"),
            "precision": precision_score(label_ids, pred_ids, average="macro"),
            "recall": recall_score(label_ids, pred_ids, average="macro"),
        }

    return compute_metrics


def main():
    """Run the full encoder-decoder text-to-text classification SFT pipeline.

    Parses CLI args, loads the tokenizer/dataset/model, either prints a debug
    batch and exits or trains with `Seq2SeqTrainer`, evaluates, saves the
    model, and writes a `run_result.json`.
    """
    args = build_arg_parser().parse_args()
    set_all_seeds(args.seed)
    print_config(args, "Text classification SFT -- encoder-decoder (text-to-text)")

    # verify_dataset()  # commented out: avoids loading the dataset twice (a streaming peek here, then a full load in load_and_prepare_data()) to cut memory overhead
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    train_dataset, eval_dataset = load_and_prepare_data(args, tokenizer)

    torch_dtype = torch.bfloat16 if args.mixed_precision == "bf16" else (torch.float16 if args.mixed_precision == "fp16" else torch.float32)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model, dtype=torch_dtype)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    total_params = sum(p.numel() for p in model.parameters())
    print_banner("MODEL")
    print(f"Architecture: {ARCHITECTURE} ({args.model})")
    print(f"Total parameters: {total_params:,}")
    print()

    if args.debug_first_batch:
        print_formatted_examples_t5(train_dataset, tokenizer, num_examples=2)
        print("--debug_first_batch set: exiting without training.")
        return

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        bf16=(args.mixed_precision == "bf16"),
        fp16=(args.mixed_precision == "fp16"),
        gradient_checkpointing=args.gradient_checkpointing,
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        predict_with_generate=True,
        generation_max_length=args.max_target_length,
        seed=args.seed,
        report_to=[],
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=build_compute_metrics(tokenizer),
    )

    print_banner("TRAINING")
    start = time.time()
    trainer.train()
    train_runtime_seconds = time.time() - start

    print_banner("FINAL EVALUATION")
    eval_metrics = trainer.evaluate()
    print(eval_metrics)

    save_model(model, tokenizer, args.output_dir, strategy="full")

    write_run_result(
        output_dir=args.output_dir,
        stage="supervised-finetuning",
        task="text_classification",
        modality="text",
        architecture=ARCHITECTURE,
        variant="standard",
        cot_enabled=False,
        model_name=args.model,
        dataset_name=DATASET_NAME,
        save_strategy="full",
        hyperparameters=vars(args),
        metrics={
            "accuracy": eval_metrics.get("eval_accuracy"),
            "f1_macro": eval_metrics.get("eval_f1_macro"),
            "f1_weighted": eval_metrics.get("eval_f1_weighted"),
            "precision": eval_metrics.get("eval_precision"),
            "recall": eval_metrics.get("eval_recall"),
            "total_parameters": total_params,
        },
        num_train_samples=len(train_dataset),
        num_eval_samples=len(eval_dataset),
        train_runtime_seconds=train_runtime_seconds,
    )
    print("Done.")


if __name__ == "__main__":
    main()
