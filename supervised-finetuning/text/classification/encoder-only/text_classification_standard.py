"""Text classification SFT -- encoder-only.

Dataset: Tohrumi/glue_sst2_10k (sentence -> sentiment label), same corpus as
the decoder-only/encoder-decoder classification scripts, so results are
directly comparable across architecture families (see
experiments/architecture-family-classification/).

Architecturally different from the decoder-only scripts: no prompt template,
no prompt/completion loss masking, no generation-based evaluation. A
classification head sits on top of the encoder's pooled [CLS]-equivalent
representation and is trained with plain cross-entropy over the 2 classes --
a classification head can't emit a reasoning trace, so there's no CoT
variant of this script. Uses AutoModelForSequenceClassification. Since the
base model here is small (~150M params) full-parameter finetuning is
standard practice, so this script has no --quantization/--lora flags,
unlike the decoder-only scripts.

Usage:
    python text_classification_standard.py --debug_first_batch --max_samples 20
"""

import argparse
import time

import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from common.data_selection import select_samples
from common.logging_utils import print_banner, print_config
from common.model_saving import save_model
from common.run_results import write_run_result
from common.seeding import set_all_seeds

DATASET_NAME = "Tohrumi/glue_sst2_10k"
ARCHITECTURE = "encoder-only"


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for the encoder-only classification run.

    Returns:
        argparse.ArgumentParser: Parser covering model/precision, optimization,
            data-selection, output/checkpointing, and seed/debug flags for
            this script.
    """
    p = argparse.ArgumentParser(description="SFT an encoder-only model for text classification (classification head).")

    p.add_argument("--model", type=str, default="answerdotai/ModernBERT-base", help="Encoder-only base checkpoint. Default: answerdotai/ModernBERT-base (fp16 ~1.2GB).")
    p.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"], help="Mixed precision. Default: bf16.")
    p.add_argument("--batch_size", type=int, default=16, help="Per-device training batch size. Default: 16 (small model, full-parameter training -- much larger batches fit than the decoder-only scripts).")
    p.add_argument("--eval_batch_size", type=int, default=32, help="Per-device eval batch size. Default: 32.")
    p.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps. Default: 1.")
    p.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate. Default: 2e-5 (typical for full-parameter encoder finetuning, much lower than the decoder-only scripts' LoRA learning rate).")
    p.add_argument("--epochs", type=int, default=3, help="Training epochs. Default: 3.")
    p.add_argument("--max_length", type=int, default=256, help="Max sequence length. Default: 256.")
    p.add_argument("--warmup_steps", type=int, default=100, help="LR warmup steps. Default: 100.")
    p.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay. Default: 0.01.")
    p.add_argument("--gradient_checkpointing", action="store_true", default=False, help="Enable gradient checkpointing. Default: off (small model, rarely needed).")

    p.add_argument("--max_samples", type=int, default=-1, help="Number of training rows to use. -1 (default) = full split.")
    p.add_argument("--sample_selection", type=str, default="random", choices=["random", "first", "last"], help="Which rows to pick when --max_samples is positive. Default: random.")
    p.add_argument("--max_eval_samples", type=int, default=-1, help="Eval rows to use. -1 (default) = full eval split (cheap -- no generation involved).")

    p.add_argument("--output_dir", type=str, default="./output/supervised-finetuning/text/classification/encoder-only/standard", help="Output directory.")
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
    """Load the train/eval splits and tokenize them for a classification head.

    Tokenizes the raw sentence and renames the `label` column to `labels`
    (the name `Trainer`/`AutoModelForSequenceClassification` expect).

    Args:
        args (argparse.Namespace): Parsed CLI args; uses `max_samples`,
            `sample_selection`, `max_eval_samples`, `seed`, and `max_length`.
        tokenizer: Tokenizer used to encode the sentences.

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
        """Tokenize a batch of raw sentences.

        Args:
            examples (dict): Batched raw rows with a `sentence` column (as
                produced by `Dataset.map(batched=True)`).

        Returns:
            dict: Tokenized `input_ids`/`attention_mask` for the sentences.
        """
        return tokenizer(examples["sentence"], truncation=True, max_length=args.max_length)

    train_dataset = train_raw.map(tokenize_fn, batched=True, desc="Tokenizing (train)")
    eval_dataset = eval_raw.map(tokenize_fn, batched=True, desc="Tokenizing (eval)")
    train_dataset = train_dataset.rename_column("label", "labels")
    eval_dataset = eval_dataset.rename_column("label", "labels")

    print(f"Train samples: {len(train_dataset)}")
    print(f"Eval samples: {len(eval_dataset)}")
    print()
    return train_dataset, eval_dataset


def print_formatted_examples_classification(dataset, tokenizer, num_examples=2):
    """Print a few tokenized input/label pairs for debugging.

    Args:
        dataset: Tokenized HF `Dataset` with `input_ids` and `labels` columns.
        tokenizer: Tokenizer used to decode the token ids back to text.
        num_examples (int): Number of examples to print. Default: 2.
    """
    print_banner("FORMATTED EXAMPLES")
    for i in range(min(num_examples, len(dataset))):
        example = dataset[i]
        print(f"\n--- Example {i + 1} ---")
        print(f"Input: {tokenizer.decode(example['input_ids'], skip_special_tokens=False)}")
        print(f"Label: {example['labels']} ({'positive' if example['labels'] == 1 else 'negative'})")
    print()


def compute_metrics(eval_pred):
    """Compute classification metrics from a `Trainer` eval prediction.

    Args:
        eval_pred: `(logits, labels)` tuple from `Trainer`, where `logits`
            has shape `(num_examples, 2)`.

    Returns:
        dict: Accuracy, macro/weighted F1, precision, recall, and AUROC
            (`nan` if undefined, e.g. only one class present in `labels`).
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = accuracy_score(labels, predictions)
    f1_macro = f1_score(labels, predictions, average="macro")
    f1_weighted = f1_score(labels, predictions, average="weighted")
    precision = precision_score(labels, predictions, average="macro")
    recall = recall_score(labels, predictions, average="macro")
    try:
        probs_positive = torch.softmax(torch.tensor(logits), dim=-1)[:, 1].numpy()
        auroc = roc_auc_score(labels, probs_positive)
    except ValueError:
        auroc = None
    return {
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "precision": precision,
        "recall": recall,
        "auroc": auroc if auroc is not None else float("nan"),
    }


def main():
    """Run the full encoder-only classification-head SFT pipeline end to end.

    Parses CLI args, loads the tokenizer/dataset/model, either prints a debug
    batch and exits or trains with `Trainer`, evaluates, saves the model, and
    writes a `run_result.json`.
    """
    args = build_arg_parser().parse_args()
    set_all_seeds(args.seed)
    print_config(args, "Text classification SFT -- encoder-only (classification head)")

    # verify_dataset()  # commented out: avoids loading the dataset twice (a streaming peek here, then a full load in load_and_prepare_data()) to cut memory overhead
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    train_dataset, eval_dataset = load_and_prepare_data(args, tokenizer)

    torch_dtype = torch.bfloat16 if args.mixed_precision == "bf16" else (torch.float16 if args.mixed_precision == "fp16" else torch.float32)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2, dtype=torch_dtype)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    total_params = sum(p.numel() for p in model.parameters())
    print_banner("MODEL")
    print(f"Architecture: {ARCHITECTURE} ({args.model}), num_labels=2")
    print(f"Total parameters: {total_params:,}")
    print()

    if args.debug_first_batch:
        print_formatted_examples_classification(train_dataset, tokenizer, num_examples=2)
        print("--debug_first_batch set: exiting without training.")
        return

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    training_args = TrainingArguments(
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
        seed=args.seed,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
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
            "auroc": eval_metrics.get("eval_auroc"),
            "total_parameters": total_params,
        },
        num_train_samples=len(train_dataset),
        num_eval_samples=len(eval_dataset),
        train_runtime_seconds=train_runtime_seconds,
    )
    print("Done.")


if __name__ == "__main__":
    main()
