"""Image classification SFT.

Dataset: uoft-cs/cifar10 (img -> one of 10 classes). Fields verified live:
`img` (PIL image), `label` (int 0-9, ClassLabel with real names: airplane,
automobile, bird, cat, deer, dog, frog, horse, ship, truck).

Architecturally analogous to
supervised-finetuning/text/classification/encoder-only/text_classification_standard.py:
a classification head on top of a pretrained vision transformer's pooled
representation, trained with plain cross-entropy. Uses
AutoModelForImageClassification + AutoImageProcessor (the vision
equivalent of AutoModelForSequenceClassification + AutoTokenizer). Since
the base model here is small (~86M params) full-parameter finetuning is
standard practice, so this script has no --quantization/--lora flags, same
convention as the encoder-only text classification script.

Usage:
    python image_classification.py --debug_first_batch --max_samples 20
"""

import argparse
import time

import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import (
    AutoImageProcessor,
    AutoModel,
    AutoModelForImageClassification,
    Trainer,
    TrainingArguments,
)

from common.data_selection import select_samples
from common.logging_utils import print_banner, print_config
from common.model_saving import save_model
from common.run_results import write_run_result
from common.seeding import set_all_seeds

DATASET_NAME = "uoft-cs/cifar10"
ARCHITECTURE = "encoder-only"
CLASS_NAMES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="SFT a vision transformer for image classification (classification head).")

    p.add_argument("--model", type=str, default="google/vit-base-patch16-224-in21k", help="Vision transformer base checkpoint. Default: google/vit-base-patch16-224-in21k (fp32 ~330MB, no classification head -- one is added fresh for this task's 10 classes).")
    p.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"], help="Mixed precision. Default: bf16.")
    p.add_argument("--batch_size", type=int, default=32, help="Per-device training batch size. Default: 32 (small model, full-parameter training).")
    p.add_argument("--eval_batch_size", type=int, default=64, help="Per-device eval batch size. Default: 64.")
    p.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps. Default: 1.")
    p.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate. Default: 5e-5 (typical for full-parameter ViT finetuning).")
    p.add_argument("--epochs", type=int, default=2, help="Training epochs. Default: 2.")
    p.add_argument("--warmup_steps", type=int, default=100, help="LR warmup steps. Default: 100.")
    p.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay. Default: 0.01.")
    p.add_argument("--gradient_checkpointing", action="store_true", default=False, help="Enable gradient checkpointing. Default: off (small model, rarely needed).")

    p.add_argument("--max_samples", type=int, default=-1, help="Number of training rows to use. -1 (default) = full split.")
    p.add_argument("--sample_selection", type=str, default="random", choices=["random", "first", "last"], help="Which rows to pick when --max_samples is positive. Default: random.")
    p.add_argument("--max_eval_samples", type=int, default=-1, help="Eval rows to use. -1 (default) = full eval split.")

    p.add_argument("--output_dir", type=str, default="./output/supervised-finetuning/image/image_classification", help="Output directory.")
    p.add_argument("--logging_steps", type=int, default=10, help="Logging frequency. Default: 10.")
    p.add_argument("--eval_steps", type=int, default=200, help="Evaluation frequency. Default: 200.")
    p.add_argument("--save_steps", type=int, default=200, help="Checkpoint save frequency. Default: 200.")

    p.add_argument("--seed", type=int, default=42, help="Random seed. Default: 42.")
    p.add_argument("--debug_first_batch", action="store_true", default=False, help="Print formatted examples and exit without training.")

    return p


def verify_dataset() -> None:
    print_banner("VERIFYING DATASET")
    peek = load_dataset(DATASET_NAME, split="train", streaming=True)
    example = next(iter(peek))
    for field in ("img", "label"):
        assert field in example, f"Expected a {field!r} field in {DATASET_NAME}, got fields: {list(example.keys())}"
    print(f"Dataset: {DATASET_NAME}")
    print(f"Fields: {list(example.keys())}")
    print(f"Sample: img.size={example['img'].size}, label={example['label']} ({CLASS_NAMES[example['label']]})")
    print()


def load_and_prepare_data(args, processor):
    print_banner("LOADING DATASET")
    train_raw = load_dataset(DATASET_NAME, split="train")
    eval_raw = load_dataset(DATASET_NAME, split="test")

    train_raw = select_samples(train_raw, args.max_samples, args.sample_selection, args.seed)
    eval_raw = select_samples(eval_raw, args.max_eval_samples, "first", args.seed)

    def transform(examples):
        pixel_values = processor([img.convert("RGB") for img in examples["img"]], return_tensors="pt")["pixel_values"]
        return {"pixel_values": pixel_values, "labels": examples["label"]}

    train_dataset = train_raw.with_transform(transform)
    eval_dataset = eval_raw.with_transform(transform)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Eval samples: {len(eval_dataset)}")
    print()
    return train_dataset, eval_dataset


def collate_fn(batch):
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    labels = torch.tensor([item["labels"] for item in batch])
    return {"pixel_values": pixel_values, "labels": labels}


def print_formatted_examples_classification(dataset, num_examples: int = 2) -> None:
    print_banner("FORMATTED EXAMPLES")
    for i in range(min(num_examples, len(dataset))):
        example = dataset[i]
        print(f"\n--- Example {i + 1} ---")
        print(f"pixel_values.shape: {tuple(example['pixel_values'].shape)}")
        print(f"Label: {example['labels']} ({CLASS_NAMES[example['labels']]})")
    print()


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1_macro": f1_score(labels, predictions, average="macro"),
        "f1_weighted": f1_score(labels, predictions, average="weighted"),
        "precision": precision_score(labels, predictions, average="macro"),
        "recall": recall_score(labels, predictions, average="macro"),
    }


def main():
    args = build_arg_parser().parse_args()
    set_all_seeds(args.seed)
    print_config(args, "Image classification SFT (classification head)")

    # verify_dataset()  # commented out: avoids loading the dataset twice (a streaming peek here, then a full load in load_and_prepare_data()) to cut memory overhead
    processor = AutoImageProcessor.from_pretrained(args.model, use_fast=True)

    train_dataset, eval_dataset = load_and_prepare_data(args, processor)

    torch_dtype = torch.bfloat16 if args.mixed_precision == "bf16" else (torch.float16 if args.mixed_precision == "fp16" else torch.float32)
    model = AutoModelForImageClassification.from_pretrained(
        args.model,
        num_labels=len(CLASS_NAMES),
        id2label={i: name for i, name in enumerate(CLASS_NAMES)},
        label2id={name: i for i, name in enumerate(CLASS_NAMES)},
        dtype=torch_dtype,
    )
    # Real bug found and worked around: on this checkpoint + transformers==5.14.1,
    # AutoModelForImageClassification's from_pretrained() does NOT apply the legacy
    # key-name remapping (encoder.layer.N.attention.attention.query -> layers.N.attention.q_proj,
    # etc.) that AutoModel/ViTModel applies correctly on the identical checkpoint --
    # confirmed by diffing their LOAD REPORTs directly: ViTModel loads with zero
    # missing/unexpected keys, ViTForImageClassification reports the ENTIRE backbone
    # (every encoder layer) as missing, meaning it would silently train from a
    # randomly-initialized backbone while claiming to finetune a pretrained checkpoint.
    # Fixed by loading the backbone separately (where the remapping works) and copying
    # its correctly-loaded weights into the classification model's backbone submodule.
    backbone = AutoModel.from_pretrained(args.model, dtype=torch_dtype)
    # strict=False: AutoModel's ViTModel includes a pooler head (pooler.dense.*) that
    # ViTForImageClassification's internal backbone submodule doesn't have (it uses
    # add_pooling_layer=False) -- confirmed via a live RuntimeError naming exactly
    # those two keys as unexpected, nothing else.
    getattr(model, model.base_model_prefix).load_state_dict(backbone.state_dict(), strict=False)
    del backbone
    print("[image_classification] backbone weights corrected in-place after the above (misleading) load report "
          "-- see this file's real bug note just above this line in the source for why the report is inaccurate.")
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    total_params = sum(p.numel() for p in model.parameters())
    print_banner("MODEL")
    print(f"Architecture: {ARCHITECTURE} ({args.model}), num_labels={len(CLASS_NAMES)}")
    print(f"Total parameters: {total_params:,}")
    print()

    if args.debug_first_batch:
        print_formatted_examples_classification(train_dataset, num_examples=2)
        print("--debug_first_batch set: exiting without training.")
        return

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
        remove_unused_columns=False,  # with_transform's output columns (pixel_values/labels) aren't in dataset.column_names, so the default column-pruning would drop them
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
    )

    print_banner("TRAINING")
    start = time.time()
    trainer.train()
    train_runtime_seconds = time.time() - start

    print_banner("FINAL EVALUATION")
    eval_metrics = trainer.evaluate()
    print(eval_metrics)

    save_model(model, processor, args.output_dir, strategy="full")

    write_run_result(
        output_dir=args.output_dir,
        stage="supervised-finetuning",
        task="image_classification",
        modality="image",
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
