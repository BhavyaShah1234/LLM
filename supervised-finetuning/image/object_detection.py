"""Object detection SFT.

Dataset: rishitdagli/cppe-5 -- protective-equipment detection (5 classes:
Coverall, Face_Shield, Gloves, Goggles, Mask). Fields verified live:
`image_id`, `image` (PIL), `width`, `height`, `objects` (COCO-style dict:
`id`, `area`, `bbox` [x, y, w, h] absolute pixels, `category` int).
The unqualified `cppe-5` repo id (the one used in HF's own object-detection
tutorial) is NOT loadable under this project's pinned `datasets==5.0.1` --
confirmed via a live `HfUriError` ("Repository id must be 'namespace/name'"
-- the same class of legacy-non-namespaced-repo issue documented in
`pretraining/README.md`). `rishitdagli/cppe-5` is a namespaced mirror with
the identical schema, verified live before implementation.

Model: facebook/detr-resnet-50 (DETR) via AutoModelForObjectDetection.
COCO's 91-class head is reinitialized for this task's 5 classes
(`num_labels=5`) -- verified via a direct LOAD REPORT diff (see
`image/README.md`) that this reinit is clean and narrow (only
`class_labels_classifier.{weight,bias}` mismatch-reinitialized), UNLIKE
the real bug found in `image_classification.py` where an equivalent-looking
head resize silently dropped the entire backbone -- DETR's checkpoint
conversion mapping works correctly on this checkpoint, confirmed before
trusting it.

**Deliberate scope decision**: this script reports the model's own
training/eval loss (classification + bbox L1 + generalized IoU, the
standard DETR loss terms, computed internally when `labels` are passed)
rather than COCO mAP. Real mAP evaluation needs `pycocotools` (not in this
project's `requirements.txt`) and a full COCO-format prediction/ground-truth
matching pipeline -- out of scope for a toy-scale finetuning demonstration.
Loss going down and NOT NaN is still a real, meaningful training-correctness
signal, same standard this project's pretraining/ scripts use
(perplexity, not downstream task metrics).

Usage:
    python object_detection.py --debug_first_batch --max_samples 20
"""

import argparse
import time

import torch
from datasets import load_dataset
from transformers import AutoImageProcessor, AutoModelForObjectDetection, Trainer, TrainingArguments

from common.data_selection import select_samples
from common.logging_utils import print_banner, print_config
from common.model_saving import save_model
from common.run_results import write_run_result
from common.seeding import set_all_seeds

DATASET_NAME = "rishitdagli/cppe-5"
ARCHITECTURE = "encoder-decoder"  # DETR: CNN backbone (encoder-like feature extractor) + transformer encoder-decoder detection head
CLASS_NAMES = ["Coverall", "Face_Shield", "Gloves", "Goggles", "Mask"]


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for this training script.

    Returns:
        argparse.ArgumentParser: Parser covering model choice, image resize
        bounds, optimization hyperparameters, sample selection, output
        paths, and debug/seed flags.
    """
    p = argparse.ArgumentParser(description="SFT a DETR object detector.")

    p.add_argument("--model", type=str, default="facebook/detr-resnet-50", help="DETR checkpoint. Default: facebook/detr-resnet-50 (fp32 ~160MB).")
    p.add_argument("--image_shortest_edge", type=int, default=480, help="Resize images so the shortest edge is this many pixels (kept below COCO's default 800 for a smaller/faster toy-scale run). Default: 480.")
    p.add_argument("--image_longest_edge", type=int, default=800, help="Cap the longest edge at this many pixels. Default: 800.")

    p.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"], help="Mixed precision. Default: bf16.")
    p.add_argument("--batch_size", type=int, default=4, help="Per-device training batch size. Default: 4 (DETR's transformer decoder + Hungarian matching is heavier per-image than a plain classifier).")
    p.add_argument("--eval_batch_size", type=int, default=4, help="Per-device eval batch size. Default: 4.")
    p.add_argument("--gradient_accumulation_steps", type=int, default=2, help="Gradient accumulation steps. Default: 2.")
    p.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate. Default: 1e-5 (DETR finetuning typically uses a low LR).")
    p.add_argument("--epochs", type=int, default=5, help="Training epochs. Default: 5 (cppe-5 is small; a handful of epochs is standard for this dataset in HF's own tutorial).")
    p.add_argument("--warmup_steps", type=int, default=50, help="LR warmup steps. Default: 50.")
    p.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay. Default: 1e-4.")

    p.add_argument("--max_samples", type=int, default=-1, help="Number of training rows to use. -1 (default) = full split.")
    p.add_argument("--sample_selection", type=str, default="random", choices=["random", "first", "last"], help="Which rows to pick when --max_samples is positive. Default: random.")
    p.add_argument("--max_eval_samples", type=int, default=-1, help="Eval rows to use. -1 (default) = full eval split.")

    p.add_argument("--output_dir", type=str, default="./output/supervised-finetuning/image/object_detection", help="Output directory.")
    p.add_argument("--logging_steps", type=int, default=10, help="Logging frequency. Default: 10.")
    p.add_argument("--eval_steps", type=int, default=50, help="Evaluation frequency. Default: 50.")
    p.add_argument("--save_steps", type=int, default=50, help="Checkpoint save frequency. Default: 50.")

    p.add_argument("--seed", type=int, default=42, help="Random seed. Default: 42.")
    p.add_argument("--debug_first_batch", action="store_true", default=False, help="Print formatted examples and exit without training.")

    return p


def verify_dataset() -> None:
    """Stream-peek one example from the dataset and assert the expected fields exist.

    Prints the fields and one sample's image size/object count/categories.
    Not called from `main()` (see the comment there) but kept for manual
    verification runs.
    """
    print_banner("VERIFYING DATASET")
    peek = load_dataset(DATASET_NAME, split="train", streaming=True)
    example = next(iter(peek))
    for field in ("image_id", "image", "width", "height", "objects"):
        assert field in example, f"Expected a {field!r} field in {DATASET_NAME}, got fields: {list(example.keys())}"
    print(f"Dataset: {DATASET_NAME}")
    print(f"Fields: {list(example.keys())}")
    print(f"Sample: image.size={example['image'].size}, num_objects={len(example['objects']['id'])}, categories={example['objects']['category']}")
    print()


def to_coco_annotations(example):
    """Convert one dataset example's `objects` field into COCO-style detection annotations.

    Args:
        example (dict): A raw dataset example with `image_id` and an
            `objects` dict of parallel `category`/`bbox`/`area` lists.

    Returns:
        dict: `{"image_id": ..., "annotations": [...]}` in the COCO
        annotation format expected by `AutoImageProcessor`.
    """
    objects = example["objects"]
    annotations = [
        {"image_id": example["image_id"], "category_id": cat, "bbox": bbox, "area": area, "iscrowd": 0}
        for cat, bbox, area in zip(objects["category"], objects["bbox"], objects["area"])
    ]
    return {"image_id": example["image_id"], "annotations": annotations}


def load_and_prepare_data(args, processor):
    """Load the CPPE-5 train/test splits and wire up on-the-fly image/annotation preprocessing.

    Args:
        args (argparse.Namespace): Parsed CLI args; uses `max_samples`,
            `sample_selection`, `max_eval_samples`, and `seed`.
        processor: A Hugging Face `AutoImageProcessor` instance used to
            resize images and encode COCO-style annotations for DETR.

    Returns:
        tuple: `(train_dataset, eval_dataset)`, `datasets.Dataset` objects
        with a lazy `with_transform` preprocessing pipeline attached.
    """
    print_banner("LOADING DATASET")
    train_raw = load_dataset(DATASET_NAME, split="train")
    eval_raw = load_dataset(DATASET_NAME, split="test")

    train_raw = select_samples(train_raw, args.max_samples, args.sample_selection, args.seed)
    eval_raw = select_samples(eval_raw, args.max_eval_samples, "first", args.seed)

    def transform(examples):
        """Batch-transform raw images and COCO annotations into model-ready pixel values and labels.

        Args:
            examples (dict): Batch of raw dataset columns, including
                `image` (list of PIL images) plus the detection-annotation
                columns consumed by `to_coco_annotations`.

        Returns:
            dict: `{"pixel_values", "labels"}` ready for the model.
        """
        images = [img.convert("RGB") for img in examples["image"]]
        targets = [to_coco_annotations(dict(zip(examples.keys(), vals))) for vals in zip(*examples.values())]
        encoding = processor(images=images, annotations=targets, return_tensors="pt")
        return {"pixel_values": list(encoding["pixel_values"]), "labels": encoding["labels"]}

    train_dataset = train_raw.with_transform(transform)
    eval_dataset = eval_raw.with_transform(transform)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Eval samples: {len(eval_dataset)}")
    print()
    return train_dataset, eval_dataset


def collate_fn(batch):
    """Right/bottom zero-pad a batch of variable-sized images to a common size and build a pixel mask.

    Args:
        batch (list[dict]): Examples, each with a `pixel_values` tensor of
            possibly differing (C, H, W) and a `labels` dict.

    Returns:
        dict: `{"pixel_values": Tensor, "pixel_mask": Tensor, "labels": list}`
        where `pixel_mask` marks real (1) vs. padded (0) pixels.
    """
    # Manual batch padding: this transformers version's DetrImageProcessor.pad()
    # signature changed to operate on a SINGLE image (image, padded_size, ...), not a
    # batch with a return_tensors kwarg -- confirmed via a live TypeError, and via
    # inspect.signature() showing the new single-image signature. Older HF tutorials'
    # `processor.pad(pixel_values, return_tensors="pt")` no longer exists. Images in a
    # batch can have different sizes (aspect-ratio-preserving resize), so DETR needs
    # right/bottom zero-padding to a common size plus a pixel_mask marking real (1) vs
    # padded (0) pixels -- reimplemented directly rather than relying on the processor.
    pixel_values_list = [item["pixel_values"] for item in batch]
    labels = [item["labels"] for item in batch]
    max_h = max(pv.shape[-2] for pv in pixel_values_list)
    max_w = max(pv.shape[-1] for pv in pixel_values_list)

    padded_pixel_values, pixel_mask = [], []
    for pv in pixel_values_list:
        c, h, w = pv.shape
        padded = torch.zeros((c, max_h, max_w), dtype=pv.dtype)
        padded[:, :h, :w] = pv
        mask = torch.zeros((max_h, max_w), dtype=torch.long)
        mask[:h, :w] = 1
        padded_pixel_values.append(padded)
        pixel_mask.append(mask)

    return {"pixel_values": torch.stack(padded_pixel_values), "pixel_mask": torch.stack(pixel_mask), "labels": labels}


def print_formatted_examples_detection(dataset, num_examples: int = 2) -> None:
    """Print a few dataset examples' pixel shapes and detected box classes for manual inspection.

    Args:
        dataset: A dataset with `__getitem__` returning transformed examples.
        num_examples (int): Number of examples to print. Default: 2.
    """
    print_banner("FORMATTED EXAMPLES")
    for i in range(min(num_examples, len(dataset))):
        example = dataset[i]
        labels = example["labels"]
        class_labels = labels["class_labels"].tolist()
        print(f"\n--- Example {i + 1} ---")
        print(f"pixel_values.shape: {tuple(example['pixel_values'].shape)}")
        print(f"num_boxes: {len(class_labels)}, classes: {[CLASS_NAMES[c] for c in class_labels]}")
    print()


def main():
    """Run the end-to-end object detection SFT pipeline: load data, fine-tune DETR, evaluate, save."""
    args = build_arg_parser().parse_args()
    set_all_seeds(args.seed)
    print_config(args, "Object detection SFT (DETR)")

    # verify_dataset()  # commented out: avoids loading the dataset twice (a streaming peek here, then a full load in load_and_prepare_data()) to cut memory overhead
    processor = AutoImageProcessor.from_pretrained(
        args.model, size={"shortest_edge": args.image_shortest_edge, "longest_edge": args.image_longest_edge}
    )

    train_dataset, eval_dataset = load_and_prepare_data(args, processor)

    torch_dtype = torch.bfloat16 if args.mixed_precision == "bf16" else (torch.float16 if args.mixed_precision == "fp16" else torch.float32)
    model = AutoModelForObjectDetection.from_pretrained(
        args.model,
        num_labels=len(CLASS_NAMES),
        id2label={i: name for i, name in enumerate(CLASS_NAMES)},
        label2id={name: i for i, name in enumerate(CLASS_NAMES)},
        ignore_mismatched_sizes=True,  # COCO's 91-class head -> this task's 5 classes; verified via a direct LOAD REPORT diff that this is a clean, narrow reinit (only the classifier head), not the broader backbone-loading bug found in image_classification.py
        dtype=torch_dtype,
    )

    total_params = sum(p.numel() for p in model.parameters())
    print_banner("MODEL")
    print(f"Architecture: {ARCHITECTURE} ({args.model}), num_labels={len(CLASS_NAMES)}")
    print(f"Total parameters: {total_params:,}")
    print()

    if args.debug_first_batch:
        print_formatted_examples_detection(train_dataset, num_examples=2)
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
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        seed=args.seed,
        report_to=[],
        remove_unused_columns=False,  # with_transform's output columns aren't in dataset.column_names
        dataloader_pin_memory=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collate_fn,
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
        task="object_detection",
        modality="image",
        architecture=ARCHITECTURE,
        variant="standard",
        cot_enabled=False,
        model_name=args.model,
        dataset_name=DATASET_NAME,
        save_strategy="full",
        hyperparameters=vars(args),
        metrics={"eval_loss": eval_metrics.get("eval_loss"), "total_parameters": total_params},
        num_train_samples=len(train_dataset),
        num_eval_samples=len(eval_dataset),
        train_runtime_seconds=train_runtime_seconds,
    )
    print("Done.")


if __name__ == "__main__":
    main()
