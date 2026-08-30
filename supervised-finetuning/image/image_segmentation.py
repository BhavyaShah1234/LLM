"""Semantic segmentation SFT.

Dataset: mattmdjaga/human_parsing_dataset -- 17,706 (image, mask) pairs,
18-class human body-part/clothing segmentation (background, hat, hair,
sunglasses, upper-clothes, skirt, pants, dress, belt, left/right-shoe,
face, left/right-leg, left/right-arm, bag, scarf -- the standard ATR label
set, documented on the `mattmdjaga/segformer_b2_clothes` model card, which
was finetuned on this exact dataset). Fields verified live: `image` (RGB),
`mask` (single-channel "L" mode, pixel values ARE class indices directly,
0-17 -- confirmed by inspecting `np.unique()` on real mask arrays). Only a
`train` split exists (no held-out eval split shipped), so this script
carves one out via `train_test_split` -- documented explicitly since it
differs from every other script in this project, which read
already-separate train/eval splits.

**A real dataset schema pitfall avoided**: `nateraw/pascal-voc-2012` (the
other segmentation dataset considered) stores masks as 3-channel RGB
images where pixel COLORS encode class via VOC's official palette
(confirmed via `np.unique()` showing values like `[0, 128, 192, 224]`,
not small sequential class indices) -- using it directly as a
single-channel label map would silently train on nonsense labels. Chosen
`mattmdjaga/human_parsing_dataset` instead specifically because its masks
are already plain class-index images, verified before implementation
rather than assumed.

Model: nvidia/mit-b0 (SegFormer encoder, ImageNet-classification-pretrained,
no segmentation decode head) via AutoModelForSemanticSegmentation. The
decode head is necessarily fresh-initialized for this 18-class task --
verified via a direct LOAD REPORT that only `classifier.*` (the discarded
ImageNet head) and `decode_head.*` (the new segmentation head) are
missing/unexpected, with the actual encoder backbone loading cleanly --
the same diligence applied in image_classification.py and
object_detection.py before trusting a head-swap.

Metric: mean IoU + pixel accuracy, computed directly via numpy (not
`evaluate.load("mean_iou")`, which would fetch a script from the Hub --
this project has already hit `RuntimeError: Dataset scripts are no longer
supported` for a similarly hub-script-dependent load elsewhere, so this
avoids that whole risk class).

Usage:
    python image_segmentation.py --debug_first_batch --max_samples 20
"""

import argparse
import time

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation, Trainer, TrainingArguments

from common.data_selection import select_samples
from common.logging_utils import print_banner, print_config
from common.model_saving import save_model
from common.run_results import write_run_result
from common.seeding import set_all_seeds

DATASET_NAME = "mattmdjaga/human_parsing_dataset"
ARCHITECTURE = "encoder-only"
CLASS_NAMES = [
    "Background", "Hat", "Hair", "Sunglasses", "Upper-clothes", "Skirt", "Pants", "Dress",
    "Belt", "Left-shoe", "Right-shoe", "Face", "Left-leg", "Right-leg", "Left-arm", "Right-arm",
    "Bag", "Scarf",
]


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for this training script.

    Returns:
        argparse.ArgumentParser: Parser covering model choice, eval split
        sizing, optimization hyperparameters, sample selection, output
        paths, and debug/seed flags.
    """
    p = argparse.ArgumentParser(description="SFT a SegFormer model for semantic segmentation.")

    p.add_argument("--model", type=str, default="nvidia/mit-b0", help="SegFormer encoder checkpoint (no segmentation head -- one is trained fresh). Default: nvidia/mit-b0 (fp32 ~14MB, smallest SegFormer variant).")
    p.add_argument("--eval_fraction", type=float, default=0.05, help="Fraction of the (only) train split held out as eval via train_test_split, since this dataset ships no separate eval split. Default: 0.05.")

    p.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"], help="Mixed precision. Default: bf16.")
    p.add_argument("--batch_size", type=int, default=8, help="Per-device training batch size. Default: 8.")
    p.add_argument("--eval_batch_size", type=int, default=8, help="Per-device eval batch size. Default: 8.")
    p.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps. Default: 1.")
    p.add_argument("--learning_rate", type=float, default=6e-5, help="Learning rate. Default: 6e-5 (typical for SegFormer finetuning).")
    p.add_argument("--epochs", type=int, default=3, help="Training epochs. Default: 3.")
    p.add_argument("--warmup_steps", type=int, default=50, help="LR warmup steps. Default: 50.")
    p.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay. Default: 0.01.")

    p.add_argument("--max_samples", type=int, default=-1, help="Number of training rows to use. -1 (default) = full split.")
    p.add_argument("--sample_selection", type=str, default="random", choices=["random", "first", "last"], help="Which rows to pick when --max_samples is positive. Default: random.")
    p.add_argument("--max_eval_samples", type=int, default=-1, help="Eval rows to use. -1 (default) = full carved-out eval split.")

    p.add_argument("--output_dir", type=str, default="./output/supervised-finetuning/image/image_segmentation", help="Output directory.")
    p.add_argument("--logging_steps", type=int, default=10, help="Logging frequency. Default: 10.")
    p.add_argument("--eval_steps", type=int, default=100, help="Evaluation frequency. Default: 100.")
    p.add_argument("--save_steps", type=int, default=100, help="Checkpoint save frequency. Default: 100.")

    p.add_argument("--seed", type=int, default=42, help="Random seed. Default: 42.")
    p.add_argument("--debug_first_batch", action="store_true", default=False, help="Print formatted examples and exit without training.")

    return p


def verify_dataset() -> None:
    """Stream-peek one example and assert the mask is a single-channel class-index map.

    Prints the fields and one sample's image size/mask shape/unique values.
    Not called from `main()` (see the comment there) but kept for manual
    verification runs.

    Raises:
        AssertionError: If the mask is not a 2-D single-channel array (would
            indicate an RGB-palette-encoded mask like `nateraw/pascal-voc-2012`'s).
    """
    print_banner("VERIFYING DATASET")
    peek = load_dataset(DATASET_NAME, split="train", streaming=True)
    example = next(iter(peek))
    for field in ("image", "mask"):
        assert field in example, f"Expected a {field!r} field in {DATASET_NAME}, got fields: {list(example.keys())}"
    mask_arr = np.array(example["mask"])
    assert mask_arr.ndim == 2, f"Expected a single-channel class-index mask, got shape {mask_arr.shape} -- see module docstring's note on nateraw/pascal-voc-2012's RGB-palette pitfall."
    print(f"Dataset: {DATASET_NAME}")
    print(f"Fields: {list(example.keys())}")
    print(f"Sample: image.size={example['image'].size}, mask.shape={mask_arr.shape}, mask unique values={sorted(np.unique(mask_arr).tolist())}")
    print()


def load_and_prepare_data(args, processor):
    """Load the dataset, carve out an eval split, and wire up on-the-fly mask/pixel preprocessing.

    Args:
        args (argparse.Namespace): Parsed CLI args; uses `eval_fraction`,
            `seed`, `max_samples`, `sample_selection`, and `max_eval_samples`.
        processor: A Hugging Face `AutoImageProcessor` instance used to
            jointly preprocess images and segmentation maps.

    Returns:
        tuple: `(train_dataset, eval_dataset)`, `datasets.Dataset` objects
        with a lazy `with_transform` preprocessing pipeline attached.
    """
    print_banner("LOADING DATASET")
    full = load_dataset(DATASET_NAME, split="train")
    split = full.train_test_split(test_size=args.eval_fraction, seed=args.seed)
    train_raw, eval_raw = split["train"], split["test"]

    train_raw = select_samples(train_raw, args.max_samples, args.sample_selection, args.seed)
    eval_raw = select_samples(eval_raw, args.max_eval_samples, "first", args.seed)

    def transform(examples):
        """Batch-transform raw images and masks into model-ready pixel values and labels."""
        images = [img.convert("RGB") for img in examples["image"]]
        masks = [mask.convert("L") for mask in examples["mask"]]
        encoding = processor(images=images, segmentation_maps=masks, return_tensors="pt")
        return {"pixel_values": encoding["pixel_values"], "labels": encoding["labels"]}

    train_dataset = train_raw.with_transform(transform)
    eval_dataset = eval_raw.with_transform(transform)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Eval samples: {len(eval_dataset)}")
    print()
    return train_dataset, eval_dataset


def collate_fn(batch):
    """Stack a list of per-example feature dicts into a batched tensor dict.

    Args:
        batch (list[dict]): Examples, each with fixed-shape `pixel_values`
            and a `labels` segmentation map.

    Returns:
        dict: `{"pixel_values": Tensor, "labels": Tensor}` batched tensors.
    """
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    return {"pixel_values": pixel_values, "labels": labels}


def print_formatted_examples_segmentation(dataset, num_examples: int = 2) -> None:
    """Print a few dataset examples' tensor shapes and present classes for manual inspection.

    Args:
        dataset: A dataset with `__getitem__` returning transformed examples.
        num_examples (int): Number of examples to print. Default: 2.
    """
    print_banner("FORMATTED EXAMPLES")
    for i in range(min(num_examples, len(dataset))):
        example = dataset[i]
        labels = example["labels"]
        present_classes = sorted(torch.unique(labels).tolist())
        print(f"\n--- Example {i + 1} ---")
        print(f"pixel_values.shape: {tuple(example['pixel_values'].shape)}, labels.shape: {tuple(labels.shape)}")
        print(f"Classes present: {[CLASS_NAMES[c] for c in present_classes if c < len(CLASS_NAMES)]}")
    print()


def compute_mean_iou(logits, labels, num_labels: int):
    """Upsample low-res segmentation logits and compute per-class mean IoU and pixel accuracy.

    Args:
        logits (numpy.ndarray): Low-res model output, shape (N, C, h, w).
        labels (numpy.ndarray): Full-res class-index masks, shape (N, H, W).
        num_labels (int): Number of segmentation classes.

    Returns:
        dict: `{"mean_iou": float, "pixel_accuracy": float}`. Classes absent
        from both predictions and labels in this batch are skipped rather
        than counted as a meaningless 0/0 IoU.
    """
    logits_tensor = torch.from_numpy(logits)
    labels_tensor = torch.from_numpy(labels)
    upsampled = F.interpolate(logits_tensor, size=labels_tensor.shape[-2:], mode="bilinear", align_corners=False)
    predictions = upsampled.argmax(dim=1).numpy()
    labels_np = labels_tensor.numpy()

    ious = []
    correct = (predictions == labels_np).sum()
    total = predictions.size
    for cls in range(num_labels):
        pred_cls = predictions == cls
        label_cls = labels_np == cls
        union = np.logical_or(pred_cls, label_cls).sum()
        if union == 0:
            continue  # class not present in this eval batch at all -- skip rather than count a meaningless 0/0
        intersection = np.logical_and(pred_cls, label_cls).sum()
        ious.append(intersection / union)

    return {
        "mean_iou": float(np.mean(ious)) if ious else 0.0,
        "pixel_accuracy": float(correct / total),
    }


def make_compute_metrics(num_labels: int):
    """Build a Trainer-compatible `compute_metrics` closure bound to a fixed class count.

    Args:
        num_labels (int): Number of segmentation classes, closed over by the returned function.

    Returns:
        Callable: A `compute_metrics(eval_pred)` function suitable for `transformers.Trainer`.
    """
    def compute_metrics(eval_pred):
        """Compute mean IoU and pixel accuracy for a Trainer eval pass.

        Args:
            eval_pred (tuple): `(logits, labels)` as passed by `transformers.Trainer`.

        Returns:
            dict: `{"mean_iou": float, "pixel_accuracy": float}`.
        """
        logits, labels = eval_pred
        return compute_mean_iou(logits, labels, num_labels)

    return compute_metrics


def main():
    """Run the end-to-end semantic segmentation SFT pipeline: load data, fine-tune SegFormer, evaluate, save."""
    args = build_arg_parser().parse_args()
    set_all_seeds(args.seed)
    print_config(args, "Semantic segmentation SFT (SegFormer)")

    # verify_dataset()  # commented out: avoids loading the dataset twice (a streaming peek here, then a full load in load_and_prepare_data()) to cut memory overhead
    processor = AutoImageProcessor.from_pretrained(args.model)

    train_dataset, eval_dataset = load_and_prepare_data(args, processor)

    torch_dtype = torch.bfloat16 if args.mixed_precision == "bf16" else (torch.float16 if args.mixed_precision == "fp16" else torch.float32)
    model = AutoModelForSemanticSegmentation.from_pretrained(
        args.model,
        num_labels=len(CLASS_NAMES),
        id2label={i: name for i, name in enumerate(CLASS_NAMES)},
        label2id={name: i for i, name in enumerate(CLASS_NAMES)},
        dtype=torch_dtype,
    )

    total_params = sum(p.numel() for p in model.parameters())
    print_banner("MODEL")
    print(f"Architecture: {ARCHITECTURE} ({args.model}), num_labels={len(CLASS_NAMES)}")
    print(f"Total parameters: {total_params:,}")
    print()

    if args.debug_first_batch:
        print_formatted_examples_segmentation(train_dataset, num_examples=2)
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
        remove_unused_columns=False,
        eval_accumulation_steps=4,  # keep eval logits off-GPU incrementally -- (N, 18, h, w) fp32 logits across a full eval split would otherwise accumulate in GPU memory
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collate_fn,
        compute_metrics=make_compute_metrics(len(CLASS_NAMES)),
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
        task="image_segmentation",
        modality="image",
        architecture=ARCHITECTURE,
        variant="standard",
        cot_enabled=False,
        model_name=args.model,
        dataset_name=DATASET_NAME,
        save_strategy="full",
        hyperparameters=vars(args),
        metrics={
            "mean_iou": eval_metrics.get("eval_mean_iou"),
            "pixel_accuracy": eval_metrics.get("eval_pixel_accuracy"),
            "eval_loss": eval_metrics.get("eval_loss"),
            "total_parameters": total_params,
        },
        num_train_samples=len(train_dataset),
        num_eval_samples=len(eval_dataset),
        train_runtime_seconds=train_runtime_seconds,
    )
    print("Done.")


if __name__ == "__main__":
    main()
