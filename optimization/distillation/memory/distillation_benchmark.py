"""Knowledge Distillation -- isolated benchmark script (distillation, memory
-- primary placement; also reports execution-time and storage metrics, see
`../execution-time/README.md` and `../storage/README.md`, which
cross-reference this script rather than duplicating it).

Trains a small, randomly-initialized ("from scratch," zero prior
knowledge) student ViT purely by distilling from a real trained teacher --
this project's own `image_classification.py` output
(`./output/supervised-finetuning/image/image_classification`, 85.8M
params, 80.6% CIFAR-10 accuracy, verified to exist before implementing).
Standard combined distillation loss:

    loss = alpha * KL(student_soft_logits, teacher_soft_logits) + (1 - alpha) * CE(student_logits, hard_labels)

(Hinton et al. 2015) -- the student learns from BOTH the teacher's full
output distribution (which encodes relative confidence between wrong
classes -- "this looks a bit like a cat too" -- that hard labels alone
don't) and the ground-truth labels.

The student is a from-scratch `ViTConfig` sized down to roughly "ViT-Tiny"
(hidden_size=192, 4 layers, 3 heads) -- deliberately NOT a pretrained
checkpoint (would confound "did distillation transfer knowledge" with
"did the checkpoint already know this from its own pretraining").

Reports real metrics on all three resource axes this technique is
categorized under project-wide: parameter count and peak inference memory
(memory, this folder), inference throughput (execution-time), and
on-disk checkpoint size (storage) -- one training run produces all three,
consistent with this project's "measure once, cross-reference" policy for
techniques that legitimately span multiple resource axes.

Usage:
    python distillation_benchmark.py --debug_first_batch --max_samples 200
"""

import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoImageProcessor, AutoModelForImageClassification, ViTConfig, ViTForImageClassification

from common.data_selection import select_samples
from common.logging_utils import print_banner, print_config
from common.run_results import write_run_result
from common.seeding import set_all_seeds

DATASET_NAME = "uoft-cs/cifar10"
ARCHITECTURE = "encoder-only"
CLASS_NAMES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for the distillation benchmark.

    Returns:
        argparse.ArgumentParser: Parser covering teacher/student config,
        distillation hyperparameters, training settings, and I/O paths.
    """
    p = argparse.ArgumentParser(description="Distill a small from-scratch ViT student from this project's trained ViT-base teacher.")

    p.add_argument("--teacher_dir", type=str, default="./output/supervised-finetuning/image/image_classification", help="This project's own trained image classification teacher. Default: image_classification.py's output.")
    p.add_argument("--student_hidden_size", type=int, default=192, help="Student ViT hidden size (teacher is 768). Default: 192.")
    p.add_argument("--student_num_layers", type=int, default=4, help="Student ViT transformer layers (teacher has 12). Default: 4.")
    p.add_argument("--student_num_heads", type=int, default=3, help="Student ViT attention heads. Default: 3.")
    p.add_argument("--temperature", type=float, default=4.0, help="Distillation softmax temperature (Hinton et al. 2015 -- softens both distributions so the student can learn from relative confidence among wrong classes, not just the argmax). Default: 4.0.")
    p.add_argument("--alpha", type=float, default=0.7, help="Weight on the distillation (KL) loss vs. the hard-label CE loss. Default: 0.7.")

    p.add_argument("--batch_size", type=int, default=32, help="Training batch size. Default: 32.")
    p.add_argument("--eval_batch_size", type=int, default=64, help="Eval batch size. Default: 64.")
    p.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate. Default: 1e-3 (from-scratch student, higher LR than finetuning).")
    p.add_argument("--epochs", type=int, default=5, help="Training epochs. Default: 5.")
    p.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay. Default: 0.01.")

    p.add_argument("--max_samples", type=int, default=-1, help="Number of training rows to use. -1 (default) = full split.")
    p.add_argument("--sample_selection", type=str, default="random", choices=["random", "first", "last"], help="Which rows to pick when --max_samples is positive. Default: random.")
    p.add_argument("--max_eval_samples", type=int, default=-1, help="Eval rows to use. -1 (default) = full eval split.")

    p.add_argument("--output_dir", type=str, default="./output/optimization/distillation_benchmark", help="Output directory.")
    p.add_argument("--logging_steps", type=int, default=20, help="Logging frequency (in optimizer steps). Default: 20.")

    p.add_argument("--seed", type=int, default=42, help="Random seed. Default: 42.")
    p.add_argument("--debug_first_batch", action="store_true", default=False, help="Load teacher+student, run one distillation step, print losses, then exit without training.")

    return p


def load_and_prepare_data(args, processor):
    """Load CIFAR-10 train/test splits and wrap them with an image-processor transform.

    Args:
        args (argparse.Namespace): Parsed CLI args; uses `max_samples`, `sample_selection`,
            `max_eval_samples`, and `seed`.
        processor: HF image processor used to convert PIL images to pixel tensors.

    Returns:
        tuple: `(train_dataset, eval_dataset)`, HF datasets with an on-the-fly transform
        producing `pixel_values` and `labels`.
    """
    print_banner("LOADING DATASET")
    train_raw = load_dataset(DATASET_NAME, split="train")
    eval_raw = load_dataset(DATASET_NAME, split="test")

    train_raw = select_samples(train_raw, args.max_samples, args.sample_selection, args.seed)
    eval_raw = select_samples(eval_raw, args.max_eval_samples, "first", args.seed)

    def transform(examples):
        """Convert a batch of PIL images to pixel tensors via `processor`.

        Args:
            examples (dict): Batch with `img` (list of PIL images) and `label`.

        Returns:
            dict: `{"pixel_values", "labels"}` ready for the model.
        """
        pixel_values = processor([img.convert("RGB") for img in examples["img"]], return_tensors="pt")["pixel_values"]
        return {"pixel_values": pixel_values, "labels": examples["label"]}

    train_dataset = train_raw.with_transform(transform)
    eval_dataset = eval_raw.with_transform(transform)
    print(f"Train samples: {len(train_dataset)}")
    print(f"Eval samples: {len(eval_dataset)}")
    return train_dataset, eval_dataset


def collate_fn(batch):
    """Stack a list of transformed examples into a single batch tensor dict.

    Args:
        batch (list[dict]): Examples each with `pixel_values` and `labels`.

    Returns:
        dict: `{"pixel_values", "labels"}` batched tensors.
    """
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    labels = torch.tensor([item["labels"] for item in batch])
    return {"pixel_values": pixel_values, "labels": labels}


def build_student(hidden_size: int, num_layers: int, num_heads: int, num_classes: int) -> ViTForImageClassification:
    """Construct a randomly-initialized, scaled-down ViT to serve as the distillation student.

    Args:
        hidden_size (int): Transformer hidden size.
        num_layers (int): Number of transformer layers.
        num_heads (int): Number of attention heads.
        num_classes (int): Number of output classes.

    Returns:
        ViTForImageClassification: A from-scratch (non-pretrained) student model.
    """
    config = ViTConfig(
        hidden_size=hidden_size,
        num_hidden_layers=num_layers,
        num_attention_heads=num_heads,
        intermediate_size=hidden_size * 4,
        image_size=224,
        patch_size=16,
        num_labels=num_classes,
    )
    return ViTForImageClassification(config)  # random init -- no pretrained weights, all knowledge must come from distillation


def distillation_loss(student_logits, teacher_logits, labels, temperature: float, alpha: float):
    """Compute the combined KL (soft-target) + cross-entropy (hard-label) distillation loss.

    Args:
        student_logits (torch.Tensor): Student model output logits.
        teacher_logits (torch.Tensor): Teacher model output logits.
        labels (torch.Tensor): Ground-truth class labels.
        temperature (float): Softmax temperature used to soften both distributions.
        alpha (float): Weight on the KD loss vs. the hard-label CE loss.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]: `(total_loss, kd_loss, ce_loss)`.
    """
    soft_targets = F.log_softmax(student_logits / temperature, dim=-1)
    soft_teacher = F.softmax(teacher_logits / temperature, dim=-1)
    kd_loss = F.kl_div(soft_targets, soft_teacher, reduction="batchmean") * (temperature**2)
    ce_loss = F.cross_entropy(student_logits, labels)
    return alpha * kd_loss + (1 - alpha) * ce_loss, kd_loss, ce_loss


@torch.no_grad()
def evaluate(student, loader, device):
    """Run inference over `loader` and compute accuracy and macro F1.

    Args:
        student: Model to evaluate (also used for the teacher).
        loader (torch.utils.data.DataLoader): Batches of `pixel_values`/`labels`.
        device (str): Device to run inference on.

    Returns:
        dict: `{"accuracy", "f1_macro"}` evaluation metrics.
    """
    student.eval()
    all_preds, all_labels = [], []
    for batch in loader:
        pixel_values = batch["pixel_values"].to(device)
        logits = student(pixel_values=pixel_values).logits
        all_preds.extend(logits.argmax(dim=-1).cpu().tolist())
        all_labels.extend(batch["labels"].tolist())
    student.train()
    return {"accuracy": accuracy_score(all_labels, all_preds), "f1_macro": f1_score(all_labels, all_preds, average="macro")}


@torch.no_grad()
def benchmark_inference(model, pixel_values, num_repeats: int = 20):
    """Time average per-call inference latency for a single fixed batch.

    Runs 3 warmup calls (untimed) before timing `num_repeats` calls.

    Args:
        model: Model to benchmark.
        pixel_values (torch.Tensor): Fixed input batch reused for every call.
        num_repeats (int): Number of timed forward passes. Defaults to 20.

    Returns:
        float: Average seconds per forward pass.
    """
    model.eval()
    device = next(model.parameters()).device
    pixel_values = pixel_values.to(device)
    for _ in range(3):  # warmup
        model(pixel_values=pixel_values)
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(num_repeats):
        model(pixel_values=pixel_values)
    torch.cuda.synchronize()
    return (time.time() - start) / num_repeats


def dir_size_bytes(path: str) -> int:
    """Recursively sum the size of every file under `path`.

    Args:
        path (str): Directory to measure.

    Returns:
        int: Total size in bytes.
    """
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            total += os.path.getsize(os.path.join(root, f))
    return total


def main():
    """Run the end-to-end distillation benchmark: load the trained teacher and a
    from-scratch student, distill, then report accuracy, inference latency, and
    checkpoint-size deltas via `write_run_result` (or exit early with
    `--debug_first_batch`).
    """
    args = build_arg_parser().parse_args()
    set_all_seeds(args.seed)
    print_config(args, "Knowledge distillation (from-scratch ViT-Tiny student, this project's trained ViT-base teacher)")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoImageProcessor.from_pretrained(args.teacher_dir, use_fast=True)

    train_dataset, eval_dataset = load_and_prepare_data(args, processor)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=args.eval_batch_size, shuffle=False, collate_fn=collate_fn)

    print_banner("LOADING TEACHER")
    teacher = AutoModelForImageClassification.from_pretrained(args.teacher_dir, dtype=torch.float32).to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    teacher_params = sum(p.numel() for p in teacher.parameters())
    print(f"Teacher: {teacher_params:,} params")

    print_banner("BUILDING STUDENT (random init, from scratch)")
    student = build_student(args.student_hidden_size, args.student_num_layers, args.student_num_heads, len(CLASS_NAMES)).to(device)
    student.train()
    student_params = sum(p.numel() for p in student.parameters())
    print(f"Student: {student_params:,} params ({100 * student_params / teacher_params:.1f}% of teacher's size)")

    if args.debug_first_batch:
        batch = next(iter(train_loader))
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)
        with torch.no_grad():
            teacher_logits = teacher(pixel_values=pixel_values).logits
        student_logits = student(pixel_values=pixel_values).logits
        loss, kd_loss, ce_loss = distillation_loss(student_logits, teacher_logits, labels, args.temperature, args.alpha)
        print(f"total_loss={loss.item():.4f}  kd_loss={kd_loss.item():.4f}  ce_loss={ce_loss.item():.4f}")
        print("--debug_first_batch set: exiting without training.")
        return

    optimizer = torch.optim.AdamW(student.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    print_banner("TRAINING (distillation)")
    start = time.time()
    step = 0
    for epoch in range(args.epochs):
        for batch in train_loader:
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            with torch.no_grad():
                teacher_logits = teacher(pixel_values=pixel_values).logits
            student_logits = student(pixel_values=pixel_values).logits
            loss, kd_loss, ce_loss = distillation_loss(student_logits, teacher_logits, labels, args.temperature, args.alpha)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step += 1
            if step % args.logging_steps == 0:
                print(f"epoch {epoch + 1}/{args.epochs}  step {step}  loss={loss.item():.4f}  kd={kd_loss.item():.4f}  ce={ce_loss.item():.4f}")
    train_runtime_seconds = time.time() - start

    print_banner("FINAL EVALUATION")
    teacher_metrics = evaluate(teacher, eval_loader, device)
    student_metrics = evaluate(student, eval_loader, device)
    print(f"Teacher: {teacher_metrics}")
    print(f"Student (distilled): {student_metrics}")

    print_banner("EXECUTION-TIME: single-batch inference latency")
    debug_batch = next(iter(eval_loader))["pixel_values"]
    teacher_latency = benchmark_inference(teacher, debug_batch)
    student_latency = benchmark_inference(student, debug_batch)
    print(f"Teacher: {teacher_latency * 1000:.2f} ms/batch  |  Student: {student_latency * 1000:.2f} ms/batch  |  {teacher_latency / student_latency:.2f}x speedup")

    print_banner("STORAGE: checkpoint size")
    teacher_bytes = dir_size_bytes(args.teacher_dir)
    student_dir = os.path.join(args.output_dir, "student_checkpoint")
    os.makedirs(student_dir, exist_ok=True)
    student.save_pretrained(student_dir)
    student_bytes = dir_size_bytes(student_dir)
    print(f"Teacher checkpoint: {teacher_bytes / 1e6:.1f} MB  |  Student checkpoint: {student_bytes / 1e6:.1f} MB  |  {100 * (1 - student_bytes / teacher_bytes):.1f}% smaller")

    write_run_result(
        output_dir=args.output_dir,
        stage="optimization",
        task="distillation_benchmark",
        modality="image",
        architecture=ARCHITECTURE,
        model_name=f"student(from-scratch-vit-tiny) distilled from teacher({args.teacher_dir})",
        dataset_name=DATASET_NAME,
        hyperparameters=vars(args),
        metrics={
            "teacher_accuracy": teacher_metrics["accuracy"],
            "student_accuracy": student_metrics["accuracy"],
            "teacher_f1_macro": teacher_metrics["f1_macro"],
            "student_f1_macro": student_metrics["f1_macro"],
            "teacher_params": teacher_params,
            "student_params": student_params,
            "student_param_frac_of_teacher": student_params / teacher_params,
            "teacher_latency_seconds": teacher_latency,
            "student_latency_seconds": student_latency,
            "inference_speedup": teacher_latency / student_latency,
            "teacher_checkpoint_bytes": teacher_bytes,
            "student_checkpoint_bytes": student_bytes,
            "checkpoint_size_reduction_frac": 1 - (student_bytes / teacher_bytes),
        },
        num_train_samples=len(train_dataset),
        num_eval_samples=len(eval_dataset),
        train_runtime_seconds=train_runtime_seconds,
    )
    print("Done.")


if __name__ == "__main__":
    main()
