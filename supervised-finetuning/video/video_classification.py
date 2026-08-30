"""Video classification SFT.

Dataset: nateraw/kinetics-mini -- a tiny 5-class Kinetics subset (archery,
bowling, flying_kite, high_jump, marching), 50 train / 50 validation
10-second clips. Fields verified live: `video` (torchcodec `VideoDecoder`),
`label` (int, real `ClassLabel` names). The commonly-tutorial-referenced
`sayakpaul/ucf101-subset` is NOT `datasets.load_dataset()`-loadable at all
-- it's a raw asset repo (a `.tar.gz` + loose `.avi` files, no dataset
loading script), meant for manual `hf_hub_download` + `tarfile.extractall`
+ a hand-rolled `pytorchvideo`-based `Dataset`, which every other script in
this project's `supervised-finetuning/` deliberately avoids (all use
`load_dataset()` directly). `nateraw/kinetics-mini` was chosen specifically
because it IS `load_dataset()`-loadable with the same pattern as every
other script here -- verified live before implementation, not assumed.

**Real API note**: the `video` column decodes via
`torchcodec.decoders.VideoDecoder` in this project's pinned `datasets`
version. `video.get_frames_at(indices=[...])` returns a `FrameBatch` with
`.data` shape `(len(indices), 3, H, W)` uint8 -- used to uniformly sample
16 frames per clip (`AutoConfig.from_pretrained(...).num_frames == 16` for
this model, confirmed before hardcoding it).

Model: MCG-NJU/videomae-base (VideoMAE, self-supervised-pretrained on
Kinetics, no classification head) via AutoModelForVideoClassification.

**Real bug found and fixed**: unlike every other head-swapped backbone in
this wave, VideoMAE's checkpoint stores attention biases under different
key names than this project's `transformers==5.14.1` expects --
`q_bias`/`v_bias` (the original VideoMAE/BEiT design: no learned bias for
the key projection at all) vs. this version's `query.bias`/`key.bias`/
`value.bias` (three separate biases). Confirmed via a direct LOAD REPORT
on the BARE backbone (`AutoModel`, not just the classification wrapper --
ruling out the wrapper-specific bug class found in
`image/image_classification.py`) that `query.bias`/`key.bias`/
`value.bias` are all reported MISSING (freshly zero-initialized) even
though the checkpoint's weight matrices (`query.weight`, etc.) load fine.
Fixed by manually copying the checkpoint's `q_bias`/`v_bias` tensors into
the loaded model's `query.bias`/`value.bias` per layer (`key.bias` is
correctly left at zero -- the checkpoint never had a key bias to begin
with, matching the original architecture, not a gap to fill). Verified via
exact tensor equality against the raw checkpoint's `q_bias` after the fix.

Usage:
    python video_classification.py --debug_first_batch --max_samples 20
"""

import argparse
import time

import numpy as np
import torch
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import AutoImageProcessor, AutoModelForVideoClassification, Trainer, TrainingArguments

from common.data_selection import select_samples
from common.logging_utils import print_banner, print_config
from common.model_saving import save_model
from common.run_results import write_run_result
from common.seeding import set_all_seeds

DATASET_NAME = "nateraw/kinetics-mini"
ARCHITECTURE = "encoder-only"


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for this script.

    Returns:
        argparse.ArgumentParser: Parser covering model, optimization,
        data-selection, output, and debug flags.
    """
    p = argparse.ArgumentParser(description="SFT a VideoMAE model for video classification.")

    p.add_argument("--model", type=str, default="MCG-NJU/videomae-base", help="VideoMAE checkpoint (no classification head -- one is trained fresh). Default: MCG-NJU/videomae-base (fp32 ~340MB).")

    p.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"], help="Mixed precision. Default: bf16.")
    p.add_argument("--batch_size", type=int, default=2, help="Per-device training batch size. Default: 2 (16 frames x 224x224 per video is memory-heavy).")
    p.add_argument("--eval_batch_size", type=int, default=2, help="Per-device eval batch size. Default: 2.")
    p.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps. Default: 4.")
    p.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate. Default: 5e-5.")
    p.add_argument("--epochs", type=int, default=10, help="Training epochs. Default: 10 (dataset is tiny, 50 clips).")
    p.add_argument("--warmup_steps", type=int, default=20, help="LR warmup steps. Default: 20.")
    p.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay. Default: 0.01.")

    p.add_argument("--max_samples", type=int, default=-1, help="Number of training rows to use. -1 (default) = full split.")
    p.add_argument("--sample_selection", type=str, default="random", choices=["random", "first", "last"], help="Which rows to pick when --max_samples is positive. Default: random.")
    p.add_argument("--max_eval_samples", type=int, default=-1, help="Eval rows to use. -1 (default) = full eval split.")

    p.add_argument("--output_dir", type=str, default="./output/supervised-finetuning/video/video_classification", help="Output directory.")
    p.add_argument("--logging_steps", type=int, default=5, help="Logging frequency. Default: 5.")
    p.add_argument("--eval_steps", type=int, default=25, help="Evaluation frequency. Default: 25.")
    p.add_argument("--save_steps", type=int, default=25, help="Checkpoint save frequency. Default: 25.")

    p.add_argument("--seed", type=int, default=42, help="Random seed. Default: 42.")
    p.add_argument("--debug_first_batch", action="store_true", default=False, help="Print formatted examples and exit without training.")

    return p


def verify_dataset() -> None:
    """Stream one example from DATASET_NAME and sanity-check its fields.

    Asserts that `video` and `label` are both present, and prints a
    preview. Not called from `main()` by default (see the comment there)
    but kept for manual verification.

    Raises:
        AssertionError: If either expected field is missing from the dataset.
    """
    print_banner("VERIFYING DATASET")
    peek = load_dataset(DATASET_NAME, split="train", streaming=True)
    example = next(iter(peek))
    for field in ("video", "label"):
        assert field in example, f"Expected a {field!r} field in {DATASET_NAME}, got fields: {list(example.keys())}"
    video = example["video"]
    print(f"Dataset: {DATASET_NAME}")
    print(f"Fields: {list(example.keys())}")
    print(f"Sample: label={example['label']}, num_frames={len(video)}, size={video.metadata.width}x{video.metadata.height}")
    print()


def sample_frames(video, num_frames: int) -> list:
    """Uniformly sample and decode a fixed number of frames from a clip.

    Args:
        video: A `torchcodec.decoders.VideoDecoder`-backed dataset video
            column value.
        num_frames (int): Number of frames to sample (evenly spaced across
            the clip, via `np.linspace`).

    Returns:
        list: `num_frames` HxWxC uint8 numpy arrays, in the layout the
        image processor expects.
    """
    total = len(video)
    indices = np.linspace(0, total - 1, num_frames).astype(int).tolist()
    frames = video.get_frames_at(indices=indices)
    return [f.permute(1, 2, 0).numpy() for f in frames.data]  # (C, H, W) uint8 -> (H, W, C) numpy, what the image processor expects


def fix_videomae_attention_bias(model, checkpoint_name: str) -> None:
    """Copy VideoMAE's q_bias/v_bias checkpoint tensors into the loaded model.

    See module docstring: the checkpoint stores q_bias/v_bias (no key
    bias, by original-architecture design); this project's transformers
    version expects separate query/key/value.bias and leaves them at zero
    when it can't find a match. Copies the real values back in, per
    encoder layer, by downloading and reading the checkpoint's raw
    safetensors file.

    Args:
        model: The loaded `AutoModelForVideoClassification` instance whose
            backbone attention biases need correcting.
        checkpoint_name (str): Hub checkpoint id to download
            `model.safetensors` from.
    """
    backbone = getattr(model, model.base_model_prefix)
    path = hf_hub_download(checkpoint_name, "model.safetensors")
    with safe_open(path, framework="pt") as f:
        for i, layer in enumerate(backbone.encoder.layer):
            attn = layer.attention.attention
            q_key = f"{model.base_model_prefix}.encoder.layer.{i}.attention.attention.q_bias"
            v_key = f"{model.base_model_prefix}.encoder.layer.{i}.attention.attention.v_bias"
            if q_key not in f.keys():
                return  # checkpoint doesn't use this bias scheme -- nothing to fix
            with torch.no_grad():
                attn.query.bias.copy_(f.get_tensor(q_key).to(attn.query.bias.dtype))
                attn.value.bias.copy_(f.get_tensor(v_key).to(attn.value.bias.dtype))
    print(f"[video_classification] corrected {len(backbone.encoder.layer)} layers' query/value attention bias from checkpoint's q_bias/v_bias (see module docstring's real-bug note)")


def load_and_prepare_data(args, processor, num_frames: int):
    """Load the train/validation splits and attach a frame-sampling transform.

    Args:
        args: Parsed CLI namespace; uses `max_samples`, `sample_selection`,
            `max_eval_samples`, and `seed`.
        processor: The model's `AutoImageProcessor`, used inside the
            attached transform to preprocess sampled frames.
        num_frames (int): Number of frames to sample per clip (see
            `sample_frames`).

    Returns:
        tuple: `(train_dataset, eval_dataset, class_names)` where the first
        two are HF `Dataset`s with a lazy `with_transform` frame-sampling
        pipeline attached, and `class_names` is the list of label names.
    """
    print_banner("LOADING DATASET")
    train_raw = load_dataset(DATASET_NAME, split="train")
    eval_raw = load_dataset(DATASET_NAME, split="validation")

    class_names = train_raw.features["label"].names
    print(f"Classes ({len(class_names)}): {class_names}")

    train_raw = select_samples(train_raw, args.max_samples, args.sample_selection, args.seed)
    eval_raw = select_samples(eval_raw, args.max_eval_samples, "first", args.seed)

    def transform(examples):
        """Sample and preprocess frames for a batch of examples (lazy, per-access).

        Args:
            examples (dict): Batch with `video` (list of video file paths)
                and `label` (integer class label) columns.

        Returns:
            dict: `{"pixel_values", "labels"}` ready for the model.
        """
        pixel_values = []
        for video in examples["video"]:
            frames = sample_frames(video, num_frames)
            encoding = processor(frames, return_tensors="pt")
            pixel_values.append(encoding["pixel_values"][0])
        return {"pixel_values": pixel_values, "labels": examples["label"]}

    train_dataset = train_raw.with_transform(transform)
    eval_dataset = eval_raw.with_transform(transform)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Eval samples: {len(eval_dataset)}")
    print()
    return train_dataset, eval_dataset, class_names


def collate_fn(batch):
    """Stack a list of transformed examples into a training batch.

    Args:
        batch: List of dicts with `pixel_values` and `labels` keys, as
            produced by the `transform` closure in `load_and_prepare_data`.

    Returns:
        dict: Batched `pixel_values` and `labels` tensors.
    """
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    labels = torch.tensor([item["labels"] for item in batch])
    return {"pixel_values": pixel_values, "labels": labels}


def print_formatted_examples_video(dataset, class_names, num_examples: int = 2) -> None:
    """Print pixel-tensor shape and label info for a few dataset examples.

    Used by `--debug_first_batch` to visually confirm preprocessing before
    committing to a full training run.

    Args:
        dataset: A transformed dataset (see `load_and_prepare_data`) to
            sample from.
        class_names: List of label names, indexed by label id.
        num_examples (int): Maximum number of examples to print. Defaults
            to 2.
    """
    print_banner("FORMATTED EXAMPLES")
    for i in range(min(num_examples, len(dataset))):
        example = dataset[i]
        print(f"\n--- Example {i + 1} ---")
        print(f"pixel_values.shape: {tuple(example['pixel_values'].shape)} (frames, channels, H, W)")
        print(f"Label: {example['labels']} ({class_names[example['labels']]})")
    print()


def compute_metrics(eval_pred):
    """Compute classification metrics from Trainer eval predictions.

    Args:
        eval_pred: `(logits, labels)` tuple as passed by
            `transformers.Trainer`'s `compute_metrics` hook.

    Returns:
        dict: `accuracy`, `f1_macro`, `precision`, and `recall`
        (macro-averaged).
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1_macro": f1_score(labels, predictions, average="macro"),
        "precision": precision_score(labels, predictions, average="macro", zero_division=0),
        "recall": recall_score(labels, predictions, average="macro", zero_division=0),
    }


def main():
    """Run the full video classification SFT pipeline: load, train, evaluate, save, record.

    Parses CLI args, loads the base VideoMAE model (fixing its checkpoint's
    attention-bias key mismatch), either dumps formatted debug examples and
    exits or trains via `Trainer`, evaluates the result, saves the model,
    and writes a `run_result.json`.
    """
    args = build_arg_parser().parse_args()
    set_all_seeds(args.seed)
    print_config(args, "Video classification SFT (VideoMAE)")

    # verify_dataset()  # commented out: avoids loading the dataset twice (a streaming peek here, then a full load in load_and_prepare_data()) to cut memory overhead
    processor = AutoImageProcessor.from_pretrained(args.model)

    from transformers import AutoConfig

    num_frames = AutoConfig.from_pretrained(args.model).num_frames
    train_dataset, eval_dataset, class_names = load_and_prepare_data(args, processor, num_frames)

    torch_dtype = torch.bfloat16 if args.mixed_precision == "bf16" else (torch.float16 if args.mixed_precision == "fp16" else torch.float32)
    model = AutoModelForVideoClassification.from_pretrained(
        args.model,
        num_labels=len(class_names),
        id2label={i: name for i, name in enumerate(class_names)},
        label2id={name: i for i, name in enumerate(class_names)},
        dtype=torch_dtype,
    )
    fix_videomae_attention_bias(model, args.model)

    total_params = sum(p.numel() for p in model.parameters())
    print_banner("MODEL")
    print(f"Architecture: {ARCHITECTURE} ({args.model}), num_labels={len(class_names)}, num_frames={num_frames}")
    print(f"Total parameters: {total_params:,}")
    print()

    if args.debug_first_batch:
        print_formatted_examples_video(train_dataset, class_names, num_examples=2)
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
        task="video_classification",
        modality="video",
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
