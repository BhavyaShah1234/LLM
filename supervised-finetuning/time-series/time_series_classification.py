"""Time-series classification from scratch.

Dataset: mineshj1291/ecg-classification -- 1920 2-lead ECG signal segments
(120 timesteps/channel), 3 balanced classes (640 examples each, 0/1/2 --
beat-type categories, standard for this kind of preprocessed ECG
benchmark). Fields verified live: `signals` (`List(List(float64))`, shape
(2 channels, 120 timesteps) per example), `labels` (int8). Only a `train`
split exists, so this script carves an eval split out via
`train_test_split`, same pattern as several other Wave 5 scripts.

**No pretrained checkpoint exists for this task**, same situation as
`tabular/tabular_classification.py` and `graph/graph_classification.py` --
there's no widely-used transferable "time-series foundation model" for an
arbitrary multi-channel signal-classification schema like this one, so
this trains a small 1D-CNN from scratch (the standard, well-established
"FCN" baseline architecture for UCR-style time-series classification
benchmarks: stacked Conv1d + BatchNorm + ReLU blocks, then global average
pooling over the time axis, then a linear classifier) with a plain PyTorch
training loop.

Usage:
    python time_series_classification.py --debug_first_batch --max_samples 200
"""

import argparse
import time

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, Dataset

from common.data_selection import select_samples
from common.logging_utils import print_banner, print_config
from common.run_results import write_run_result
from common.seeding import set_all_seeds

DATASET_NAME = "mineshj1291/ecg-classification"


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for this script.

    Returns:
        argparse.ArgumentParser: Parser covering model architecture,
        optimization, data-selection, output, and debug flags.
    """
    p = argparse.ArgumentParser(description="Train a from-scratch 1D-CNN (FCN baseline) for time-series classification.")

    p.add_argument("--channels", type=str, default="64,128,64", help="Comma-separated Conv1d channel widths for the 3 conv blocks. Default: 64,128,64 (the standard UCR-benchmark FCN baseline width).")
    p.add_argument("--kernel_sizes", type=str, default="7,5,3", help="Comma-separated Conv1d kernel sizes. Default: 7,5,3.")
    p.add_argument("--eval_fraction", type=float, default=0.15, help="Fraction of the (only) train split held out as eval via train_test_split. Default: 0.15.")

    p.add_argument("--batch_size", type=int, default=64, help="Training batch size. Default: 64.")
    p.add_argument("--eval_batch_size", type=int, default=128, help="Eval batch size. Default: 128.")
    p.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate. Default: 1e-3.")
    p.add_argument("--epochs", type=int, default=30, help="Training epochs. Default: 30.")
    p.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay. Default: 1e-5.")

    p.add_argument("--max_samples", type=int, default=-1, help="Number of training rows to use. -1 (default) = full split.")
    p.add_argument("--sample_selection", type=str, default="random", choices=["random", "first", "last"], help="Which rows to pick when --max_samples is positive. Default: random.")
    p.add_argument("--max_eval_samples", type=int, default=-1, help="Eval rows to use. -1 (default) = full carved-out eval split.")

    p.add_argument("--output_dir", type=str, default="./output/supervised-finetuning/time-series/time_series_classification", help="Output directory.")
    p.add_argument("--logging_steps", type=int, default=20, help="Logging frequency (in optimizer steps). Default: 20.")

    p.add_argument("--seed", type=int, default=42, help="Random seed. Default: 42.")
    p.add_argument("--debug_first_batch", action="store_true", default=False, help="Print formatted examples and exit without training.")

    return p


def verify_dataset() -> None:
    """Stream one example from DATASET_NAME and sanity-check its fields.

    Asserts that `signals` and `labels` are both present, and prints a
    preview. Not called from `main()` by default (see the comment there)
    but kept for manual verification.

    Raises:
        AssertionError: If either expected field is missing from the dataset.
    """
    print_banner("VERIFYING DATASET")
    peek = load_dataset(DATASET_NAME, split="train", streaming=True)
    example = next(iter(peek))
    for field in ("signals", "labels"):
        assert field in example, f"Expected a {field!r} field in {DATASET_NAME}, got fields: {list(example.keys())}"
    signals = example["signals"]
    print(f"Dataset: {DATASET_NAME}")
    print(f"Fields: {list(example.keys())}")
    print(f"Sample: {len(signals)} channels x {len(signals[0])} timesteps, label={example['labels']}")
    print()


class TimeSeriesDataset(Dataset):
    """Torch dataset that per-channel-normalizes ECG signal rows for the FCN.

    Attributes:
        rows: List of raw row dicts with `signals` and `labels` keys.
        channel_stats: List of `(mean, std)` tuples, one per channel,
            computed from the TRAIN split only (to avoid eval leakage).
    """

    def __init__(self, rows, channel_stats):
        """Initialize the dataset.

        Args:
            rows: List of raw row dicts (see class docstring).
            channel_stats: List of `(mean, std)` per-channel normalization
                stats, as returned by `compute_channel_stats`.
        """
        self.rows = rows
        self.channel_stats = channel_stats  # list of (mean, std), one per channel, computed from TRAIN split only

    def __len__(self):
        """Return the number of rows in the dataset.

        Returns:
            int: Number of rows.
        """
        return len(self.rows)

    def __getitem__(self, idx):
        """Build the normalized signal/label tensor pair for one row.

        Args:
            idx: Index of the row to fetch.

        Returns:
            dict: `signal` ((channels, timesteps) float tensor, normalized
            per-channel) and `labels` (scalar long tensor).
        """
        row = self.rows[idx]
        signal = torch.tensor(row["signals"], dtype=torch.float32)  # (channels, timesteps)
        for c in range(signal.shape[0]):
            mean, std = self.channel_stats[c]
            signal[c] = (signal[c] - mean) / std
        label = torch.tensor(row["labels"], dtype=torch.long)
        return {"signal": signal, "labels": label}


def compute_channel_stats(train_raw, num_channels: int):
    """Compute per-channel mean/std normalization stats from the train split.

    Args:
        train_raw: The raw (unnormalized) training `Dataset`, with a
            `signals` field of shape (channels, timesteps) per row.
        num_channels (int): Number of signal channels.

    Returns:
        list: `(mean, std)` tuples, one per channel, computed by
        concatenating that channel's values across all training rows (std
        is floored with a small epsilon to avoid division by zero).
    """
    stats = []
    for c in range(num_channels):
        values = np.concatenate([np.array(row["signals"][c]) for row in train_raw])
        stats.append((float(values.mean()), float(values.std() + 1e-6)))
    return stats


class FCN(nn.Module):
    """The standard "Fully Convolutional Network" baseline for UCR-style
    time-series classification: stacked Conv1d/BatchNorm/ReLU blocks,
    global average pooling over the time axis, linear classifier.

    Attributes:
        conv_blocks: Sequential stack of Conv1d/BatchNorm1d/ReLU blocks.
        classifier: Final linear layer mapping pooled channel features to
            class logits.
    """

    def __init__(self, in_channels: int, channels: list, kernel_sizes: list, num_classes: int):
        """Build the conv blocks and classifier head.

        Args:
            in_channels (int): Number of input signal channels.
            channels (list): Output channel width for each conv block.
            kernel_sizes (list): Conv1d kernel size for each conv block
                (same length as `channels`).
            num_classes (int): Number of output classes.
        """
        super().__init__()
        layers = []
        prev = in_channels
        for out_c, k in zip(channels, kernel_sizes):
            layers += [nn.Conv1d(prev, out_c, kernel_size=k, padding=k // 2), nn.BatchNorm1d(out_c), nn.ReLU()]
            prev = out_c
        self.conv_blocks = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev, num_classes)

    def forward(self, x):
        """Run the conv blocks, global-average-pool, and classify.

        Args:
            x: Input signal tensor of shape (batch, channels, timesteps).

        Returns:
            torch.Tensor: Class logits of shape (batch, num_classes).
        """
        x = self.conv_blocks(x)  # (batch, channels, timesteps)
        x = x.mean(dim=2)  # global average pool over time -> (batch, channels)
        return self.classifier(x)


def print_formatted_examples_timeseries(dataset, num_examples: int = 2) -> None:
    """Print shape/label info for a few dataset examples.

    Used by `--debug_first_batch` to visually confirm preprocessing before
    committing to a full training run.

    Args:
        dataset: A `TimeSeriesDataset` (or compatible indexable dataset) to
            sample from.
        num_examples (int): Maximum number of examples to print. Defaults
            to 2.
    """
    print_banner("FORMATTED EXAMPLES")
    for i in range(min(num_examples, len(dataset))):
        example = dataset[i]
        print(f"\n--- Example {i + 1} ---")
        print(f"signal.shape: {tuple(example['signal'].shape)}")
        print(f"Label: {example['labels'].item()}")
    print()


@torch.no_grad()
def evaluate(model, loader, device):
    """Run the model over a dataloader and compute classification metrics.

    Args:
        model: The `FCN` model to evaluate. Left in eval mode during
            inference and switched back to train mode before returning.
        loader: A `DataLoader` yielding `signal`/`labels` batches.
        device: Torch device to move batches to.

    Returns:
        dict: `loss` (mean cross-entropy), `accuracy`, `f1_macro`,
        `precision`, and `recall` (macro-averaged).
    """
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0
    loss_fn = nn.CrossEntropyLoss()
    for batch in loader:
        signal = batch["signal"].to(device)
        labels = batch["labels"].to(device)
        logits = model(signal)
        loss = loss_fn(logits, labels)
        total_loss += loss.item() * labels.size(0)
        all_preds.extend(logits.argmax(dim=1).cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
    model.train()
    return {
        "loss": total_loss / len(all_labels),
        "accuracy": accuracy_score(all_labels, all_preds),
        "f1_macro": f1_score(all_labels, all_preds, average="macro"),
        "precision": precision_score(all_labels, all_preds, average="macro", zero_division=0),
        "recall": recall_score(all_labels, all_preds, average="macro", zero_division=0),
    }


def main():
    """Run the full time-series classification pipeline: load, train, evaluate, record.

    Parses CLI args, loads and normalizes the ECG dataset, either dumps
    formatted debug examples and exits or trains the `FCN` model with a
    plain PyTorch loop, evaluates the result, and writes a
    `run_result.json` (no model checkpoint is saved for this from-scratch
    baseline).
    """
    args = build_arg_parser().parse_args()
    set_all_seeds(args.seed)
    print_config(args, "Time-series classification (from-scratch FCN, ECG)")

    # verify_dataset()  # commented out: avoids loading the dataset twice (a streaming peek here, then a full load in load_and_prepare_data()) to cut memory overhead
    full = load_dataset(DATASET_NAME, split="train")
    # stratify_by_column requires a ClassLabel-typed column (confirmed via a live
    # ValueError) -- this dataset's `labels` is a plain int8 Value, not ClassLabel.
    # Not needed anyway: the dataset is already perfectly class-balanced (640/640/640,
    # verified during dataset exploration), so a plain random split stays representative.
    split = full.train_test_split(test_size=args.eval_fraction, seed=args.seed)
    train_raw, eval_raw = split["train"], split["test"]

    num_channels = len(train_raw[0]["signals"])
    num_classes = len(set(train_raw["labels"]))
    channel_stats = compute_channel_stats(train_raw, num_channels)
    print(f"Channels: {num_channels}, classes: {num_classes}, per-channel (mean, std): {channel_stats}")

    train_raw = select_samples(train_raw, args.max_samples, args.sample_selection, args.seed)
    eval_raw = select_samples(eval_raw, args.max_eval_samples, "first", args.seed)

    train_dataset = TimeSeriesDataset(list(train_raw), channel_stats)
    eval_dataset = TimeSeriesDataset(list(eval_raw), channel_stats)
    print(f"Train samples: {len(train_dataset)}")
    print(f"Eval samples: {len(eval_dataset)}")
    print()

    if args.debug_first_batch:
        print_formatted_examples_timeseries(train_dataset, num_examples=2)
        print("--debug_first_batch set: exiting without training.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    channels = [int(c) for c in args.channels.split(",")]
    kernel_sizes = [int(k) for k in args.kernel_sizes.split(",")]
    model = FCN(num_channels, channels, kernel_sizes, num_classes).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print_banner("MODEL")
    print(f"FCN: channels={channels}, kernel_sizes={kernel_sizes}")
    print(f"Total parameters: {total_params:,}")
    print()

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=args.eval_batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    print_banner("TRAINING")
    start = time.time()
    step = 0
    for epoch in range(args.epochs):
        for batch in train_loader:
            signal = batch["signal"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            logits = model(signal)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

            step += 1
            if step % args.logging_steps == 0:
                print(f"epoch {epoch + 1}/{args.epochs}  step {step}  loss={loss.item():.4f}")
    train_runtime_seconds = time.time() - start

    print_banner("FINAL EVALUATION")
    eval_metrics = evaluate(model, eval_loader, device)
    print(eval_metrics)

    write_run_result(
        output_dir=args.output_dir,
        stage="supervised-finetuning",
        task="time_series_classification",
        modality="time-series",
        architecture="fcn-1d-cnn",
        variant="standard",
        cot_enabled=False,
        model_name="from-scratch-fcn",
        dataset_name=DATASET_NAME,
        save_strategy=None,
        hyperparameters=vars(args),
        metrics={**eval_metrics, "total_parameters": total_params},
        num_train_samples=len(train_dataset),
        num_eval_samples=len(eval_dataset),
        train_runtime_seconds=train_runtime_seconds,
    )
    print("Done.")


if __name__ == "__main__":
    main()
