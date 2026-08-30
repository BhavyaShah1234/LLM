"""Tabular classification from scratch.

Dataset: scikit-learn/adult-census-income (the classic UCI "Adult"/Census
Income dataset) -- predict whether income exceeds $50K from demographic
and occupation features. Fields verified live: 6 numeric (`age`, `fnlwgt`,
`education.num`, `capital.gain`, `capital.loss`, `hours.per.week`) + 8
categorical (`workclass`, `education`, `marital.status`, `occupation`,
`relationship`, `race`, `sex`, `native.country`) + binary target `income`
('<=50K' / '>50K'). Some categorical values are literally the string `'?'`
(missing-value placeholder in the original UCI data) -- treated as its own
category rather than dropped or imputed, the simplest correct handling.
Only a `train` split exists, so this script carves an eval split out via
`train_test_split`, same pattern as several other Wave 5 scripts (audio,
video, image segmentation).

**No pretrained checkpoint exists for this task** -- unlike every other
supervised-finetuning/ script, which starts from a pretrained backbone,
tabular models are dataset-specific (there's no transferable "tabular
foundation model" equivalent to ViT/BERT/Wav2Vec2 in general use for
mixed categorical/numeric feature sets like this). This script trains a
small entity-embedding MLP from scratch instead -- architecturally the
tabular-DL analogue of pretraining/clm.py's "no pretrained weights to
adapt, genuinely training from scratch" framing, and correspondingly
implemented with a plain PyTorch training loop rather than HF's Trainer
(no HF model class exists for this architecture to attach one to).

Usage:
    python tabular_classification.py --debug_first_batch --max_samples 200
"""

import argparse
import time

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset

from common.data_selection import select_samples
from common.logging_utils import print_banner, print_config
from common.run_results import write_run_result
from common.seeding import set_all_seeds

DATASET_NAME = "scikit-learn/adult-census-income"
NUMERIC_COLUMNS = ["age", "fnlwgt", "education.num", "capital.gain", "capital.loss", "hours.per.week"]
CATEGORICAL_COLUMNS = ["workclass", "education", "marital.status", "occupation", "relationship", "race", "sex", "native.country"]
TARGET_COLUMN = "income"


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for this training script.

    Returns:
        argparse.ArgumentParser: Parser covering MLP architecture sizing,
        eval split fraction, optimization hyperparameters, sample
        selection, output paths, and debug/seed flags.
    """
    p = argparse.ArgumentParser(description="Train a from-scratch entity-embedding MLP for tabular classification.")

    p.add_argument("--embedding_dim", type=int, default=16, help="Embedding dimension per categorical column. Default: 16.")
    p.add_argument("--hidden_dims", type=str, default="128,64", help="Comma-separated MLP hidden layer sizes. Default: 128,64.")
    p.add_argument("--dropout", type=float, default=0.2, help="Dropout probability between MLP layers. Default: 0.2.")
    p.add_argument("--eval_fraction", type=float, default=0.15, help="Fraction of the (only) train split held out as eval via train_test_split. Default: 0.15.")

    p.add_argument("--batch_size", type=int, default=256, help="Training batch size. Default: 256 (a tiny model + tabular rows -- much larger batches fit than any image/audio/video script here).")
    p.add_argument("--eval_batch_size", type=int, default=512, help="Eval batch size. Default: 512.")
    p.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate. Default: 1e-3 (typical for a small from-scratch MLP with Adam).")
    p.add_argument("--epochs", type=int, default=10, help="Training epochs. Default: 10.")
    p.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay. Default: 1e-5.")

    p.add_argument("--max_samples", type=int, default=-1, help="Number of training rows to use. -1 (default) = full split.")
    p.add_argument("--sample_selection", type=str, default="random", choices=["random", "first", "last"], help="Which rows to pick when --max_samples is positive. Default: random.")
    p.add_argument("--max_eval_samples", type=int, default=-1, help="Eval rows to use. -1 (default) = full carved-out eval split.")

    p.add_argument("--output_dir", type=str, default="./output/supervised-finetuning/tabular/tabular_classification", help="Output directory.")
    p.add_argument("--logging_steps", type=int, default=20, help="Logging frequency (in optimizer steps). Default: 20.")

    p.add_argument("--seed", type=int, default=42, help="Random seed. Default: 42.")
    p.add_argument("--debug_first_batch", action="store_true", default=False, help="Print formatted examples and exit without training.")

    return p


def verify_dataset() -> None:
    """Stream-peek one example from the dataset and assert the expected fields exist.

    Prints the numeric/categorical column names and a sample target value.
    Not called from `main()` (see the comment there) but kept for manual
    verification runs.
    """
    print_banner("VERIFYING DATASET")
    peek = load_dataset(DATASET_NAME, split="train", streaming=True)
    example = next(iter(peek))
    for field in NUMERIC_COLUMNS + CATEGORICAL_COLUMNS + [TARGET_COLUMN]:
        assert field in example, f"Expected a {field!r} field in {DATASET_NAME}, got fields: {list(example.keys())}"
    print(f"Dataset: {DATASET_NAME}")
    print(f"Numeric columns: {NUMERIC_COLUMNS}")
    print(f"Categorical columns: {CATEGORICAL_COLUMNS}")
    print(f"Target: {TARGET_COLUMN!r}, sample value: {example[TARGET_COLUMN]!r}")
    print()


def build_preprocessing(train_raw):
    """Derive numeric normalization stats and categorical vocabularies from the training split only.

    Computed on the TRAINING split only (never eval) -- standard practice,
    avoids leaking eval-set statistics into preprocessing.

    Args:
        train_raw: The raw (unselected) training split, indexable by column name.

    Returns:
        tuple: `(numeric_stats, vocabs, target_map)` where `numeric_stats`
        maps each numeric column to `(mean, std)`, `vocabs` maps each
        categorical column to a `{value: index}` map with 0 reserved for
        unseen values, and `target_map` maps the two target label strings
        to `{0, 1}`.

    Raises:
        AssertionError: If the target column has other than exactly two
            distinct values.
    """
    numeric_stats = {}
    for col in NUMERIC_COLUMNS:
        values = np.array(train_raw[col], dtype=np.float32)
        numeric_stats[col] = (float(values.mean()), float(values.std() + 1e-6))

    vocabs = {}
    for col in CATEGORICAL_COLUMNS:
        unique_values = sorted(set(train_raw[col]))
        vocabs[col] = {v: i + 1 for i, v in enumerate(unique_values)}  # 0 reserved for unseen-at-eval-time values

    target_values = sorted(set(train_raw[TARGET_COLUMN]))
    assert len(target_values) == 2, f"Expected a binary target, got {target_values}"
    target_map = {v: i for i, v in enumerate(target_values)}

    return numeric_stats, vocabs, target_map


class TabularDataset(Dataset):
    """A `torch.utils.data.Dataset` that normalizes numeric columns and vocab-indexes categorical columns per row.

    Attributes:
        rows: The underlying rows (list of dict-like records).
        numeric_stats (dict): Per-numeric-column `(mean, std)`, from `build_preprocessing`.
        vocabs (dict): Per-categorical-column `{value: index}` map, from `build_preprocessing`.
        target_map (dict): `{label string: 0 or 1}` binary target mapping.
    """

    def __init__(self, rows, numeric_stats, vocabs, target_map):
        """Store the rows and preprocessing artifacts used to featurize them on access.

        Args:
            rows: The underlying rows (list of dict-like records).
            numeric_stats (dict): Per-numeric-column `(mean, std)` normalization stats.
            vocabs (dict): Per-categorical-column `{value: index}` vocabulary.
            target_map (dict): `{label string: 0 or 1}` binary target mapping.
        """
        self.rows = rows
        self.numeric_stats = numeric_stats
        self.vocabs = vocabs
        self.target_map = target_map

    def __len__(self):
        """Return the number of rows in the dataset.

        Returns:
            int: Number of rows.
        """
        return len(self.rows)

    def __getitem__(self, idx):
        """Featurize one row into normalized numeric, vocab-indexed categorical, and label tensors.

        Args:
            idx (int): Row index.

        Returns:
            dict: `{"numeric": Tensor, "categorical": Tensor, "labels": Tensor}`.
        """
        row = self.rows[idx]
        numeric = torch.tensor(
            [(row[col] - self.numeric_stats[col][0]) / self.numeric_stats[col][1] for col in NUMERIC_COLUMNS],
            dtype=torch.float32,
        )
        categorical = torch.tensor(
            [self.vocabs[col].get(row[col], 0) for col in CATEGORICAL_COLUMNS], dtype=torch.long
        )
        label = torch.tensor(self.target_map[row[TARGET_COLUMN]], dtype=torch.long)
        return {"numeric": numeric, "categorical": categorical, "labels": label}


class EntityEmbeddingMLP(nn.Module):
    """A from-scratch MLP with learned entity embeddings for each categorical column, concatenated with normalized numeric features.

    Attributes:
        embeddings (nn.ModuleList): One `nn.Embedding` per categorical column.
        mlp (nn.Sequential): Feedforward classification head over the
            concatenated numeric + embedded-categorical feature vector.
    """

    def __init__(self, vocabs, num_numeric: int, embedding_dim: int, hidden_dims: list, dropout: float, num_classes: int):
        """Build one embedding table per categorical column plus a feedforward classification head.

        Args:
            vocabs (dict): Per-categorical-column `{value: index}` vocabulary,
                used to size each embedding table.
            num_numeric (int): Number of numeric input columns.
            embedding_dim (int): Embedding dimension shared by every categorical column.
            hidden_dims (list[int]): Hidden layer sizes for the MLP.
            dropout (float): Dropout probability applied after each hidden layer.
            num_classes (int): Number of output classes for the final linear layer.
        """
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(len(vocab) + 1, embedding_dim) for vocab in vocabs.values()])
        input_dim = num_numeric + embedding_dim * len(vocabs)
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev_dim, h), nn.ReLU(), nn.Dropout(dropout)]
            prev_dim = h
        layers.append(nn.Linear(prev_dim, num_classes))
        self.mlp = nn.Sequential(*layers)

    def forward(self, numeric, categorical):
        """Embed categorical columns, concatenate with numeric features, and classify.

        Args:
            numeric (torch.Tensor): Normalized numeric features, shape (batch, num_numeric).
            categorical (torch.Tensor): Vocab-indexed categorical features, shape (batch, num_categorical_columns).

        Returns:
            torch.Tensor: Class logits, shape (batch, num_classes).
        """
        embedded = [emb(categorical[:, i]) for i, emb in enumerate(self.embeddings)]
        x = torch.cat([numeric] + embedded, dim=1)
        return self.mlp(x)


def print_formatted_examples_tabular(dataset, target_map, num_examples: int = 2) -> None:
    """Print a few dataset examples' normalized/indexed features and labels for manual inspection.

    Args:
        dataset: A `TabularDataset`-like object with `__getitem__`.
        target_map (dict): `{label string: 0 or 1}` mapping, inverted here for display.
        num_examples (int): Number of examples to print. Default: 2.
    """
    print_banner("FORMATTED EXAMPLES")
    inv_target_map = {v: k for k, v in target_map.items()}
    for i in range(min(num_examples, len(dataset))):
        example = dataset[i]
        print(f"\n--- Example {i + 1} ---")
        print(f"numeric (normalized): {example['numeric'].tolist()}")
        print(f"categorical (vocab indices): {example['categorical'].tolist()}")
        print(f"Label: {example['labels'].item()} ({inv_target_map[example['labels'].item()]})")
    print()


@torch.no_grad()
def evaluate(model, loader, device):
    """Run a full evaluation pass and compute loss/accuracy/F1/precision/recall/AUROC.

    Args:
        model (EntityEmbeddingMLP): The model to evaluate; toggled to eval mode and back to train mode.
        loader (torch.utils.data.DataLoader): Loader over the eval rows.
        device (str): Device to move batches to (e.g. "cuda" or "cpu").

    Returns:
        dict: Metric name to float value.
    """
    model.eval()
    all_preds, all_labels, all_probs_positive = [], [], []
    total_loss = 0.0
    loss_fn = nn.CrossEntropyLoss()
    for batch in loader:
        numeric = batch["numeric"].to(device)
        categorical = batch["categorical"].to(device)
        labels = batch["labels"].to(device)
        logits = model(numeric, categorical)
        loss = loss_fn(logits, labels)
        total_loss += loss.item() * labels.size(0)
        probs = torch.softmax(logits, dim=1)
        all_preds.extend(logits.argmax(dim=1).cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
        all_probs_positive.extend(probs[:, 1].cpu().tolist())
    model.train()
    return {
        "loss": total_loss / len(all_labels),
        "accuracy": accuracy_score(all_labels, all_preds),
        "f1_macro": f1_score(all_labels, all_preds, average="macro"),
        "precision": precision_score(all_labels, all_preds, average="macro", zero_division=0),
        "recall": recall_score(all_labels, all_preds, average="macro", zero_division=0),
        "auroc": roc_auc_score(all_labels, all_probs_positive),
    }


def main():
    """Run the end-to-end tabular classification pipeline: load data, train a from-scratch entity-embedding MLP, evaluate, log results."""
    args = build_arg_parser().parse_args()
    set_all_seeds(args.seed)
    print_config(args, "Tabular classification (from-scratch entity-embedding MLP)")

    # verify_dataset()  # commented out: avoids loading the dataset twice (a streaming peek here, then a full load in load_and_prepare_data()) to cut memory overhead
    full = load_dataset(DATASET_NAME, split="train")
    split = full.train_test_split(test_size=args.eval_fraction, seed=args.seed)
    train_raw, eval_raw = split["train"], split["test"]

    numeric_stats, vocabs, target_map = build_preprocessing(train_raw)
    print(f"Categorical vocab sizes: {{k: len(v) for k, v in vocabs.items()}} = " f"{ {k: len(v) for k, v in vocabs.items()} }")
    print(f"Target classes: {target_map}")

    train_raw = select_samples(train_raw, args.max_samples, args.sample_selection, args.seed)
    eval_raw = select_samples(eval_raw, args.max_eval_samples, "first", args.seed)

    train_dataset = TabularDataset(list(train_raw), numeric_stats, vocabs, target_map)
    eval_dataset = TabularDataset(list(eval_raw), numeric_stats, vocabs, target_map)
    print(f"Train samples: {len(train_dataset)}")
    print(f"Eval samples: {len(eval_dataset)}")
    print()

    if args.debug_first_batch:
        print_formatted_examples_tabular(train_dataset, target_map, num_examples=2)
        print("--debug_first_batch set: exiting without training.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    hidden_dims = [int(h) for h in args.hidden_dims.split(",")]
    model = EntityEmbeddingMLP(vocabs, len(NUMERIC_COLUMNS), args.embedding_dim, hidden_dims, args.dropout, num_classes=2).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print_banner("MODEL")
    print(f"EntityEmbeddingMLP: {len(vocabs)} categorical embeddings (dim={args.embedding_dim}), hidden_dims={hidden_dims}")
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
            numeric = batch["numeric"].to(device)
            categorical = batch["categorical"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            logits = model(numeric, categorical)
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
        task="tabular_classification",
        modality="tabular",
        architecture="mlp-entity-embedding",
        variant="standard",
        cot_enabled=False,
        model_name="from-scratch-entity-embedding-mlp",
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
