"""Graph classification from scratch.

Dataset: TUDataset "MUTAG" -- 188 molecular graphs, binary classification
(mutagenic vs. non-mutagenic, the classic small graph-classification
benchmark). 7 node features (one-hot atom type), 4 edge features (bond
type). Loaded via `torch_geometric.datasets.TUDataset`, NOT
`datasets.load_dataset()` -- deliberately: graph ML has its own standard
data ecosystem (torch_geometric's built-in dataset zoo with automatic
download/caching), and `torch_geometric` is already a pinned dependency in
this project's `requirements.txt` specifically for this stage (see that
file's header comment). Verified live before implementation (dataset size,
num_classes, num_node_features, one real Data object's shape) rather than
assumed from the dataset's reputation.

**No pretrained checkpoint exists for this task**, same situation as
`tabular/tabular_classification.py` -- there's no transferable "graph
foundation model" for arbitrary graph-classification schemas, so this
trains a small GIN (Graph Isomorphism Network) from scratch, using a plain
PyTorch/`torch_geometric` training loop rather than HF's `Trainer` (no HF
model class exists for this architecture).

MUTAG has no predefined train/test split (unlike most HF datasets in this
project) -- common practice for TUDataset benchmarks is a random split,
done here via `torch.utils.data.random_split` with this script's `--seed`.

Usage:
    python graph_classification.py --debug_first_batch
"""

import argparse
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv, global_add_pool

from common.logging_utils import print_banner, print_config
from common.run_results import write_run_result
from common.seeding import set_all_seeds

DATASET_NAME = "MUTAG"
DATASET_ROOT = "./output/.torch_geometric_data"


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train a from-scratch GIN for graph classification (TUDataset MUTAG).")

    p.add_argument("--hidden_dim", type=int, default=64, help="GIN hidden dimension. Default: 64.")
    p.add_argument("--num_layers", type=int, default=3, help="Number of GIN layers. Default: 3.")
    p.add_argument("--dropout", type=float, default=0.2, help="Dropout probability. Default: 0.2.")
    p.add_argument("--eval_fraction", type=float, default=0.15, help="Fraction of the 188 graphs held out as eval (MUTAG has no predefined split). Default: 0.15.")

    p.add_argument("--batch_size", type=int, default=16, help="Training batch size (graphs per batch). Default: 16.")
    p.add_argument("--eval_batch_size", type=int, default=32, help="Eval batch size. Default: 32.")
    p.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate. Default: 1e-3.")
    p.add_argument("--epochs", type=int, default=100, help="Training epochs. Default: 100 (MUTAG is tiny -- 188 graphs total -- and GIN benchmarks conventionally train for many epochs on it).")
    p.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay. Default: 1e-4.")

    p.add_argument("--output_dir", type=str, default="./output/supervised-finetuning/graph/graph_classification", help="Output directory.")
    p.add_argument("--logging_steps", type=int, default=5, help="Logging frequency (in epochs). Default: 5.")

    p.add_argument("--seed", type=int, default=42, help="Random seed. Default: 42.")
    p.add_argument("--debug_first_batch", action="store_true", default=False, help="Print one batch's shapes and exit without training.")

    return p


def verify_dataset(dataset) -> None:
    print_banner("VERIFYING DATASET")
    print(f"Dataset: TUDataset({DATASET_NAME!r})")
    print(f"Num graphs: {len(dataset)}")
    print(f"Num classes: {dataset.num_classes}")
    print(f"Num node features: {dataset.num_node_features}")
    print(f"Sample graph: {dataset[0]}")
    print()


class GIN(nn.Module):
    def __init__(self, num_node_features: int, hidden_dim: int, num_layers: int, num_classes: int, dropout: float):
        super().__init__()
        self.convs = nn.ModuleList()
        in_dim = num_node_features
        for _ in range(num_layers):
            mlp = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
            self.convs.append(GINConv(mlp))
            in_dim = hidden_dim
        self.dropout = dropout
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, edge_index, batch):
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        graph_embedding = global_add_pool(x, batch)  # sum-pool node embeddings into one vector per graph
        graph_embedding = F.dropout(graph_embedding, p=self.dropout, training=self.training)
        return self.classifier(graph_embedding)


def print_formatted_batch(loader) -> None:
    print_banner("FORMATTED EXAMPLE BATCH")
    batch = next(iter(loader))
    print(f"batch.x.shape: {tuple(batch.x.shape)} (total nodes across batch, node feature dim)")
    print(f"batch.edge_index.shape: {tuple(batch.edge_index.shape)}")
    print(f"batch.y: {batch.y.tolist()}")
    print(f"batch.batch (graph assignment per node), first 20: {batch.batch[:20].tolist()}")
    print()


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels, all_probs_positive = [], [], []
    total_loss = 0.0
    loss_fn = nn.CrossEntropyLoss()
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch.x.float(), batch.edge_index, batch.batch)
        loss = loss_fn(logits, batch.y)
        total_loss += loss.item() * batch.y.size(0)
        probs = torch.softmax(logits, dim=1)
        all_preds.extend(logits.argmax(dim=1).cpu().tolist())
        all_labels.extend(batch.y.cpu().tolist())
        all_probs_positive.extend(probs[:, 1].cpu().tolist())
    model.train()
    return {
        "loss": total_loss / len(all_labels),
        "accuracy": accuracy_score(all_labels, all_preds),
        "f1_macro": f1_score(all_labels, all_preds, average="macro"),
        "precision": precision_score(all_labels, all_preds, average="macro", zero_division=0),
        "recall": recall_score(all_labels, all_preds, average="macro", zero_division=0),
        "auroc": roc_auc_score(all_labels, all_probs_positive) if len(set(all_labels)) > 1 else float("nan"),
    }


def main():
    args = build_arg_parser().parse_args()
    set_all_seeds(args.seed)
    print_config(args, "Graph classification (from-scratch GIN, TUDataset MUTAG)")

    dataset = TUDataset(root=DATASET_ROOT, name=DATASET_NAME)
    verify_dataset(dataset)

    dataset = dataset.shuffle()
    num_eval = int(len(dataset) * args.eval_fraction)
    eval_dataset = dataset[:num_eval]
    train_dataset = dataset[num_eval:]
    print(f"Train graphs: {len(train_dataset)}")
    print(f"Eval graphs: {len(eval_dataset)}")
    print()

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=args.eval_batch_size, shuffle=False)

    if args.debug_first_batch:
        print_formatted_batch(train_loader)
        print("--debug_first_batch set: exiting without training.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GIN(dataset.num_node_features, args.hidden_dim, args.num_layers, dataset.num_classes, args.dropout).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print_banner("MODEL")
    print(f"GIN: {args.num_layers} layers, hidden_dim={args.hidden_dim}")
    print(f"Total parameters: {total_params:,}")
    print()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    print_banner("TRAINING")
    start = time.time()
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            logits = model(batch.x.float(), batch.edge_index, batch.batch)
            loss = loss_fn(logits, batch.y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.y.size(0)
        if (epoch + 1) % args.logging_steps == 0:
            print(f"epoch {epoch + 1}/{args.epochs}  train_loss={epoch_loss / len(train_dataset):.4f}")
    train_runtime_seconds = time.time() - start

    print_banner("FINAL EVALUATION")
    eval_metrics = evaluate(model, eval_loader, device)
    print(eval_metrics)

    write_run_result(
        output_dir=args.output_dir,
        stage="supervised-finetuning",
        task="graph_classification",
        modality="graph",
        architecture="gin",
        variant="standard",
        cot_enabled=False,
        model_name="from-scratch-gin",
        dataset_name=f"torch_geometric.datasets.TUDataset({DATASET_NAME})",
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
