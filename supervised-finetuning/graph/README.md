# Graph (Supervised Finetuning)

Status: **done (Wave 5)**.

- `graph_classification.py` -- **done**. Dataset: `TUDataset("MUTAG")` via `torch_geometric`. Model: a from-scratch GIN (Graph Isomorphism Network).

## `graph_classification.py`

Dataset: **`TUDataset("MUTAG")`** -- 188 molecular graphs, binary
classification (mutagenic vs. non-mutagenic), 7 node features (one-hot
atom type), 4 edge features (bond type). Loaded via
`torch_geometric.datasets.TUDataset`, **not** `datasets.load_dataset()` --
deliberately: graph ML has its own standard data ecosystem
(`torch_geometric`'s built-in dataset zoo with automatic download), and
`torch_geometric` is already a pinned dependency in this project's
`requirements.txt` specifically for this stage (see that file's header
comment: *"present for future waves ... graph modality"*). Verified live
before implementation (188 graphs, 2 classes, 7 node features, one real
`Data` object's shape) rather than assumed from the dataset's reputation.
MUTAG has no predefined train/test split (unlike most HF datasets used
elsewhere in this project) -- carved via a random 85/15 split, `--seed`-controlled.

### Why this script trains from scratch

Same situation as `tabular/tabular_classification.py`: no transferable
"graph foundation model" exists for arbitrary graph-classification
schemas, so this trains a small GIN from scratch with a plain PyTorch/
`torch_geometric` training loop -- no HF model class or `Trainer`
involved, and no `save_model()` call for the same reason as the tabular
script (a custom `nn.Module`, no HF-compatible save format to reuse).

### Real training result

`--epochs 100` on an 85/15 random split of the 188 graphs: **82.1%
accuracy, 80.1% macro F1, 90.1% AUROC**. This lands squarely in the range
of published GIN results on MUTAG (typically 80-90% accuracy depending on
split, a well-studied small benchmark) -- a real, meaningful confirmation
the graph batching (`torch_geometric`'s `batch.batch` node-to-graph
assignment vector, used by `global_add_pool` to sum node embeddings back
into one vector per graph) and GIN message-passing implementation are
correct.

### Usage

```bash
python graph_classification.py --debug_first_batch
python graph_classification.py --epochs 100
```

### Dataset Reference (this folder's slice of the root README's master table)

| Script | Dataset | Split(s) | Role |
|---|---|---|---|
| `graph_classification.py` | `torch_geometric.datasets.TUDataset("MUTAG")` | random 85/15 split (no predefined split) | primary + only corpus |
