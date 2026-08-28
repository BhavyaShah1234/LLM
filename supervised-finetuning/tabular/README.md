# Tabular (Supervised Finetuning)

Status: **done (Wave 5)**.

- `tabular_classification.py` -- **done**. Dataset: `scikit-learn/adult-census-income`. Model: a from-scratch entity-embedding MLP (no pretrained checkpoint exists for this task -- see below).

## `tabular_classification.py`

Dataset: **`scikit-learn/adult-census-income`** -- the classic UCI "Adult"/
Census Income dataset: predict whether income exceeds $50K from 6 numeric
(`age`, `fnlwgt`, `education.num`, `capital.gain`, `capital.loss`,
`hours.per.week`) and 8 categorical (`workclass`, `education`,
`marital.status`, `occupation`, `relationship`, `race`, `sex`,
`native.country`) features, verified live before implementation. Some
categorical values are literally the string `'?'` (the original UCI
data's missing-value placeholder) -- treated as its own category rather
than dropped or imputed. Only a `train` split exists, so this script
carves an eval split out via `train_test_split`, same pattern as several
other Wave 5 scripts.

### Why this script trains from scratch, unlike every other supervised-finetuning/ script

There's no pretrained checkpoint to start from -- unlike ViT/BERT/
Wav2Vec2/AST/VideoMAE, there's no widely-used "tabular foundation model"
for arbitrary mixed categorical/numeric feature sets (tabular models are
inherently dataset-specific: the categorical vocabulary and numeric
feature meaning are tied to this exact schema). This is the same
"nothing to adapt, genuinely training from scratch" situation as
`pretraining/clm.py` -- and correspondingly implemented with a plain
PyTorch training loop rather than HF's `Trainer`, since no HF model class
exists for this architecture (`EntityEmbeddingMLP`: a small learned
embedding table per categorical column, concatenated with normalized
numeric features, fed through a 2-layer MLP). For the same reason, this
script doesn't call `common/model_saving.py`'s `save_model()` -- that
helper is built around HF `PreTrainedModel` + tokenizer/processor pairs,
and this is a custom `nn.Module` with hand-rolled categorical vocabularies
that don't fit that interface without extra plumbing not worth building
for one script.

### Real training result

`--epochs 10` on the full ~27,700-row train split (2,000-ish held out for
eval): **86.1% accuracy, 79.4% macro F1, 91.3% AUROC**. This closely
matches published baselines for this exact dataset (typically 85-87%
accuracy for standard models), a real, meaningful confirmation the
pipeline (categorical vocab building, numeric normalization computed only
from train data, entity embeddings) is correct -- not just "loss went
down," but landing in the expected range for a well-studied benchmark.

### Usage

```bash
python tabular_classification.py --debug_first_batch --max_samples 200
python tabular_classification.py --epochs 10
```

### Dataset Reference (this folder's slice of the root README's master table)

| Script | Dataset | Split(s) | Role |
|---|---|---|---|
| `tabular_classification.py` | `scikit-learn/adult-census-income` | `train`, carved into train/eval via `train_test_split` (dataset ships no separate eval split) | primary + only corpus |
