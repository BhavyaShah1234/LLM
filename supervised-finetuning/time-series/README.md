# Time Series (Supervised Finetuning)

Status: **done (Wave 5)**.

- `time_series_classification.py` -- **done**. Dataset: `mineshj1291/ecg-classification`. Model: a from-scratch FCN (1D-CNN, the standard UCR-benchmark baseline architecture).

## `time_series_classification.py`

Dataset: **`mineshj1291/ecg-classification`** -- 1920 2-lead ECG signal
segments (120 timesteps/channel), 3 perfectly balanced classes (640
examples each, beat-type categories). Fields verified live: `signals`
(`List(List(float64))`, shape (2 channels, 120 timesteps)), `labels`
(int8). Only a `train` split exists, so this script carves an eval split
out via `train_test_split`, same pattern as several other Wave 5 scripts.

### Real dataset search note

Several plausible HF time-series datasets turned out unusable on
inspection: `ETDataset/ett` and `HuggingFaceM4/something_something_v2`
both use the now-unsupported dataset-script loading mechanism
(`RuntimeError: Dataset scripts are no longer supported`), and
`JTF2000/TimeSeriesClassification` turned out to be a malformed CSV
upload -- each individual timestep became its own COLUMN with the header
row itself misinterpreted as data (verified by inspecting `list(example.keys())`,
which showed numeric-looking values like `'-0.57796699'` and
duplicate-suffixed names like `'-0.57796699.1'` as field NAMES, not
values). `mineshj1291/ecg-classification` was chosen after actually
loading and inspecting it live -- clean schema, balanced classes, real
usable shapes.

### Real API constraint found: `stratify_by_column` needs a `ClassLabel` type

`train_test_split(..., stratify_by_column="labels")` raised a live
`ValueError` -- stratification is only supported for `ClassLabel`-typed
columns, and this dataset's `labels` is a plain `Value(int8)`. Not a
blocker here: the dataset is already perfectly class-balanced (640/640/640),
so a plain (non-stratified) random split stays representative anyway.

### Why this script trains from scratch

Same situation as `tabular/tabular_classification.py` and
`graph/graph_classification.py`: no widely-used transferable "time-series
foundation model" exists for an arbitrary multi-channel signal
classification schema, so this trains a small FCN (Fully Convolutional
Network -- the standard UCR-benchmark baseline: stacked Conv1d/BatchNorm/
ReLU blocks, global average pooling over the time axis, linear classifier)
from scratch with a plain PyTorch training loop.

### Real training result

`--epochs 30` on the full dataset (~1632 train / 288 eval rows): **67.7%
accuracy, 66.8% macro F1** on a perfectly balanced 3-class task (chance =
33.3%) -- clearly, meaningfully better than random, with training loss
decreasing from 1.01 to ~0.42 over the run. Not a state-of-the-art result
(a real UCR-benchmark submission would use more careful architecture
tuning and possibly data augmentation), but a genuine, working
confirmation that the from-scratch FCN pipeline (signal normalization
computed from train data only, Conv1d over the time axis, global pooling)
is learning real signal, consistent with the toy-scale standard applied
throughout this project.

### Usage

```bash
python time_series_classification.py --debug_first_batch --max_samples 200
python time_series_classification.py --epochs 30
```

### Dataset Reference (this folder's slice of the root README's master table)

| Script | Dataset | Split(s) | Role |
|---|---|---|---|
| `time_series_classification.py` | `mineshj1291/ecg-classification` | `train`, carved into train/eval via `train_test_split` (dataset ships no separate eval split) | primary + only corpus |
