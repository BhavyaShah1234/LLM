# Audio (Supervised Finetuning)

Status: **done (Wave 5)**.

- `audio_classification.py` -- **done**. Dataset: `ashraq/esc50`. Model: `MIT/ast-finetuned-audioset-10-10-0.4593` (Audio Spectrogram Transformer).

## `audio_classification.py`

Dataset: **`ashraq/esc50`** (ESC-50) -- 2000 5-second environmental sound
clips, 50 classes (dog, rain, crying_baby, chainsaw, ...). Fields verified
live: `filename`, `fold`, `target` (int), `category` (string), `esc10`,
`src_file`, `take`, `audio`. Only a `train` split exists, so this script
carves an eval split out via `train_test_split(test_size=0.1)`, same
pattern as `image/image_segmentation.py`. `category` is a raw string
column, so class names are derived live from the data (sorted unique
(target, category) pairs), not hardcoded.

### Real API note: `datasets==5.0.1`'s audio decoding changed

The `audio` column decodes via `datasets.features._torchcodec.AudioDecoder`
in this project's pinned `datasets` version, not the `{"array":...,
"sampling_rate":...}` dict some older HF audio tutorials assume --
confirmed via `type(example["audio"])`. Getting a raw waveform needs
`audio.get_all_samples()` (a `torchcodec.AudioSamples` with `.data`
(channels, samples) and `.sample_rate`). ESC-50's native rate is 44100 Hz;
downmixed to mono and resampled to 16000 Hz via
`torchaudio.functional.resample`.

### Real bug found and fixed: wrong model family for the task, caught via a decisive overfit sanity check

The first version used `facebook/wav2vec2-base` (speech-pretrained). A
real training run's loss barely moved (3.91 -> 3.85 over 5 epochs).
Rather than write this off as "just needs more toy-scale data," ran the
standard sanity check for exactly this symptom: **can the model overfit a
tiny subset it should trivially memorize?** On 10 examples (up to 10
distinct classes), even 100 epochs at 5x the learning rate with the
feature encoder unfrozen left loss plateaued at a nonzero constant
(2.164) with vanishing gradient (`grad_norm` -> 0.03) -- a textbook
mode-collapse signature, not slow convergence. Separately verified the
input pipeline was not at fault: `input_values` for 5 different real
examples showed correct per-sample normalization (mean~0, std~1), varying
min/max, and correct distinct labels -- no degenerate/constant data
anywhere.

This isolated the problem to the model choice: wav2vec2's speech-focused
pretraining transfers poorly to generic environmental sound classification
(a known community pain point). Switched to **MIT/ast-finetuned-audioset-10-10-0.4593**
(AST, pretrained on AudioSet -- general sound *events*, a much closer
domain match to ESC-50 than speech). The identical overfit check on the
identical 10-example subset: loss 0.664 -> 0.0015 in 30 epochs -- clean,
healthy memorization, confirming the pipeline was never the problem.

### Real check before trusting the head-swap

AST's checkpoint already has a 527-class AudioSet head, resized to this
task's 50 classes via `ignore_mismatched_sizes=True`. Verified via a
direct LOAD REPORT that only `classifier.dense.{weight,bias}`
MISMATCH-reinitializes (the head), with no missing backbone layers -- the
same diligence applied to every other head-swap in this wave, given this
exact flag combination previously (see `image/image_classification.py`)
silently dropped an entire backbone on a different architecture.

### Real training result

`--max_samples 600 --epochs 8`: **96.0% eval accuracy, 94.2% macro F1**,
training loss converging to ~0.008 by the final epochs. A strong, clean
result that further confirms the AST fix -- the earlier wav2vec2 result
never approached this regardless of hyperparameters tried.

### Usage

```bash
python audio_classification.py --debug_first_batch --max_samples 20
python audio_classification.py --max_samples 600 --epochs 8
```

### Dataset Reference (this folder's slice of the root README's master table)

| Script | Dataset | Split(s) | Role |
|---|---|---|---|
| `audio_classification.py` | `ashraq/esc50` | `train`, carved into train/eval via `train_test_split` (dataset ships no separate eval split) | primary + only corpus |
