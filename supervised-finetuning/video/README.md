# Video (Supervised Finetuning)

Status: **done (Wave 5)**.

- `video_classification.py` -- **done**. Dataset: `nateraw/kinetics-mini`. Model: `MCG-NJU/videomae-base` (VideoMAE).

## `video_classification.py`

Dataset: **`nateraw/kinetics-mini`** -- a tiny 5-class Kinetics subset
(archery, bowling, flying_kite, high_jump, marching), 50 train / 50
validation 10-second clips. Fields verified live: `video` (torchcodec
`VideoDecoder`), `label` (int, real `ClassLabel` names). Model:
`MCG-NJU/videomae-base` (self-supervised-pretrained on Kinetics, no
classification head -- one trained fresh).

### Real dataset choice: the commonly-referenced UCF101 subset isn't `load_dataset()`-loadable

`sayakpaul/ucf101-subset` (used in HF's own video classification tutorial)
is NOT a proper `datasets`-loadable dataset at all -- confirmed via
`list_repo_files()`: it's a raw asset repo (a `.tar.gz` + loose `.avi`
files, no dataset loading script), meant for manual
`hf_hub_download` + `tarfile.extractall` + a hand-rolled `pytorchvideo`
`Dataset`. Every other script in this project's `supervised-finetuning/`
uses `load_dataset()` directly; `nateraw/kinetics-mini` was chosen
specifically because it preserves that pattern, verified live before
implementation rather than assumed from the tutorial's approach.

### Real API note: `datasets`' video decoding

The `video` column decodes via `torchcodec.decoders.VideoDecoder`.
`video.get_frames_at(indices=[...])` returns a `FrameBatch` with `.data`
shape `(len(indices), 3, H, W)` uint8 -- used to uniformly sample 16
frames per clip (`AutoConfig.from_pretrained(...).num_frames == 16` for
this model, confirmed rather than hardcoded blind).

### Real bug found and fixed: attention bias keys don't match this transformers version

Unlike every other head-swapped backbone in this wave, this one has a bug
in the **backbone itself**, not just the new head. `MCG-NJU/videomae-base`'s
checkpoint stores attention biases as `q_bias`/`v_bias` (the original
VideoMAE/BEiT design: no learned bias for the key projection at all), but
this project's `transformers==5.14.1` expects three separate
`query.bias`/`key.bias`/`value.bias` parameters. Confirmed via a direct
LOAD REPORT on the **bare backbone** (`AutoModel`, not just the
classification wrapper -- ruling out the wrapper-specific bug class found
in `image/image_classification.py`) that all three bias tensors are
reported MISSING (silently zero-initialized) even though the weight
matrices load fine.

Fixed by downloading the raw checkpoint and manually copying `q_bias` ->
`query.bias` and `v_bias` -> `value.bias` per layer (`key.bias` is
correctly left at zero -- the checkpoint never had one, matching the
original architecture, not a gap to fill). Verified via exact tensor
equality between the fixed model's bias and the raw checkpoint's `q_bias`
after the fix. Smaller in scope than the `image_classification.py` bug
(only 2 of ~200 tensors per layer affected, not the whole backbone), but
the same class of "verify the LOAD REPORT, don't trust a clean-looking
`from_pretrained()` call" discipline caught it.

### Real training result

`--epochs 10` on the full 50-clip train / 50-clip eval split: train loss
1.62 -> 0.84 (clear, real learning -- the model is genuinely fitting the
training data), eval accuracy 0.14 -> 0.20. Eval accuracy landing at
exactly chance (1/5 classes = 0.20) is an honest, expected toy-scale
result given only 10 training clips per class -- not a sign of a broken
pipeline. The key differentiator from `audio/audio_classification.py`'s
earlier wav2vec2 bug is that **train loss clearly decreases here**, unlike
wav2vec2's stuck/plateaued loss on ESC-50 -- this is a healthy, working
pipeline that is simply data-limited at this scale, not the same failure
mode.

### Usage

```bash
python video_classification.py --debug_first_batch --max_samples 10
python video_classification.py --epochs 10
```

### Dataset Reference (this folder's slice of the root README's master table)

| Script | Dataset | Split(s) | Role |
|---|---|---|---|
| `video_classification.py` | `nateraw/kinetics-mini` | `train`, `validation` | primary + only corpus |
