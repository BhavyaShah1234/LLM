# Image (Supervised Finetuning)

Status: **all four image tasks done**. Flat task files live directly in this folder (no architecture-family subfolder layer, unlike text/ -- see the root README's Wave 2 preview note on this deliberate divergence).

- `vqa_no_cot.py` / `vqa_cot.py` -- **done**. Dataset: `opendatalab/ChartVerse-SFT-1.8M` (streamed). Model: `Qwen/Qwen2-VL-2B` (base). See "Real bugs found and fixed" in `supervised-finetuning/README.md`.
- `image_classification.py` -- **done (Wave 5)**. Dataset: `uoft-cs/cifar10`. Model: `google/vit-base-patch16-224-in21k`.
- `object_detection.py` -- **done (Wave 5)**. Dataset: `rishitdagli/cppe-5`. Model: `facebook/detr-resnet-50`.
- `image_segmentation.py` -- **done (Wave 5)**. Dataset: `mattmdjaga/human_parsing_dataset`. Model: `nvidia/mit-b0` (SegFormer).

## `image_classification.py`

Dataset: **`uoft-cs/cifar10`** -- `img` (PIL image) + `label` (int 0-9, real
class names: airplane, automobile, bird, cat, deer, dog, frog, horse, ship,
truck), verified live before implementation. Architecturally analogous to
`text/classification/encoder-only/text_classification_standard.py`: a
classification head on a pretrained vision transformer, full-parameter
finetuning (small model, no `--lora`/`--quantization` needed, same
convention as that script).

### Real bug found and fixed: `AutoModelForImageClassification` silently drops the entire pretrained backbone on this checkpoint

`google/vit-base-patch16-224-in21k`'s checkpoint stores legacy key names
(`encoder.layer.0.attention.attention.query.weight`, etc. -- pre-refactor
ViT naming). `transformers==5.14.1`'s current `ViTModel` internally uses
different names (`vit.layers.0.attention.q_proj.weight`, a newer
LLaMA-style fused naming) and has a `_checkpoint_conversion_mapping` that
correctly remaps old keys to new ones on load -- confirmed via
`AutoModel.from_pretrained(...)` loading with **zero** missing/unexpected
keys.

**`AutoModelForImageClassification.from_pretrained(...)` on the identical
checkpoint does NOT apply this same remapping.** Its own printed load
report shows the ENTIRE backbone (every one of the 12 encoder layers) as
`MISSING` -- meaning the model would silently train from a
randomly-initialized backbone while claiming to finetune a pretrained
ViT. This would still technically "work" (loss decreases, some accuracy
achieved) which is exactly what makes it dangerous -- a from-scratch-ViT
result could easily be mistaken for a genuine finetuning result without
reading the load report closely.

Confirmed via a live diff: `ViTModel.from_pretrained(...)` (bare backbone)
loads cleanly; `ViTForImageClassification.from_pretrained(...)` (backbone +
head, same checkpoint) reports the entire backbone missing. Fixed by
loading the backbone separately (where the remapping works correctly) and
copying its state dict into the classification model's backbone submodule
after `from_pretrained` finishes:

```python
model = AutoModelForImageClassification.from_pretrained(args.model, num_labels=10, ...)
backbone = AutoModel.from_pretrained(args.model, dtype=torch_dtype)
getattr(model, model.base_model_prefix).load_state_dict(backbone.state_dict(), strict=False)
```

(`strict=False` because `AutoModel`'s `ViTModel` includes a pooler head
`ViTForImageClassification`'s internal backbone doesn't have --
`base_model_prefix` is the standard HF attribute name for a model's
backbone submodule, `"vit"` here, used instead of hardcoding `.vit` so the
same pattern could generalize to other architectures hitting this bug.)

**Verified the fix matters, not just cosmetically**: with the fix, a real
training run (`--max_samples 3000 --epochs 3`) reached **80.6% eval
accuracy** on 10-class CIFAR-10 in ~64 seconds of training. A
randomly-initialized ViT backbone could not plausibly reach that accuracy
from only 3000 training images in 3 epochs -- this result itself is strong
evidence the pretrained weights are genuinely loaded and being finetuned,
not a coincidence.

### Usage

```bash
python image_classification.py --debug_first_batch --max_samples 20
python image_classification.py --max_samples 3000 --max_eval_samples 500 --epochs 3
```

### Dataset Reference (this folder's slice of the root README's master table)

| Script | Dataset | Split(s) | Role |
|---|---|---|---|
| `image_classification.py` | `uoft-cs/cifar10` | `train` (training), `test` (eval) | primary + only corpus |
| `object_detection.py` | `rishitdagli/cppe-5` | `train` (training), `test` (eval) | primary + only corpus |
| `image_segmentation.py` | `mattmdjaga/human_parsing_dataset` | `train`, carved into train/eval via `train_test_split` (dataset ships no separate eval split) | primary + only corpus |

## `image_segmentation.py`

Dataset: **`mattmdjaga/human_parsing_dataset`** -- 17,706 (image, mask)
pairs, 18-class human body-part/clothing segmentation (the standard ATR
label set, documented on the `mattmdjaga/segformer_b2_clothes` model card,
which was finetuned on this exact dataset). Only a `train` split exists,
so this script carves an eval split out via `train_test_split(test_size=
0.05)` -- the one script in this project that does this, documented
explicitly since every other script reads already-separate splits. Model:
`nvidia/mit-b0` (SegFormer encoder, ImageNet-pretrained, no segmentation
head -- fresh decode head trained for this task).

### Real dataset pitfall avoided: `nateraw/pascal-voc-2012`'s masks aren't class indices

The other segmentation dataset considered, `nateraw/pascal-voc-2012`,
stores masks as 3-channel RGB images where pixel *colors* encode class via
VOC's official palette -- confirmed via `np.unique()` on a real mask
returning `[0, 128, 192, 224]`, not small sequential integers. Using it
directly as a class-index label map (as `AutoModelForSemanticSegmentation`
expects) would have silently trained on nonsense labels -- a real,
easy-to-miss trap given the mask still "looks like an image" and would
load without error. `verify_dataset()` in this script explicitly asserts
the mask is single-channel (`mask_arr.ndim == 2`) specifically to catch
this class of mistake early if the dataset is ever swapped.

### Real check before trusting the head-swap

`nvidia/mit-b0` has no segmentation decode head at all (it's an
ImageNet-classification-pretrained encoder), so the decode head is
necessarily fresh for this task. Verified via a direct LOAD REPORT that
only `classifier.*` (the discarded ImageNet head) and `decode_head.*` (the
new segmentation head) are missing/unexpected -- the encoder backbone
itself loads with no missing layers, the same diligence applied in
`image_classification.py` and `object_detection.py` before trusting a
head-swap on a new architecture (this bug class doesn't strike every
architecture, but each one needs its own check, not an assumption).

### Real training result

`--max_samples 500 --max_eval_samples 100 --epochs 3`: train loss
2.97 -> 1.86, eval loss 1.93 -> 1.81, `eval_mean_iou` 0.093 -> 0.103,
`eval_pixel_accuracy` 0.756 -> 0.768, all in the expected direction over
~40s of training. Modest mean IoU is expected at this toy scale (500
training images, 3 epochs, the smallest SegFormer variant) -- many of the
18 classes (individual shoes, belt, scarf) are thin and rare per image, so
per-class IoU for those is genuinely hard to move with this little data;
pixel accuracy is higher mostly because "Background" dominates most
images' pixel count. `evaluate.load("mean_iou")` was deliberately not
used -- it fetches a metric script from the Hub, and this project has
already hit `RuntimeError: Dataset scripts are no longer supported` for a
similarly hub-script-dependent load elsewhere (see `object_detection.py`'s
README section on `cppe-5`) -- so mean IoU and pixel accuracy are computed
directly via numpy instead.

### Usage

```bash
python image_segmentation.py --debug_first_batch --max_samples 20
python image_segmentation.py --max_samples 500 --max_eval_samples 100 --epochs 3
```

## `object_detection.py`

Dataset: **`rishitdagli/cppe-5`** -- protective-equipment detection, 5
classes (Coverall, Face_Shield, Gloves, Goggles, Mask), COCO-style
annotations (`bbox` [x, y, w, h], `category`, `area`). Model:
`facebook/detr-resnet-50` (DETR) via `AutoModelForObjectDetection`, COCO's
91-class head reinitialized for 5 classes.

### Real bug found: the unqualified `cppe-5` repo id doesn't load

HF's own object-detection tutorial uses `load_dataset("cppe-5", ...)`, but
that fails under this project's pinned `datasets==5.0.1` with a live
`HfUriError` ("Repository id must be 'namespace/name'") -- the same class
of legacy-non-namespaced-repo issue already documented in
`pretraining/README.md` for a different dataset. Fixed by using
`rishitdagli/cppe-5`, a namespaced mirror verified live to have the
identical schema.

### Real API drift found: `DetrImageProcessor.pad()`'s signature changed

Older HF tutorials batch-pad variable-sized images via
`processor.pad(pixel_values, return_tensors="pt")`. In this project's
pinned `transformers==5.14.1`, that raises `TypeError: DetrImageProcessor.pad()
got an unexpected keyword argument 'return_tensors'` -- confirmed via
`inspect.signature()` that the method's signature changed to operate on a
**single** image (`image, padded_size, annotation=None, ...`), not a
batch. Fixed by reimplementing batch padding directly in the collate
function: find the max height/width across the batch, zero-pad each image
to that size, and build a `pixel_mask` marking real (1) vs. padded (0)
pixels -- the standard DETR batching approach, just no longer provided by
the processor's public API in this version.

### Real check before trusting the head resize: verified the backbone actually loads

`num_labels=5` with `ignore_mismatched_sizes=True` looks structurally
identical to the exact combination that silently dropped the ENTIRE
backbone in `image_classification.py` (see that script's real-bug note).
Before trusting it here, diffed `DetrModel.from_pretrained(...)` (bare
backbone) against `DetrForObjectDetection.from_pretrained(..., num_labels=5,
ignore_mismatched_sizes=True)`'s LOAD REPORTs directly: both show only
`class_labels_classifier.{weight,bias}` MISMATCH-reinitialized (the
classifier head, exactly as intended) and no missing backbone layers --
DETR's checkpoint conversion mapping works correctly here, unlike ViT's.
Worth re-verifying per-architecture rather than assuming this bug class
always strikes.

### Real training result

`--max_samples 200 --max_eval_samples 50 --epochs 5`: train loss
3.98 -> 3.74, eval loss 4.27 -> 4.17, both decreasing and never NaN across
5 epochs (~73s). This script reports the model's own training/eval loss
(classification + bbox L1 + generalized IoU, DETR's standard loss terms)
rather than COCO mAP -- real mAP needs `pycocotools` (not in this
project's `requirements.txt`) and a full prediction/ground-truth matching
pipeline, out of scope for a toy-scale demonstration. Loss decreasing and
staying finite is still a real, meaningful training-correctness signal,
consistent with how `pretraining/` scripts use perplexity rather than
downstream metrics.

### Usage

```bash
python object_detection.py --debug_first_batch --max_samples 20
python object_detection.py --max_samples 200 --max_eval_samples 50 --epochs 5
```
