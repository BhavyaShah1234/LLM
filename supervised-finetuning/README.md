# Supervised Finetuning

Status: **done (Wave 2)** for the original 18 text/VQA scripts (16
rewrites + 2 new architecture-family scripts), built and smoke-tested
(`--debug_first_batch`) against live dataset schemas rather than the
archived scripts' assumptions. That verification step caught real,
previously-unknown bugs in the *original* Feb-2026 scripts -- not just the
fake-news field-name issue found during planning, but several more (see
"Real bugs found and fixed" below). 17 of 18 scripts additionally passed a
short real training run; the 18th (`image/vqa_no_cot.py`) is verified via
`--debug_first_batch` with the same fixes applied, pending a full
training-run confirmation of the last VL model download.

**Wave 5 added all 6 remaining modalities this project originally
scoped** (image classification/detection/segmentation, audio, video,
tabular, graph, time-series -- 8 image/non-text scripts total), every one
trained for real, not just smoke-tested. See "Wave 5: non-text modalities"
below for the summary and each modality folder's own README
(`image/README.md`, `audio/README.md`, `video/README.md`,
`tabular/README.md`, `graph/README.md`, `time-series/README.md`) for full
detail -- this wave surfaced several genuinely new bug classes beyond the
dataset-field-name pattern Wave 2 found: a `transformers` library bug that
silently drops an entire pretrained backbone on one head-swap combination
but not others (caught by diffing LOAD REPORTs on every new script, not
assumed safe), an image processor API whose batch-padding method changed
signature, a checkpoint using an attention-bias naming scheme this
project's pinned `transformers` version doesn't remap, and one case where
a plausible-looking model choice (wav2vec2 for generic audio
classification) turned out to be a genuine architectural mismatch, caught
only by running the standard "can it overfit a tiny subset" sanity check
rather than assuming more data would fix a stuck loss curve.

## Scope

Rewrites the repo's original 16 standalone SFT scripts (archived at
`old/*.py`) into the new structure -- plus 2 new scripts that add an
encoder-only and an encoder-decoder variant of text classification, so this
stage also anchors `experiments/architecture-family-classification/`.

Every script here is monolithic and standalone (task-specific data loading,
prompt formatting, loss masking, and evaluation logic all stay inline, per
this project's conventions) but shares plumbing via `common/`
(`model_loading`, `quantization`, `peft_setup`, `model_saving`, `seeding`,
`logging_utils`, `run_results`, `data_selection`).

## Model selection

Per the root README's model-selection philosophy: default to a **base**
(non-instruction-tuned) checkpoint whose **unquantized** weights already fit
comfortably in this project's 8GB VRAM budget, since prompt-masked SFT works
fine starting from base weights for every task below -- none of them have
RLHF's hard instruction-tuned-prerequisite.

- **Text tasks**: `Qwen/Qwen3-1.7B-Base` (fp16 ≈ 3.4GB, confirmed via HF Hub file sizes). Replaces the original scripts' `Qwen/Qwen3-4B` default, which needed 4-bit quantization just to load.
- **VQA**: `Qwen/Qwen2-VL-2B` (base, non-instruct; fp16 ≈ 4.4GB, confirmed).
- **Encoder-only classification**: `answerdotai/ModernBERT-base` (149.6M params, fp16 ≈ 1.2GB) -- chosen over `microsoft/deberta-v3-base` as the newer (2024) architecture, per this project's prefer-newer-open-source-models note.
- **Encoder-decoder classification**: raw pretrained `t5-base` (222.9M params) -- **not** `google/flan-t5-base`, since FLAN-T5 is already instruction-tuned, which would violate the prefer-base rule.

## Real bugs found and fixed

Verifying every dataset's actual live schema (rather than trusting the
archived scripts' field-access code) surfaced real bugs beyond the one
originally flagged during Wave 1 planning:

- **`text_classification_{no_cot,cot}.py`** (`domofon/fake_news_cot_reasoning`, fields `title/input/reasoning/status`): the archived `cot.py` read `text`/`label` as primary fields -- neither exists in this dataset -- while `no_cot.py` correctly used `input`/`status`. The two scripts were extracting different (and one of them wrong) data from the same dataset.
- **`ner_standard.py`** (`MorryShah/complex_ner`): the archived script expected CoNLL-style `tokens`/`ner_tags` fields. The real schema is `text`/`entities` (already `{"text", "type"}` dicts) -- `tokens`/`ner_tags` don't exist at all, so every training example was built from an empty token list.
- **`mcq_standard.py`** (`araag2/MedMCQA`, config `processed`): the archived script read lowercase `question`/`opa`.../`cop` fields. The real schema uses capitalized `Question`/`Option_A`... and the answer field is `Label` (already a letter), not a numeric `cop`/`answer` index -- every field access was wrong. A second, separate bug found later (while building `rlhf/grpo/grpo.py`, which reuses this dataset): both the archived script and this project's own rewrite originally evaluated against the `test` split, whose `Label` field is `None` for every row (labels withheld, standard practice to prevent leaderboard cheating) -- every accuracy comparison would silently be wrong regardless of model quality. Fixed by evaluating against `dev` instead, in both `mcq_standard.py` and `grpo.py`.
- **`mcq_{no_cot,cot}.py`** (`HPAI-BSC/medmcqa-cot-llama31`): the archived scripts read a nonexistent `options` field; the real `question` field is already self-contained (question text + "Options: A. ... D." embedded), so no separate field exists or is needed.
- **`instruction_tuning_{no_cot,cot}.py`** (`domofon/evol-instruct-code-cot-80k`): both read a nonexistent `output` field for the target text; the real field is `response`.
- **`chat_tuning_standard.py`** (`devrev-research/MathChatSync-reasoning`): the archived script expected flat `user`/`assistant` fields. The real (and only) field is `conversations`, a ShareGPT-style list -- the flat fields never existed, so every training example's prompt/response was empty.
- **`vqa_{no_cot,cot}.py`** (`opendatalab/ChartVerse-SFT-1.8M`): three real bugs. (1) the image field is `images` (a **list**), not `image`/`chart` -- every example silently fell back to a blank placeholder image. (2) neither script ever constructed a `labels` field at all -- training received no loss signal whatsoever, not a subtler accuracy issue. (3) `vqa_cot.py` looked for a `reasoning` field; the real field is `cot_solution` (which does exist and contains genuine chart-reasoning text, resolving the "may not have an explicit reasoning column" concern noted during planning) -- but `cot_solution`'s own text already includes a `<think>` wrapper, which this project's rewrite also wraps in `<think>...</think>`, so it must be stripped first or the tag ends up duplicated (`<think><think>...`), caught via `--debug_first_batch` during smoke testing.
- Several archived scripts (`instruction_tuning_cot.py`, `chat_tuning_{standard,cot}.py`, `vqa_{no_cot,cot}.py`) built prompt strings with a literal `\\n` (the two characters `\` and `n`) instead of a real newline escape, confirmed via `cat -A` on the source files.
- **`instruction_tuning_standard.py`** (`ibivibiv/math_instruct`): not a bug, but a scale surprise found while smoke testing -- this dataset has **114M rows / ~10GB**, discovered only once a naive `load_dataset(..., split="train")` call hung. Fixed by streaming with a capped materialization (`MAX_RAW_EXAMPLES`), the same pattern already used for the large/streamed chat and VQA datasets.
- **`vqa_{no_cot,cot}.py`**, separately: materializing `MAX_RAW_EXAMPLES=5000` full-resolution chart images (some over 2700px wide) into memory OOM-killed the process during smoke testing (RSS grew past 27GB). Fixed by resizing each image to `MAX_IMAGE_SIDE=672` immediately on ingest and reducing the cap to 800.

Every fix above is also called out in its script's module docstring.

**One more thing confirmed empirically, not a bug but worth flagging**: the
root README's "unquantized base fits in 8GB" rule is about *loading* the
base model, not *training* it. Full-parameter finetuning of
`Qwen/Qwen3-1.7B-Base` (gradients + Adam optimizer states on top of the
weights) does **not** fit in 8GB -- confirmed via a real `CUDA out of
memory` error running `text_classification_standard.py` without `--lora`.
`--lora` (optionally with `--quantization 4bit`) is effectively required for
real training runs on this project's hardware for every decoder-only script,
even though the base loads fine unquantized for inference/`--debug_first_batch`.

## Wave 5: non-text modalities

All 7 new scripts trained for real (not just smoke-tested), each starting
from a verified-live dataset (several plausible-looking datasets were
ruled out on inspection -- see each README for specifics) and, where a
pretrained checkpoint exists, verified via a direct LOAD REPORT diff
before trusting a head-swap.

| Script | Dataset | Model | Real result | Real bug found |
|---|---|---|---|---|
| `image/image_classification.py` | `uoft-cs/cifar10` | `google/vit-base-patch16-224-in21k` | 80.6% accuracy | `AutoModelForImageClassification` silently drops the ENTIRE pretrained backbone on this checkpoint (a `transformers` library bug, not this project's code) -- fixed by loading the backbone separately and copying its state dict in |
| `image/object_detection.py` | `rishitdagli/cppe-5` | `facebook/detr-resnet-50` | train/eval loss decreasing, never NaN over 5 epochs | unqualified `cppe-5` repo id isn't loadable under this project's `datasets` version (non-namespaced legacy repo); `DetrImageProcessor.pad()`'s batch-padding signature changed and needed reimplementing manually |
| `image/image_segmentation.py` | `mattmdjaga/human_parsing_dataset` | `nvidia/mit-b0` (SegFormer) | 0.103 mean IoU, 0.768 pixel accuracy | avoided a different dataset (`nateraw/pascal-voc-2012`) whose masks encode class via RGB *color*, not index -- would have silently trained on nonsense labels |
| `audio/audio_classification.py` | `ashraq/esc50` | `MIT/ast-finetuned-audioset-10-10-0.4593` | 96.0% accuracy | first model choice (`facebook/wav2vec2-base`, speech-pretrained) couldn't even overfit a 10-example subset -- a real architectural mismatch, not a bug in this project's pipeline, caught via a decisive overfit sanity check before switching models |
| `video/video_classification.py` | `nateraw/kinetics-mini` | `MCG-NJU/videomae-base` | train loss 1.62 -> 0.84 (eval near chance, expected at 10 clips/class) | checkpoint's attention biases (`q_bias`/`v_bias`) don't match this `transformers` version's expected key names even on the BARE backbone -- fixed by copying the raw checkpoint's bias tensors in manually |
| `tabular/tabular_classification.py` | `scikit-learn/adult-census-income` | from-scratch entity-embedding MLP | 86.1% accuracy, 91.3% AUROC (matches published baselines) | -- |
| `graph/graph_classification.py` | `torch_geometric.datasets.TUDataset("MUTAG")` | from-scratch GIN | 82.1% accuracy, 90.1% AUROC (matches published GIN benchmarks) | -- |
| `time-series/time_series_classification.py` | `mineshj1291/ecg-classification` | from-scratch FCN (1D-CNN) | 67.7% accuracy vs. 33.3% chance | two other candidate datasets ruled out live: dataset-script-based loading no longer supported, and a malformed CSV upload where timesteps became column NAMES |

Tabular, graph, and time-series train from scratch (no HF `Trainer`, no
`save_model()`) -- there's no transferable pretrained-checkpoint concept
for arbitrary tabular schemas, graph-classification tasks, or signal
schemas the way there is for images/audio/video/text, the same "genuinely
nothing to adapt" situation as `pretraining/`.

## Script matrix (18 scripts)

| # | Path | Architecture | Dataset | Model | Verified |
|---|---|---|---|---|---|
| 1 | `text/classification/decoder-only/text_classification_standard.py` | decoder-only | `Tohrumi/glue_sst2_10k` | `Qwen/Qwen3-1.7B-Base` | debug + short training run |
| 2 | `text/classification/decoder-only/text_classification_no_cot.py` | decoder-only | `domofon/fake_news_cot_reasoning` | `Qwen/Qwen3-1.7B-Base` | debug (field-name bug fixed) |
| 3 | `text/classification/decoder-only/text_classification_cot.py` | decoder-only | `domofon/fake_news_cot_reasoning` | `Qwen/Qwen3-1.7B-Base` | debug (field-name bug fixed) |
| 17 | `text/classification/encoder-only/text_classification_standard.py` | encoder-only | `Tohrumi/glue_sst2_10k` | `answerdotai/ModernBERT-base` | debug (149.6M params confirmed) |
| 18 | `text/classification/encoder-decoder/text_classification_standard.py` | encoder-decoder | `Tohrumi/glue_sst2_10k` | `t5-base` | debug (222.9M params confirmed) |
| 4 | `text/ner/decoder-only/ner_standard.py` | decoder-only | `MorryShah/complex_ner` | `Qwen/Qwen3-1.7B-Base` | debug (schema bug fixed) |
| 5 | `text/ner/decoder-only/ner_no_cot.py` | decoder-only | `zilliz/natural_questions-context-relevance-with-think` | `Qwen/Qwen3-1.7B-Base` | debug (span extraction corrected) |
| 6 | `text/ner/decoder-only/ner_cot.py` | decoder-only | same dataset as #5 | `Qwen/Qwen3-1.7B-Base` | debug |
| 7 | `text/mcq/decoder-only/mcq_standard.py` | decoder-only | `araag2/MedMCQA` (config `processed`) | `Qwen/Qwen3-1.7B-Base` | debug (field-name bug fixed) |
| 8 | `text/mcq/decoder-only/mcq_no_cot.py` | decoder-only | `HPAI-BSC/medmcqa-cot-llama31` | `Qwen/Qwen3-1.7B-Base` | debug (options-field bug fixed) |
| 9 | `text/mcq/decoder-only/mcq_cot.py` | decoder-only | same dataset as #8 | `Qwen/Qwen3-1.7B-Base` | debug |
| 10 | `text/instruction-tuning/decoder-only/instruction_tuning_standard.py` | decoder-only | `ibivibiv/math_instruct` | `Qwen/Qwen3-1.7B-Base` | debug (streaming fix for 114M-row dataset) |
| 11 | `text/instruction-tuning/decoder-only/instruction_tuning_no_cot.py` | decoder-only | `domofon/evol-instruct-code-cot-80k` | `Qwen/Qwen3-1.7B-Base` | debug (field-name bug fixed) |
| 12 | `text/instruction-tuning/decoder-only/instruction_tuning_cot.py` | decoder-only | same dataset as #11 | `Qwen/Qwen3-1.7B-Base` | debug (field-name + `\n` bug fixed) |
| 13 | `text/chat-tuning/decoder-only/chat_tuning_standard.py` | decoder-only | `devrev-research/MathChatSync-reasoning` | `Qwen/Qwen3-1.7B-Base` | debug (ShareGPT schema bug fixed) |
| 14 | `text/chat-tuning/decoder-only/chat_tuning_cot.py` | decoder-only | `PJMixers-Dev/oumi-ai_lmsys_chat_1m_clean_R1-1k-think-1k-response-ShareGPT` | `Qwen/Qwen3-1.7B-Base` | debug |
| 15 | `image/vqa_no_cot.py` | decoder-only | `opendatalab/ChartVerse-SFT-1.8M` (streamed) | `Qwen/Qwen2-VL-2B` (base) | debug (image/labels bugs fixed, OOM fixed) |
| 16 | `image/vqa_cot.py` | decoder-only | same dataset as #15 | `Qwen/Qwen2-VL-2B` (base) | debug (image/labels/reasoning-field bugs fixed, OOM fixed) |

Rows 15-16 live flat under `image/` (no architecture-family subfolder layer,
unlike `text/`) alongside the planned `image_classification.py`,
`object_detection.py`, `image_segmentation.py` -- a deliberate divergence
between `text/`'s nested convention and `image/`/`audio/`/`video/`'s flat
convention, per the root README.

## Architecture-specific implementation notes

- **Encoder-only** (`text_classification_standard.py` under `encoder-only/`): no prompt template, no loss masking, no generation-based eval -- a classification head trained with plain cross-entropy via `Trainer.compute_metrics`. No CoT variant is possible (a classification head can't emit a reasoning trace). No `--quantization`/`--lora` flags -- the base model is small enough that full-parameter finetuning is standard practice.
- **Encoder-decoder** (`text_classification_standard.py` under `encoder-decoder/`): text-to-text formulation (`"classify sentiment: {text}" -> "{label}"`) trained via `Seq2SeqTrainer` + `DataCollatorForSeq2Seq`, with `predict_with_generate=True` for real generation-based eval.
- **VQA** (`image/vqa_{no_cot,cot}.py`): the prompt text includes the processor's own image-token placeholder (`processor.image_token`, e.g. `<|image_pad|>`) directly, rather than relying on `apply_chat_template` (the base, non-Instruct checkpoint this project defaults to often has no chat template). Because Qwen2-VL uses dynamic image resolution (`image_grid_thw` varies per example), a small custom collator concatenates `pixel_values`/`image_grid_thw` across a batch (matching how Qwen2-VL expects multi-image batches) rather than using a generic padding collator.

## Prompt templates (ported from `old/README_v1.md`, still accurate)

**Instruction-tuning format**:
```
### Instruction:
{system_instruction}

### Input:
{input_text}

### Response:
{output}
```

**Chat format** (Vicuna-style fallback when a model has no chat template):
```
SYSTEM: {system_message}
USER: {user_message}
ASSISTANT: {assistant_response}
```

**VQA format**:
```
{image_token}
### Instruction:
Answer the question about the image.

### Input:
{question}

### Response:
{answer}
```

## Loss masking rules (ported from `old/README_v1.md`, still accurate)

| Task | Loss computed on | Masked |
|---|---|---|
| Text classification (decoder-only) | Label tokens only | Instruction + Input |
| Text classification (encoder-only) | N/A -- plain classification cross-entropy, no token-level masking | N/A |
| Text classification (encoder-decoder) | Full (short) target sequence | N/A -- seq2seq loss, no masking needed |
| NER | Entity JSON only | Instruction + Input |
| MCQ | Answer letter only | Instruction + Input + Options |
| Instruction tuning (no CoT) | Output only | Instruction + Input |
| Instruction tuning (CoT) | `<think>` content + Output | Instruction + Input |
| Chat | Assistant messages only | System + User messages |
| Chat (CoT) | Assistant (incl. `<think>`) | System + User messages |
| VQA (no CoT) | Answer only | Instruction + Input + image tokens |
| VQA (CoT) | `<think>` + Answer | Instruction + Input + image tokens |

Implementation (decoder-only/encoder-decoder scripts): set `labels[i] = -100`
at every masked token position before passing to the model -- HF's
cross-entropy loss ignores `-100` positions.

## Evaluation metrics

- **Classification (decoder-only, encoder-only) / MCQ**: accuracy, macro F1, weighted F1, precision, recall, AUROC (binary only).
- **Classification (encoder-decoder)**: same, computed from generated text decoded back to labels.
- **NER**: entity-level F1/precision/recall (exact `(entity_text, label)` match).
- **Generation (instruction tuning, chat, VQA)**: ROUGE-L (via `rouge_score`) and BERTScore F1 (via `bert_score`); VQA additionally reports exact-match accuracy.
- **CoT-specific**: CoT usage rate (% of predictions containing `<think>` tags), average CoT length (words), CoT-enabled vs. CoT-disabled accuracy/ROUGE-L delta on the same dataset -- this is what `experiments/cot-vs-no-cot-classification/` reads.

## Shared CLI conventions

Same as every other stage: `--max_samples` (default `-1` = full split),
`--sample_selection {random,first,last}` (default `random`), `--seed`,
`--output_dir`, `--debug_first_batch`. Unlike `pretraining/`, decoder-only
and encoder-decoder scripts *do* expose `--quantization {no,4bit,8bit}`,
`--lora*`, `--mixed_precision`, and `--save_strategy
{adapter_only,merged,adapter_and_merged,base_reference}` (via
`common/quantization.py`, `common/peft_setup.py`, `common/model_saving.py`)
since they load a pretrained base and typically LoRA-adapt it. The
encoder-only script omits `--quantization`/`--lora` (see "Architecture-specific
implementation notes" above).

Datasets far larger than this project's typical few-hundred-MB SFT corpora
(`ibivibiv/math_instruct` at 114M rows, `opendatalab/ChartVerse-SFT-1.8M`,
`PJMixers-Dev/...-ShareGPT`) are streamed with a `MAX_RAW_EXAMPLES` cap
rather than fully downloaded, found necessary during smoke testing (see
"Real bugs found and fixed").

## Dataset Reference (this folder's slice of the root README's master table)

| Script(s) | Dataset | Role |
|---|---|---|
| `text_classification_standard.py` (all 3 architectures) | `Tohrumi/glue_sst2_10k` | primary training + eval corpus |
| `text_classification_{no_cot,cot}.py` | `domofon/fake_news_cot_reasoning` | primary training + eval corpus |
| `ner_standard.py` | `MorryShah/complex_ner` | primary training + eval corpus |
| `ner_{no_cot,cot}.py` | `zilliz/natural_questions-context-relevance-with-think` | primary training + eval corpus |
| `mcq_standard.py` | `araag2/MedMCQA` (config `processed`) | primary training + eval corpus |
| `mcq_{no_cot,cot}.py` | `HPAI-BSC/medmcqa-cot-llama31` | primary training + eval corpus |
| `instruction_tuning_standard.py` | `ibivibiv/math_instruct` (streamed, capped) | primary training + eval corpus |
| `instruction_tuning_{no_cot,cot}.py` | `domofon/evol-instruct-code-cot-80k` | primary training + eval corpus |
| `chat_tuning_standard.py` | `devrev-research/MathChatSync-reasoning` | primary training + eval corpus |
| `chat_tuning_cot.py` | `PJMixers-Dev/oumi-ai_lmsys_chat_1m_clean_R1-1k-think-1k-response-ShareGPT` (streamed, capped) | primary training + eval corpus |
| `vqa_{no_cot,cot}.py` | `opendatalab/ChartVerse-SFT-1.8M` (streamed, capped, images resized on ingest) | primary training + eval corpus |
| `image_classification.py` (Wave 5) | `uoft-cs/cifar10` | primary training + eval corpus |
| `object_detection.py` (Wave 5) | `rishitdagli/cppe-5` | primary training + eval corpus |
| `image_segmentation.py` (Wave 5) | `mattmdjaga/human_parsing_dataset` | primary training + eval corpus |
| `audio_classification.py` (Wave 5) | `ashraq/esc50` | primary training + eval corpus |
| `video_classification.py` (Wave 5) | `nateraw/kinetics-mini` | primary training + eval corpus |
| `tabular_classification.py` (Wave 5) | `scikit-learn/adult-census-income` | primary training + eval corpus |
| `graph_classification.py` (Wave 5) | `torch_geometric.datasets.TUDataset("MUTAG")` | primary training + eval corpus |
| `time_series_classification.py` (Wave 5) | `mineshj1291/ecg-classification` | primary training + eval corpus |

To swap any dataset, edit the named script's dataset-loading constant and
re-run its `--debug_first_batch` verification step to confirm the new
schema's fields match what the script's formatting/masking logic expects.
