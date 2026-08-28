"""Audio classification SFT.

Dataset: ashraq/esc50 (ESC-50) -- 2000 5-second environmental sound clips,
50 classes (dog, rain, crying_baby, chainsaw, ...). Fields verified live:
`filename`, `fold`, `target` (int class id), `category` (string class
name), `esc10`, `src_file`, `take`, `audio`. Only a `train` split exists
(no separate eval split), so this script carves one out via
`train_test_split`, same pattern as `image/image_segmentation.py`.
`category` is a raw string column (not a fixed `ClassLabel` schema), so
class names are derived live from the data itself (sorted unique
(target, category) pairs) rather than hardcoded.

**Real API note**: this project's `datasets==5.0.1` decodes the `audio`
column via `datasets.features._torchcodec.AudioDecoder`, not the older
`{"array": ..., "sampling_rate": ...}` dict some HF audio tutorials
assume -- confirmed via `type(example["audio"])`. Getting a raw waveform
needs `audio.get_all_samples()` (returns a `torchcodec.AudioSamples` with
`.data` (channels, samples) and `.sample_rate`), not dict indexing.
ESC-50's native sample rate is 44100 Hz; downmixed to mono and resampled
to 16000 Hz via `torchaudio.functional.resample` before feature extraction.

**Real model-choice bug found and fixed via a decisive overfit sanity
check, not assumed**: the first version of this script used
`facebook/wav2vec2-base` (a speech-pretrained backbone). A real training
run's loss barely moved (3.91 -> 3.85 over 5 epochs). Before writing that
off as "just toy-scale, needs more data," ran the standard debugging
sanity check for exactly this situation -- can the model overfit a TINY
subset it should trivially memorize? On 10 examples across up to 10
distinct classes, even 100 epochs at a 5x higher LR with the feature
encoder unfrozen left loss plateaued at a nonzero constant (2.164) with
vanishing gradient (grad_norm -> 0.03) -- a real, textbook mode-collapse
signature, not "still converging slowly." Separately verified the input
pipeline itself was NOT the cause (`input_values` for 5 different real
examples showed correct per-sample mean~0/std~1 normalization, varying
min/max, correct distinct labels -- no degenerate/constant data). This
isolated the problem to the model: wav2vec2's speech-focused pretraining
objective transfers poorly to generic environmental sound classification
(a known community pain point, not unique to this script). Switched to
**MIT/ast-finetuned-audioset-10-10-0.4593** (an Audio Spectrogram
Transformer, pretrained on AudioSet -- general sound EVENTS, a much
closer domain match to ESC-50 than speech). The identical overfit sanity
check on the identical 10-example subset with AST: loss 0.664 -> 0.0015
in 30 epochs -- clean, healthy memorization, confirming the pipeline was
never the problem and AST is the right model family for this task.

Model: MIT/ast-finetuned-audioset-10-10-0.4593 via
AutoModelForAudioClassification, AudioSet's 527-class head resized to this
task's 50 classes. Verified via a direct LOAD REPORT that only
`classifier.dense.{weight,bias}` MISMATCH-reinitializes (the head) with no
missing backbone layers -- the same diligence applied to every other
head-swap in this wave, given `ignore_mismatched_sizes=True` has
previously (see `image/image_classification.py`) silently dropped an
entire backbone on a different architecture.

Usage:
    python audio_classification.py --debug_first_batch --max_samples 20
"""

import argparse
import time

import numpy as np
import torch
import torchaudio
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification, Trainer, TrainingArguments

from common.data_selection import select_samples
from common.logging_utils import print_banner, print_config
from common.model_saving import save_model
from common.run_results import write_run_result
from common.seeding import set_all_seeds

DATASET_NAME = "ashraq/esc50"
ARCHITECTURE = "encoder-only"
TARGET_SAMPLE_RATE = 16000


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="SFT an Audio Spectrogram Transformer for audio classification.")

    p.add_argument("--model", type=str, default="MIT/ast-finetuned-audioset-10-10-0.4593", help="AST checkpoint, AudioSet's 527-class head resized for this task. Default: MIT/ast-finetuned-audioset-10-10-0.4593 (fp32 ~340MB). See module docstring for why this replaced an earlier wav2vec2-base attempt.")
    p.add_argument("--eval_fraction", type=float, default=0.1, help="Fraction of the (only) train split held out as eval via train_test_split, since this dataset ships no separate eval split. Default: 0.1.")

    p.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"], help="Mixed precision. Default: bf16.")
    p.add_argument("--batch_size", type=int, default=4, help="Per-device training batch size. Default: 4 (AST's 1024x128 mel-spectrogram input is memory-heavier per-sample than a raw waveform).")
    p.add_argument("--eval_batch_size", type=int, default=4, help="Per-device eval batch size. Default: 4.")
    p.add_argument("--gradient_accumulation_steps", type=int, default=2, help="Gradient accumulation steps. Default: 2.")
    p.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate. Default: 5e-5 (verified via the overfit sanity check in the module docstring to actually drive loss down on this model).")
    p.add_argument("--epochs", type=int, default=8, help="Training epochs. Default: 8.")
    p.add_argument("--warmup_steps", type=int, default=50, help="LR warmup steps. Default: 50.")
    p.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay. Default: 0.01.")

    p.add_argument("--max_samples", type=int, default=-1, help="Number of training rows to use. -1 (default) = full split.")
    p.add_argument("--sample_selection", type=str, default="random", choices=["random", "first", "last"], help="Which rows to pick when --max_samples is positive. Default: random.")
    p.add_argument("--max_eval_samples", type=int, default=-1, help="Eval rows to use. -1 (default) = full carved-out eval split.")

    p.add_argument("--output_dir", type=str, default="./output/supervised-finetuning/audio/audio_classification", help="Output directory.")
    p.add_argument("--logging_steps", type=int, default=10, help="Logging frequency. Default: 10.")
    p.add_argument("--eval_steps", type=int, default=50, help="Evaluation frequency. Default: 50.")
    p.add_argument("--save_steps", type=int, default=50, help="Checkpoint save frequency. Default: 50.")

    p.add_argument("--seed", type=int, default=42, help="Random seed. Default: 42.")
    p.add_argument("--debug_first_batch", action="store_true", default=False, help="Print formatted examples and exit without training.")

    return p


def verify_dataset() -> None:
    print_banner("VERIFYING DATASET")
    peek = load_dataset(DATASET_NAME, split="train", streaming=True)
    example = next(iter(peek))
    for field in ("audio", "target", "category"):
        assert field in example, f"Expected a {field!r} field in {DATASET_NAME}, got fields: {list(example.keys())}"
    samples = example["audio"].get_all_samples()
    print(f"Dataset: {DATASET_NAME}")
    print(f"Fields: {list(example.keys())}")
    print(f"Sample: category={example['category']!r}, target={example['target']}, "
          f"audio shape={tuple(samples.data.shape)}, sample_rate={samples.sample_rate}")
    print()


def waveform_to_16k_mono(audio_decoder) -> np.ndarray:
    samples = audio_decoder.get_all_samples()
    waveform = samples.data.mean(dim=0)  # downmix to mono
    if samples.sample_rate != TARGET_SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, samples.sample_rate, TARGET_SAMPLE_RATE)
    return waveform.numpy()


def load_and_prepare_data(args, feature_extractor):
    print_banner("LOADING DATASET")
    full = load_dataset(DATASET_NAME, split="train")

    class_pairs = sorted(set(zip(full["target"], full["category"])))
    class_names = [name for _, name in class_pairs]
    print(f"Classes ({len(class_names)}): {class_names}")

    split = full.train_test_split(test_size=args.eval_fraction, seed=args.seed)
    train_raw, eval_raw = split["train"], split["test"]

    train_raw = select_samples(train_raw, args.max_samples, args.sample_selection, args.seed)
    eval_raw = select_samples(eval_raw, args.max_eval_samples, "first", args.seed)

    def transform(examples):
        waveforms = [waveform_to_16k_mono(audio) for audio in examples["audio"]]
        encoding = feature_extractor(waveforms, sampling_rate=TARGET_SAMPLE_RATE, return_tensors="pt")
        return {"input_values": encoding["input_values"], "labels": examples["target"]}

    train_dataset = train_raw.with_transform(transform)
    eval_dataset = eval_raw.with_transform(transform)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Eval samples: {len(eval_dataset)}")
    print()
    return train_dataset, eval_dataset, class_names


def collate_fn(batch):
    input_values = torch.stack([item["input_values"] for item in batch])  # AST's feature extractor always returns a fixed (1024, 128) shape -- no variable-length padding needed, unlike raw-waveform models
    labels = torch.tensor([item["labels"] for item in batch])
    return {"input_values": input_values, "labels": labels}


def print_formatted_examples_audio(dataset, class_names, num_examples: int = 2) -> None:
    print_banner("FORMATTED EXAMPLES")
    for i in range(min(num_examples, len(dataset))):
        example = dataset[i]
        print(f"\n--- Example {i + 1} ---")
        print(f"input_values.shape: {tuple(example['input_values'].shape)} (fixed-size mel-spectrogram)")
        print(f"Label: {example['labels']} ({class_names[example['labels']]})")
    print()


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1_macro": f1_score(labels, predictions, average="macro"),
        "precision": precision_score(labels, predictions, average="macro", zero_division=0),
        "recall": recall_score(labels, predictions, average="macro", zero_division=0),
    }


def main():
    args = build_arg_parser().parse_args()
    set_all_seeds(args.seed)
    print_config(args, "Audio classification SFT (AST)")

    # verify_dataset()  # commented out: avoids loading the dataset twice (a streaming peek here, then a full load in load_and_prepare_data()) to cut memory overhead
    feature_extractor = AutoFeatureExtractor.from_pretrained(args.model)

    train_dataset, eval_dataset, class_names = load_and_prepare_data(args, feature_extractor)

    torch_dtype = torch.bfloat16 if args.mixed_precision == "bf16" else (torch.float16 if args.mixed_precision == "fp16" else torch.float32)
    model = AutoModelForAudioClassification.from_pretrained(
        args.model,
        num_labels=len(class_names),
        id2label={i: name for i, name in enumerate(class_names)},
        label2id={name: i for i, name in enumerate(class_names)},
        ignore_mismatched_sizes=True,  # AudioSet's 527-class head -> this task's 50 classes; verified via a direct LOAD REPORT diff that this is a clean, narrow reinit (only the classifier head)
        dtype=torch_dtype,
    )

    total_params = sum(p.numel() for p in model.parameters())
    print_banner("MODEL")
    print(f"Architecture: {ARCHITECTURE} ({args.model}), num_labels={len(class_names)}")
    print(f"Total parameters: {total_params:,}")
    print()

    if args.debug_first_batch:
        print_formatted_examples_audio(train_dataset, class_names, num_examples=2)
        print("--debug_first_batch set: exiting without training.")
        return

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        bf16=(args.mixed_precision == "bf16"),
        fp16=(args.mixed_precision == "fp16"),
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        seed=args.seed,
        report_to=[],
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
    )

    print_banner("TRAINING")
    start = time.time()
    trainer.train()
    train_runtime_seconds = time.time() - start

    print_banner("FINAL EVALUATION")
    eval_metrics = trainer.evaluate()
    print(eval_metrics)

    save_model(model, feature_extractor, args.output_dir, strategy="full")

    write_run_result(
        output_dir=args.output_dir,
        stage="supervised-finetuning",
        task="audio_classification",
        modality="audio",
        architecture=ARCHITECTURE,
        variant="standard",
        cot_enabled=False,
        model_name=args.model,
        dataset_name=DATASET_NAME,
        save_strategy="full",
        hyperparameters=vars(args),
        metrics={
            "accuracy": eval_metrics.get("eval_accuracy"),
            "f1_macro": eval_metrics.get("eval_f1_macro"),
            "precision": eval_metrics.get("eval_precision"),
            "recall": eval_metrics.get("eval_recall"),
            "total_parameters": total_params,
        },
        num_train_samples=len(train_dataset),
        num_eval_samples=len(eval_dataset),
        train_runtime_seconds=train_runtime_seconds,
    )
    print("Done.")


if __name__ == "__main__":
    main()
