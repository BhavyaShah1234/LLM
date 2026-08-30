"""Visual Question Answering SFT -- decoder-only (vision-language), no CoT.

Dataset: opendatalab/ChartVerse-SFT-1.8M (streamed). Fields verified live:
id, images, code, question, answer, code_solution, cot_solution. This fixes
THREE real bugs vs. the archived version of this script
(old/vqa_no_cot.py): (1) it read a nonexistent `image`/`chart` field for the
image (the real field is `images`, a LIST of PIL images) -- every example
therefore silently fell back to a blank white placeholder image, so training
never actually saw a real chart; (2) it never constructed a `labels` field
at all, so the model received no loss signal during training whatsoever --
not a subtler accuracy bug, training was a no-op; (3) its prompt string was
built with a literal `\\n` instead of a real newline. All three are fixed
here. See supervised-finetuning/README.md.

Image placeholder handling: the prompt text includes the processor's own
image token placeholder (`processor.image_token`, e.g. `<|image_pad|>`)
directly rather than relying on `apply_chat_template` -- the base (non-Instruct)
checkpoint this project defaults to often has no chat template, and the
image-token-in-plain-text approach works with `Qwen2VLProcessor` regardless.

Format: <image token>\\n### Instruction: ... ### Input: {question} ### Response: {answer}
Loss: answer tokens only (computed via a two-pass tokenization: the prompt
alone, then prompt+answer, both with the same image -- the image-token
expansion is then identical in both, so the length difference isolates
exactly the answer span, the same masking approach used by every text-only
script in this project, generalized to include the image).

Because Qwen2-VL uses dynamic image resolution (`image_grid_thw` varies per
example), examples are NOT batch-padded by a generic collator -- a small
custom collator concatenates `pixel_values`/`image_grid_thw` (which is how
Qwen2-VL expects multi-image batches) and pads `input_ids`/`labels`
separately. Default `--batch_size 1` is recommended; larger batch sizes work
too (the custom collator handles padding across examples) but increase peak
memory since chart images can be high resolution.

Usage:
    python vqa_no_cot.py --debug_first_batch --max_samples 20
"""

import argparse
import time

import torch
from datasets import load_dataset
from PIL import Image
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset
from transformers import AutoProcessor, Trainer, TrainingArguments

from common.data_selection import select_samples
from common.logging_utils import print_banner, print_config
from common.model_loading import load_vision_language_model
from common.model_saving import save_model
from common.run_results import write_run_result
from common.seeding import set_all_seeds

DATASET_NAME = "opendatalab/ChartVerse-SFT-1.8M"
ARCHITECTURE = "decoder-only"
INSTRUCTION = "Answer the question about the image."
MAX_RAW_EXAMPLES = 800  # dataset is streamed and huge; cap what gets materialized
MAX_IMAGE_SIDE = 672  # chart images can be very high-res (e.g. 2780x1577) -- found via an
# OOM crash while smoke-testing (materializing MAX_RAW_EXAMPLES=5000 full-resolution images
# grew RSS past 27GB before the kernel OOM-killed the process); resizing on ingest keeps
# each image's memory footprint bounded regardless of source resolution.


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for this script.

    Returns:
        argparse.ArgumentParser: Parser covering model/quantization/LoRA,
        optimization, data-selection, output, and debug flags.
    """
    p = argparse.ArgumentParser(description="SFT a vision-language decoder-only model for VQA (no CoT).")

    p.add_argument("--model", type=str, default="Qwen/Qwen2-VL-2B", help="Base VL checkpoint (non-instruct). Default: Qwen/Qwen2-VL-2B (fp16 ~4.4GB).")
    p.add_argument("--quantization", type=str, default="no", choices=["no", "4bit", "8bit"], help="Quantization for loading. Default: no.")

    p.add_argument("--lora", action="store_true", default=False, help="Enable LoRA finetuning.")
    p.add_argument("--lora_r", type=int, default=16, help="LoRA rank. Default: 16.")
    p.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha. Default: 32.")
    p.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout. Default: 0.05.")
    p.add_argument("--lora_target_modules", type=str, default="q_proj,k_proj,v_proj,o_proj", help="Comma-separated LoRA target modules.")

    p.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"], help="Mixed precision. Default: bf16.")
    p.add_argument("--batch_size", type=int, default=1, help="Per-device training batch size. Default: 1 (see module docstring on dynamic image resolution).")
    p.add_argument("--eval_batch_size", type=int, default=1, help="Per-device eval batch size. Default: 1.")
    p.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Gradient accumulation steps. Default: 8.")
    p.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate. Default: 1e-4.")
    p.add_argument("--epochs", type=int, default=3, help="Training epochs. Default: 3.")
    p.add_argument("--max_length", type=int, default=1024, help="Max total sequence length (text tokens; image tokens are additional). Default: 1024.")
    p.add_argument("--warmup_steps", type=int, default=100, help="LR warmup steps. Default: 100.")
    p.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay. Default: 0.01.")
    p.add_argument("--gradient_checkpointing", action="store_true", default=True, help="Enable gradient checkpointing. Default: on.")

    p.add_argument("--max_samples", type=int, default=-1, help="Number of training rows to use (from the first MAX_RAW_EXAMPLES streamed). -1 (default) = use all of them.")
    p.add_argument("--sample_selection", type=str, default="random", choices=["random", "first", "last"], help="Which rows to pick when --max_samples is positive. Default: random.")
    p.add_argument("--max_eval_samples", type=int, default=50, help="Eval rows to use (VL generation is slow). Default: 50.")

    p.add_argument("--output_dir", type=str, default="./output/supervised-finetuning/image/vqa_no_cot", help="Output directory.")
    p.add_argument("--logging_steps", type=int, default=10, help="Logging frequency. Default: 10.")
    p.add_argument("--eval_steps", type=int, default=200, help="Evaluation frequency. Default: 200.")
    p.add_argument("--save_steps", type=int, default=200, help="Checkpoint save frequency. Default: 200.")
    p.add_argument("--save_strategy", type=str, default="adapter_only", choices=["full", "adapter_only", "merged", "adapter_and_merged", "base_reference"], help="Model saving strategy. Default: adapter_only when --lora, otherwise 'full'.")

    p.add_argument("--seed", type=int, default=42, help="Random seed. Default: 42.")
    p.add_argument("--debug_first_batch", action="store_true", default=False, help="Print formatted examples and exit without training.")

    return p


def load_and_resize_image(images_field) -> Image.Image:
    """Extract the first image from a dataset row and downsize it.

    Falls back to a blank placeholder when the row has no usable image, so
    downstream batching never has to special-case a missing image.

    Args:
        images_field: The row's raw `images` value (expected to be a
            sequence of PIL-like image objects, or empty/None).

    Returns:
        Image.Image: An RGB image thumbnailed to at most MAX_IMAGE_SIDE on
        its longest side, or a 224x224 white placeholder if none was usable.
    """
    if images_field and hasattr(images_field[0], "convert"):
        image = images_field[0].convert("RGB")
        image.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))
        return image
    return Image.new("RGB", (224, 224), (255, 255, 255))


class VQADataset(Dataset):
    """Torch dataset that formats VQA rows into prompt/answer pairs for training.

    Each item is tokenized with the image, with the prompt span masked out
    of `labels` so loss is only computed on the answer tokens.

    Attributes:
        rows: List of preprocessed dicts with `image`, `question`, and
            `answer` keys.
        processor: The model's `AutoProcessor`, used for joint text/image
            tokenization.
        max_length: Max token length passed to the processor.
        image_token: The processor's image placeholder token string.
    """

    def __init__(self, rows, processor, max_length: int):
        """Initialize the dataset.

        Args:
            rows: List of preprocessed row dicts (see class docstring).
            processor: The model's `AutoProcessor`.
            max_length (int): Max token length for truncation.
        """
        self.rows = rows
        self.processor = processor
        self.max_length = max_length
        self.image_token = getattr(processor, "image_token", "<|image_pad|>")

    def __len__(self):
        """Return the number of rows in the dataset.

        Returns:
            int: Number of rows.
        """
        return len(self.rows)

    def __getitem__(self, idx):
        """Build the tokenized, label-masked training example for one row.

        Args:
            idx: Index of the row to fetch.

        Returns:
            dict: `input_ids`, `attention_mask`, `labels` (prompt span set to
            -100), `pixel_values`, and `image_grid_thw` tensors for the
            example.
        """
        row = self.rows[idx]
        image = row["image"]
        question = row["question"]
        answer = row["answer"]

        prompt = f"{self.image_token}\n### Instruction:\n{INSTRUCTION}\n\n### Input:\n{question}\n\n### Response:\n"
        full_text = prompt + answer

        prompt_inputs = self.processor(text=[prompt], images=[image], return_tensors="pt", truncation=True, max_length=self.max_length)
        full_inputs = self.processor(text=[full_text], images=[image], return_tensors="pt", truncation=True, max_length=self.max_length)

        input_ids = full_inputs["input_ids"][0]
        labels = input_ids.clone()
        prompt_len = min(prompt_inputs["input_ids"].shape[1], input_ids.shape[0])
        labels[:prompt_len] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": full_inputs["attention_mask"][0],
            "labels": labels,
            "pixel_values": full_inputs["pixel_values"],
            "image_grid_thw": full_inputs["image_grid_thw"],
        }


def vqa_collate_fn(features, pad_token_id: int):
    """Pad and stack a list of VQADataset examples into a training batch.

    Args:
        features: List of per-example dicts as returned by
            `VQADataset.__getitem__`.
        pad_token_id (int): Token id used to right-pad `input_ids`.

    Returns:
        dict: Batched `input_ids`, `attention_mask`, `labels` (padded with
        -100), `pixel_values`, and `image_grid_thw` tensors.
    """
    max_len = max(f["input_ids"].shape[0] for f in features)
    input_ids, attention_mask, labels = [], [], []
    for f in features:
        pad_len = max_len - f["input_ids"].shape[0]
        input_ids.append(torch.cat([f["input_ids"], torch.full((pad_len,), pad_token_id, dtype=torch.long)]))
        attention_mask.append(torch.cat([f["attention_mask"], torch.zeros(pad_len, dtype=torch.long)]))
        labels.append(torch.cat([f["labels"], torch.full((pad_len,), -100, dtype=torch.long)]))
    return {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_mask),
        "labels": torch.stack(labels),
        "pixel_values": torch.cat([f["pixel_values"] for f in features], dim=0),
        "image_grid_thw": torch.cat([f["image_grid_thw"] for f in features], dim=0),
    }


def verify_dataset() -> None:
    """Stream one example from DATASET_NAME and sanity-check its fields.

    Asserts that `images`, `question`, and `answer` are all present, and
    prints a preview. Not called from `main()` by default (see the comment
    there) but kept for manual verification.

    Raises:
        AssertionError: If any expected field is missing from the dataset.
    """
    print_banner("VERIFYING DATASET")
    peek = load_dataset(DATASET_NAME, split="train", streaming=True)
    example = next(iter(peek))
    for field in ("images", "question", "answer"):
        assert field in example, f"Expected a {field!r} field in {DATASET_NAME}, got fields: {list(example.keys())}"
    print(f"Dataset: {DATASET_NAME} (streamed)")
    print(f"Fields: {list(example.keys())}")
    print(f"Sample: {len(example['images'])} image(s), question={example['question'][:100]!r}")
    print()


def load_and_prepare_data(args, processor):
    """Stream, preprocess, split, and subsample the dataset into train/eval sets.

    Args:
        args: Parsed CLI namespace; uses `max_samples`, `sample_selection`,
            `max_eval_samples`, `seed`, and `max_length`.
        processor: The model's `AutoProcessor`, passed through to the
            returned datasets.

    Returns:
        tuple: `(train_dataset, eval_dataset, eval_rows)` where the first
        two are `VQADataset` instances and `eval_rows` is the raw list of
        eval-split row dicts (used later for generation-based evaluation).
    """
    print_banner("LOADING DATASET")
    raw = load_dataset(DATASET_NAME, split="train", streaming=True)
    # Extract only what's needed (and resize the image) immediately on ingest, rather than
    # keeping the full raw row (unresized image + unused columns like code/code_solution) --
    # see MAX_IMAGE_SIDE's comment for why this matters.
    rows = [
        {
            "image": load_and_resize_image(row.get("images")),
            "question": str(row.get("question", "")),
            "answer": str(row.get("answer", "")),
        }
        for _, row in zip(range(MAX_RAW_EXAMPLES), raw)
    ]

    n = len(rows)
    test_size = max(1, n // 10)
    eval_rows = rows[-test_size:]
    train_rows = rows[:-test_size]

    import random as _random
    if args.max_samples != -1:
        if args.sample_selection == "random":
            _random.Random(args.seed).shuffle(train_rows)
            train_rows = train_rows[: args.max_samples]
        elif args.sample_selection == "first":
            train_rows = train_rows[: args.max_samples]
        elif args.sample_selection == "last":
            train_rows = train_rows[-args.max_samples :]
    if args.max_eval_samples != -1:
        eval_rows = eval_rows[: args.max_eval_samples]

    print(f"Train samples: {len(train_rows)}")
    print(f"Eval samples: {len(eval_rows)}")
    print()

    return VQADataset(train_rows, processor, args.max_length), VQADataset(eval_rows, processor, args.max_length), eval_rows


def print_formatted_examples_vqa(dataset, processor, num_examples=2):
    """Print decoded prompt/answer text for a few dataset examples.

    Used by `--debug_first_batch` to visually confirm formatting and label
    masking before committing to a full training run.

    Args:
        dataset: A `VQADataset` (or compatible indexable dataset) to sample
            from.
        processor: The model's `AutoProcessor`, used to decode token ids
            back to text.
        num_examples: Maximum number of examples to print. Defaults to 2.
    """
    print_banner("FORMATTED EXAMPLES")
    for i in range(min(num_examples, len(dataset))):
        example = dataset[i]
        input_ids = example["input_ids"]
        labels = example["labels"]
        full_text = processor.tokenizer.decode(input_ids, skip_special_tokens=False)
        label_tokens = [tid for tid, lab in zip(input_ids.tolist(), labels.tolist()) if lab != -100]
        label_text = processor.tokenizer.decode(label_tokens, skip_special_tokens=True)
        print(f"\n--- Example {i + 1} ---")
        print(f"Full text (image tokens expanded):\n{full_text[:500]}...")
        print(f"\nAnswer (loss computed on): {label_text!r}")
        print(f"image_grid_thw: {example['image_grid_thw'].tolist()}")
    print()


def evaluate_model(model, processor, eval_rows, args):
    """Generate answers for the eval set and score exact-match accuracy.

    Args:
        model: The trained (or base) vision-language model to generate from.
        processor: The model's `AutoProcessor`, used to build prompts and
            decode generations.
        eval_rows: Raw eval-split row dicts (as returned by
            `load_and_prepare_data`) to evaluate on.
        args: Parsed CLI namespace; uses `max_length`.

    Returns:
        dict: `exact_match_accuracy`.
    """
    print_banner("EVALUATION")
    model.eval()
    predictions, ground_truths = [], []
    image_token = getattr(processor, "image_token", "<|image_pad|>")

    with torch.no_grad():
        for i, row in enumerate(eval_rows):
            if i % 10 == 0:
                print(f"  progress: {i}/{len(eval_rows)}")
            image = row["image"]
            question = row["question"]
            prompt = f"{image_token}\n### Instruction:\n{INSTRUCTION}\n\n### Input:\n{question}\n\n### Response:\n"
            inputs = processor(text=[prompt], images=[image], return_tensors="pt", truncation=True, max_length=args.max_length).to(model.device)
            output = model.generate(**inputs, max_new_tokens=32, do_sample=False, pad_token_id=processor.tokenizer.pad_token_id)
            generated = processor.tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

            predictions.append(generated)
            ground_truths.append(str(row.get("answer", "")).strip())

    exact_match = accuracy_score(ground_truths, predictions)
    print(f"Exact-match accuracy: {exact_match:.4f}")
    return {"exact_match_accuracy": exact_match}


def main():
    """Run the full VQA SFT pipeline: load, train, evaluate, save, record.

    Parses CLI args, loads and preprocesses the dataset, loads the base
    vision-language model (optionally quantized/LoRA-adapted), either dumps
    formatted debug examples and exits or trains via `Trainer`, evaluates
    the result, saves the model, and writes a `run_result.json`.
    """
    args = build_arg_parser().parse_args()
    set_all_seeds(args.seed)
    print_config(args, "VQA SFT -- decoder-only (vision-language), no CoT")

    # verify_dataset()  # commented out: avoids loading the dataset twice (a streaming peek here, then a full load in load_and_prepare_data()) to cut memory overhead
    processor = AutoProcessor.from_pretrained(args.model)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    train_dataset, eval_dataset, eval_rows = load_and_prepare_data(args, processor)

    torch_dtype = torch.bfloat16 if args.mixed_precision == "bf16" else torch.float16
    quant_config = None
    if args.quantization != "no":
        from common.quantization import build_quantization_config

        quant_config = build_quantization_config(args.quantization, args.mixed_precision)
    model = load_vision_language_model(args.model, quant_config, torch_dtype, args.gradient_checkpointing)

    if args.lora:
        from common.peft_setup import apply_lora, build_lora_config

        lora_config = build_lora_config(args.lora_r, args.lora_alpha, args.lora_dropout, args.lora_target_modules)
        model = apply_lora(model, lora_config, prepare_for_kbit=(quant_config is not None))

    if args.debug_first_batch:
        print_formatted_examples_vqa(train_dataset, processor, num_examples=2)
        print("--debug_first_batch set: exiting without training.")
        return

    data_collator = lambda features: vqa_collate_fn(features, processor.tokenizer.pad_token_id)
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
        gradient_checkpointing=args.gradient_checkpointing,
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        seed=args.seed,
        report_to=[],
        ddp_find_unused_parameters=(True if args.lora else None),
        remove_unused_columns=False,
    )

    trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=data_collator)

    print_banner("TRAINING")
    start = time.time()
    trainer.train()
    train_runtime_seconds = time.time() - start

    metrics = evaluate_model(model, processor, eval_rows, args)

    save_strategy = args.save_strategy if args.lora else "full"
    save_model(model, processor.tokenizer, args.output_dir, strategy=save_strategy, base_model_name=args.model, is_lora=args.lora)
    processor.save_pretrained(args.output_dir)

    write_run_result(
        output_dir=args.output_dir,
        stage="supervised-finetuning",
        task="vqa",
        modality="image",
        architecture=ARCHITECTURE,
        variant="no_cot",
        cot_enabled=False,
        model_name=args.model,
        dataset_name=DATASET_NAME,
        save_strategy=save_strategy,
        hyperparameters=vars(args),
        metrics=metrics,
        num_train_samples=len(train_dataset),
        num_eval_samples=len(eval_dataset),
        train_runtime_seconds=train_runtime_seconds,
    )
    print("Done.")


if __name__ == "__main__":
    main()
