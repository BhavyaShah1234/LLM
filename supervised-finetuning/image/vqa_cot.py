"""Visual Question Answering SFT -- decoder-only (vision-language), with CoT.

Dataset: opendatalab/ChartVerse-SFT-1.8M (streamed), same as vqa_no_cot.py.
This dataset DOES have a genuine reasoning column -- `cot_solution` --
confirmed live, which fixes a real bug in the archived version of this
script (old/vqa_cot.py): it looked for a `reasoning` field, which doesn't
exist, so it always fell back to a generic placeholder sentence instead of
the dataset's actual chart-reasoning text (on top of the same wrong-image-field
and no-labels bugs described in vqa_no_cot.py's docstring, which also apply
here and are fixed the same way). `cot_solution`'s own text already includes
a `<think>` wrapper (found while smoke testing -- an early version of this
fix double-tagged `<think><think>...`), so it's stripped before this
project's own `<think>{reasoning}</think>` wrapping is applied. See
supervised-finetuning/README.md.

Format: <image token>\\n### Instruction: ... ### Input: {question} ### Response: <think>{cot_solution}</think>{answer}
Loss: on BOTH the reasoning and answer tokens.

See vqa_no_cot.py's docstring for the image-token-placeholder approach and
the custom collator (Qwen2-VL's dynamic image resolution needs both).

Usage:
    python vqa_cot.py --debug_first_batch --max_samples 20
"""

import argparse
import re
import time

import numpy as np
import torch
from bert_score import score as bert_score
from datasets import load_dataset
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoProcessor, Trainer, TrainingArguments

from common.logging_utils import print_banner, print_config
from common.model_loading import load_vision_language_model
from common.model_saving import save_model
from common.run_results import write_run_result
from common.seeding import set_all_seeds

DATASET_NAME = "opendatalab/ChartVerse-SFT-1.8M"
ARCHITECTURE = "decoder-only"
INSTRUCTION = "Answer the question about the image. Think through your visual reasoning."
MAX_RAW_EXAMPLES = 800  # see vqa_no_cot.py's MAX_IMAGE_SIDE comment: materializing 5000
# full-resolution images OOM-killed the process during smoke testing (RSS past 27GB).
MAX_IMAGE_SIDE = 672


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="SFT a vision-language decoder-only model for VQA (with CoT).")

    p.add_argument("--model", type=str, default="Qwen/Qwen2-VL-2B", help="Base VL checkpoint (non-instruct). Default: Qwen/Qwen2-VL-2B (fp16 ~4.4GB).")
    p.add_argument("--quantization", type=str, default="no", choices=["no", "4bit", "8bit"], help="Quantization for loading. Default: no.")

    p.add_argument("--lora", action="store_true", default=False, help="Enable LoRA finetuning.")
    p.add_argument("--lora_r", type=int, default=16, help="LoRA rank. Default: 16.")
    p.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha. Default: 32.")
    p.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout. Default: 0.05.")
    p.add_argument("--lora_target_modules", type=str, default="q_proj,k_proj,v_proj,o_proj", help="Comma-separated LoRA target modules.")

    p.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"], help="Mixed precision. Default: bf16.")
    p.add_argument("--batch_size", type=int, default=1, help="Per-device training batch size. Default: 1.")
    p.add_argument("--eval_batch_size", type=int, default=1, help="Per-device eval batch size. Default: 1.")
    p.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Gradient accumulation steps. Default: 8.")
    p.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate. Default: 1e-4.")
    p.add_argument("--epochs", type=int, default=3, help="Training epochs. Default: 3.")
    p.add_argument("--max_length", type=int, default=1536, help="Max total sequence length (longer, to fit reasoning). Default: 1536.")
    p.add_argument("--warmup_steps", type=int, default=100, help="LR warmup steps. Default: 100.")
    p.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay. Default: 0.01.")
    p.add_argument("--gradient_checkpointing", action="store_true", default=True, help="Enable gradient checkpointing. Default: on.")

    p.add_argument("--max_samples", type=int, default=-1, help="Number of training rows to use (from the first MAX_RAW_EXAMPLES streamed). -1 (default) = use all of them.")
    p.add_argument("--sample_selection", type=str, default="random", choices=["random", "first", "last"], help="Which rows to pick when --max_samples is positive. Default: random.")
    p.add_argument("--max_eval_samples", type=int, default=30, help="Eval rows to use (CoT VL generation is slow). Default: 30.")

    p.add_argument("--output_dir", type=str, default="./output/supervised-finetuning/image/vqa_cot", help="Output directory.")
    p.add_argument("--logging_steps", type=int, default=10, help="Logging frequency. Default: 10.")
    p.add_argument("--eval_steps", type=int, default=200, help="Evaluation frequency. Default: 200.")
    p.add_argument("--save_steps", type=int, default=200, help="Checkpoint save frequency. Default: 200.")
    p.add_argument("--save_strategy", type=str, default="adapter_only", choices=["full", "adapter_only", "merged", "adapter_and_merged", "base_reference"], help="Model saving strategy. Default: adapter_only when --lora, otherwise 'full'.")

    p.add_argument("--seed", type=int, default=42, help="Random seed. Default: 42.")
    p.add_argument("--debug_first_batch", action="store_true", default=False, help="Print formatted examples and exit without training.")

    return p


def load_and_resize_image(images_field) -> Image.Image:
    if images_field and hasattr(images_field[0], "convert"):
        image = images_field[0].convert("RGB")
        image.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))
        return image
    return Image.new("RGB", (224, 224), (255, 255, 255))


def get_reasoning(row) -> str:
    # cot_solution's own text already includes a <think> wrapper (confirmed while smoke
    # testing -- a sample started with "<think>Okay, let's tackle this problem..."), so it
    # must be stripped here before this project's own <think>{reasoning}</think> wrapping is
    # applied, or the tag ends up duplicated.
    reasoning = str(row.get("cot_solution", "") or "")
    reasoning = reasoning.replace("<think>", "").replace("</think>", "").strip()[:800]
    return reasoning or "Analyzing the visual elements of the chart to determine the answer."


class VQACoTDataset(Dataset):
    def __init__(self, rows, processor, max_length: int):
        self.rows = rows
        self.processor = processor
        self.max_length = max_length
        self.image_token = getattr(processor, "image_token", "<|image_pad|>")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        image = row["image"]
        question = row["question"]
        answer = row["answer"]
        reasoning = row["reasoning"]

        prompt = f"{self.image_token}\n### Instruction:\n{INSTRUCTION}\n\n### Input:\n{question}\n\n### Response:\n"
        response = f"<think>{reasoning}</think>{answer}"
        full_text = prompt + response

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
    print_banner("VERIFYING DATASET")
    peek = load_dataset(DATASET_NAME, split="train", streaming=True)
    example = next(iter(peek))
    for field in ("images", "question", "answer", "cot_solution"):
        assert field in example, f"Expected a {field!r} field in {DATASET_NAME}, got fields: {list(example.keys())}"
    print(f"Dataset: {DATASET_NAME} (streamed)")
    print(f"Fields: {list(example.keys())}")
    print(f"Sample reasoning: {get_reasoning(example)[:150]!r}")
    print()


def load_and_prepare_data(args, processor):
    print_banner("LOADING DATASET")
    raw = load_dataset(DATASET_NAME, split="train", streaming=True)
    rows = [
        {
            "image": load_and_resize_image(row.get("images")),
            "question": str(row.get("question", "")),
            "answer": str(row.get("answer", "")),
            "reasoning": get_reasoning(row),
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
    print("Loss computed on: BOTH reasoning and answer tokens")
    print()

    return VQACoTDataset(train_rows, processor, args.max_length), VQACoTDataset(eval_rows, processor, args.max_length), eval_rows


def print_formatted_examples_vqa(dataset, processor, num_examples=2):
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
        print(f"\nResponse (loss on BOTH think + answer): {label_text[:400]}")
        print(f"image_grid_thw: {example['image_grid_thw'].tolist()}")
    print()


def evaluate_model(model, processor, eval_rows, args):
    print_banner("EVALUATION")
    model.eval()
    predictions, references, cot_outputs = [], [], []
    image_token = getattr(processor, "image_token", "<|image_pad|>")

    with torch.no_grad():
        for i, row in enumerate(eval_rows):
            if i % 10 == 0:
                print(f"  progress: {i}/{len(eval_rows)}")
            image = row["image"]
            question = row["question"]
            prompt = f"{image_token}\n### Instruction:\n{INSTRUCTION}\n\n### Input:\n{question}\n\n### Response:\n"
            inputs = processor(text=[prompt], images=[image], return_tensors="pt", truncation=True, max_length=args.max_length).to(model.device)
            output = model.generate(**inputs, max_new_tokens=256, do_sample=False, pad_token_id=processor.tokenizer.pad_token_id)
            generated = processor.tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

            cot_match = re.search(r"<think>(.*?)</think>", generated, re.DOTALL)
            cot_text = cot_match.group(1).strip() if cot_match else ""
            remaining = (generated[cot_match.end():] if cot_match else generated).strip()
            cot_outputs.append(cot_text)

            predictions.append(remaining or " ")
            references.append(str(row.get("answer", "")).strip() or " ")

    exact_matches = [1.0 if p.strip() == r.strip() else 0.0 for p, r in zip(predictions, references)]
    exact_match = sum(exact_matches) / len(exact_matches) if exact_matches else 0.0

    try:
        _, _, bert_f1 = bert_score(predictions, references, lang="en", verbose=False)
        bertscore_f1 = float(bert_f1.mean())
    except Exception as e:
        print(f"  BERTScore failed ({e}), skipping.")
        bertscore_f1 = None

    num_with_cot = sum(1 for c in cot_outputs if c)
    avg_cot_length = float(np.mean([len(c.split()) for c in cot_outputs if c])) if num_with_cot else 0.0
    cot_usage_rate = num_with_cot / len(eval_rows) if eval_rows else 0.0

    print(f"Exact-match accuracy: {exact_match:.4f}  BERTScore(F1): {bertscore_f1}")
    print(f"CoT usage rate: {cot_usage_rate:.2%}  Avg CoT length: {avg_cot_length:.1f} words")
    return {
        "exact_match_accuracy": exact_match,
        "bertscore_f1": bertscore_f1,
        "cot_enabled": True,
        "cot_usage_rate": cot_usage_rate,
        "avg_cot_length_words": avg_cot_length,
    }


def main():
    args = build_arg_parser().parse_args()
    set_all_seeds(args.seed)
    print_config(args, "VQA SFT -- decoder-only (vision-language), with CoT")

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
        variant="cot",
        cot_enabled=True,
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
