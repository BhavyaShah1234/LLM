"""Medical MCQ SFT -- decoder-only, with CoT.

Dataset: HPAI-BSC/medmcqa-cot-llama31 (same as mcq_no_cot.py). `response`
contains the model-generated reasoning followed by "Answer: X." -- the
reasoning portion (everything before "Answer:") is used as CoT supervision.

Format: ### Instruction: ... ### Input: ... ### Response: <think>{reasoning}</think>Answer: {X}
Loss: on BOTH the reasoning and "Answer: X" tokens.

Usage:
    python mcq_cot.py --debug_first_batch --max_samples 20
"""

import argparse
import re
import time

import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import Dataset
from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments

from common.data_selection import select_samples
from common.logging_utils import print_banner, print_config, print_formatted_examples
from common.model_loading import load_causal_lm, load_tokenizer
from common.model_saving import save_model
from common.peft_setup import apply_lora, build_lora_config
from common.quantization import build_quantization_config
from common.run_results import write_run_result
from common.seeding import set_all_seeds

DATASET_NAME = "HPAI-BSC/medmcqa-cot-llama31"
ARCHITECTURE = "decoder-only"
INSTRUCTION = "Answer the medical question. Think through your reasoning step by step."
ANSWER_RE = re.compile(r"Answer:\s*([A-D])", re.IGNORECASE)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="SFT a decoder-only model for medical MCQ (with CoT).")

    p.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B-Base", help="Base checkpoint. Default: Qwen/Qwen3-1.7B-Base.")
    p.add_argument("--quantization", type=str, default="no", choices=["no", "4bit", "8bit"], help="Quantization for loading. Default: no.")

    p.add_argument("--lora", action="store_true", default=False, help="Enable LoRA finetuning.")
    p.add_argument("--lora_r", type=int, default=16, help="LoRA rank. Default: 16.")
    p.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha. Default: 32.")
    p.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout. Default: 0.05.")
    p.add_argument("--lora_target_modules", type=str, default="q_proj,k_proj,v_proj,o_proj", help="Comma-separated LoRA target modules.")

    p.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"], help="Mixed precision. Default: bf16.")
    p.add_argument("--batch_size", type=int, default=1, help="Per-device training batch size. Default: 1.")
    p.add_argument("--eval_batch_size", type=int, default=2, help="Per-device eval batch size. Default: 2.")
    p.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Gradient accumulation steps. Default: 8.")
    p.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate. Default: 2e-4.")
    p.add_argument("--epochs", type=int, default=3, help="Training epochs. Default: 3.")
    p.add_argument("--max_length", type=int, default=2048, help="Max sequence length. Default: 2048.")
    p.add_argument("--warmup_steps", type=int, default=100, help="LR warmup steps. Default: 100.")
    p.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay. Default: 0.01.")
    p.add_argument("--gradient_checkpointing", action="store_true", default=True, help="Enable gradient checkpointing. Default: on.")

    p.add_argument("--max_samples", type=int, default=-1, help="Number of training rows to use. -1 (default) = full split.")
    p.add_argument("--sample_selection", type=str, default="random", choices=["random", "first", "last"], help="Which rows to pick when --max_samples is positive. Default: random.")
    p.add_argument("--max_eval_samples", type=int, default=100, help="Eval rows to use (CoT generation is slow). Default: 100.")

    p.add_argument("--output_dir", type=str, default="./output/supervised-finetuning/text/mcq/decoder-only/cot", help="Output directory.")
    p.add_argument("--logging_steps", type=int, default=10, help="Logging frequency. Default: 10.")
    p.add_argument("--eval_steps", type=int, default=200, help="Evaluation frequency. Default: 200.")
    p.add_argument("--save_steps", type=int, default=200, help="Checkpoint save frequency. Default: 200.")
    p.add_argument("--save_strategy", type=str, default="adapter_only", choices=["full", "adapter_only", "merged", "adapter_and_merged", "base_reference"], help="Model saving strategy. Default: adapter_only when --lora, otherwise 'full'.")

    p.add_argument("--seed", type=int, default=42, help="Random seed. Default: 42.")
    p.add_argument("--debug_first_batch", action="store_true", default=False, help="Print formatted examples and exit without training.")

    return p


def extract_answer(response: str) -> str:
    match = ANSWER_RE.search(response)
    return match.group(1).upper() if match else "A"


def extract_reasoning(response: str) -> str:
    reasoning = ANSWER_RE.split(response)[0].strip()
    reasoning = reasoning.replace("<think>", "").replace("</think>", "").strip()[:800]
    return reasoning or "Analyzing the options to determine the correct answer."


class MCQCoTDataset(Dataset):
    def __init__(self, rows, tokenizer, max_length: int):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        answer = extract_answer(row["response"])
        reasoning = extract_reasoning(row["response"])

        prompt = f"### Instruction:\n{INSTRUCTION}\n\n### Input:\n{row['question']}\n\n### Response:\n"
        response = f"<think>{reasoning}</think>Answer: {answer}"
        full_text = prompt + response

        prompt_ids = self.tokenizer(prompt, add_special_tokens=True, truncation=False)["input_ids"]
        full = self.tokenizer(full_text, max_length=self.max_length, truncation=True, add_special_tokens=True)

        input_ids = full["input_ids"]
        labels = input_ids.copy()
        prompt_len = min(len(prompt_ids), len(input_ids))
        labels[:prompt_len] = [-100] * prompt_len

        return {"input_ids": input_ids, "attention_mask": full["attention_mask"], "labels": labels}


def verify_dataset() -> None:
    print_banner("VERIFYING DATASET")
    peek = load_dataset(DATASET_NAME, split="train", streaming=True)
    example = next(iter(peek))
    for field in ("question", "response"):
        assert field in example, f"Expected a {field!r} field in {DATASET_NAME}, got fields: {list(example.keys())}"
    print(f"Dataset: {DATASET_NAME}")
    print(f"Fields: {list(example.keys())}")
    print(f"Extracted answer from sample response: {extract_answer(example['response'])!r}")
    print()


def load_and_prepare_data(args, tokenizer):
    print_banner("LOADING DATASET")
    all_raw = load_dataset(DATASET_NAME, split="train")

    n = len(all_raw)
    test_size = n // 10
    eval_raw = all_raw.select(range(n - test_size, n))
    train_raw = all_raw.select(range(n - test_size))

    train_raw = select_samples(train_raw, args.max_samples, args.sample_selection, args.seed)
    eval_raw = select_samples(eval_raw, args.max_eval_samples, "first", args.seed)

    print(f"Train samples: {len(train_raw)}")
    print(f"Eval samples: {len(eval_raw)}")
    print("Loss computed on: BOTH reasoning and 'Answer: X' tokens")
    print()

    train_rows = list(train_raw)
    eval_rows = list(eval_raw)
    return MCQCoTDataset(train_rows, tokenizer, args.max_length), MCQCoTDataset(eval_rows, tokenizer, args.max_length), eval_rows


def decode_example(example, index, tokenizer):
    input_ids = example["input_ids"]
    labels = example["labels"]
    full_text = tokenizer.decode(input_ids, skip_special_tokens=False)
    label_tokens = [tid for tid, lab in zip(input_ids, labels) if lab != -100]
    label_text = tokenizer.decode(label_tokens, skip_special_tokens=True)
    return f"Full text:\n{full_text[:800]}...\n\nResponse (loss on BOTH think + answer): {label_text[:400]}"


def evaluate_model(model, tokenizer, eval_rows, args):
    print_banner("EVALUATION")
    model.eval()
    predictions, ground_truths, cot_outputs = [], [], []

    with torch.no_grad():
        for i, row in enumerate(eval_rows):
            if i % 20 == 0:
                print(f"  progress: {i}/{len(eval_rows)}")
            prompt = f"### Instruction:\n{INSTRUCTION}\n\n### Input:\n{row['question']}\n\n### Response:\n"
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=args.max_length).to(model.device)
            output = model.generate(**inputs, max_new_tokens=256, do_sample=False, pad_token_id=tokenizer.pad_token_id)
            generated = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

            cot_match = re.search(r"<think>(.*?)</think>", generated, re.DOTALL)
            cot_text = cot_match.group(1).strip() if cot_match else ""
            remaining = (generated[cot_match.end():] if cot_match else generated).upper()
            cot_outputs.append(cot_text)

            pred = next((c for c in remaining[:20] if c in "ABCD"), "A")
            predictions.append(pred)
            ground_truths.append(extract_answer(row["response"]))

    accuracy = accuracy_score(ground_truths, predictions)
    f1_macro = f1_score(ground_truths, predictions, average="macro")
    num_with_cot = sum(1 for c in cot_outputs if c)
    avg_cot_length = float(np.mean([len(c.split()) for c in cot_outputs if c])) if num_with_cot else 0.0
    cot_usage_rate = num_with_cot / len(eval_rows) if eval_rows else 0.0

    print(f"Accuracy: {accuracy:.4f}  F1(macro): {f1_macro:.4f}")
    print(f"CoT usage rate: {cot_usage_rate:.2%}  Avg CoT length: {avg_cot_length:.1f} words")
    return {
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "cot_enabled": True,
        "cot_usage_rate": cot_usage_rate,
        "avg_cot_length_words": avg_cot_length,
    }


def main():
    args = build_arg_parser().parse_args()
    set_all_seeds(args.seed)
    print_config(args, "Medical MCQ SFT -- decoder-only, with CoT")

    # verify_dataset()  # commented out: avoids loading the dataset twice (a streaming peek here, then a full load in load_and_prepare_data()) to cut memory overhead
    tokenizer = load_tokenizer(args.model)

    train_dataset, eval_dataset, eval_rows = load_and_prepare_data(args, tokenizer)

    quant_config = build_quantization_config(args.quantization, args.mixed_precision)
    torch_dtype = torch.bfloat16 if args.mixed_precision == "bf16" else torch.float16
    model = load_causal_lm(args.model, quant_config, torch_dtype, args.gradient_checkpointing)

    if args.lora:
        lora_config = build_lora_config(args.lora_r, args.lora_alpha, args.lora_dropout, args.lora_target_modules)
        model = apply_lora(model, lora_config, prepare_for_kbit=(quant_config is not None))

    if args.debug_first_batch:
        print_formatted_examples(train_dataset, tokenizer, num_examples=2, decode_fn=decode_example)
        print("--debug_first_batch set: exiting without training.")
        return

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True)
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
    )

    trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=data_collator)

    print_banner("TRAINING")
    start = time.time()
    trainer.train()
    train_runtime_seconds = time.time() - start

    metrics = evaluate_model(model, tokenizer, eval_rows, args)

    save_strategy = args.save_strategy if args.lora else "full"
    save_model(model, tokenizer, args.output_dir, strategy=save_strategy, base_model_name=args.model, is_lora=args.lora)

    write_run_result(
        output_dir=args.output_dir,
        stage="supervised-finetuning",
        task="mcq",
        modality="text",
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
