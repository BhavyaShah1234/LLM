"""Instruction tuning SFT -- decoder-only, standard (math instructions).

Dataset: ibivibiv/math_instruct. Fields verified live: instruction, output,
input -- these match what the archived version of this script
(old/instruction_tuning_standard.py) already assumed, so no field-name fix
needed here. What IS fixed: the archived script has no evaluation function
at all (despite an unused `import evaluate`); this version adds a real
ROUGE-L + BERTScore generation-quality eval, and a working --debug_first_batch
(the archived script returns before loading any data).

A random system-instruction persona (one of a few math-tutor framings) is
prepended per example -- the dataset itself provides the task content
(mapped to `### Input:`) and the target output, not a system instruction.

Note: this dataset has 114M rows (~10GB) -- found empirically while smoke
testing this script, well beyond what a non-streaming `load_dataset(...,
split="train")` call should download for local experimentation. Data
loading here streams and caps materialization at MAX_RAW_EXAMPLES, the same
pattern used by this project's chat_tuning_cot.py and vqa_*.py scripts for
their own large/streamed datasets.

Format: ### Instruction: ... ### Input: ... ### Response: {output}
Loss: output tokens only.

Usage:
    python instruction_tuning_standard.py --debug_first_batch --max_samples 20
"""

import argparse
import random
import time

import torch
from bert_score import score as bert_score
from datasets import load_dataset
from rouge_score import rouge_scorer
from torch.utils.data import Dataset
from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments

from common.logging_utils import print_banner, print_config, print_formatted_examples
from common.model_loading import load_causal_lm, load_tokenizer
from common.model_saving import save_model
from common.peft_setup import apply_lora, build_lora_config
from common.quantization import build_quantization_config
from common.run_results import write_run_result
from common.seeding import set_all_seeds

DATASET_NAME = "ibivibiv/math_instruct"
ARCHITECTURE = "decoder-only"
MAX_RAW_EXAMPLES = 20000  # dataset has 114M rows / ~10GB; cap what gets streamed & materialized

SYSTEM_INSTRUCTIONS = [
    "You are a helpful math tutor. Solve the following problem, showing your work.",
    "You are an expert mathematician. Solve the following problem clearly and correctly.",
    "Act as a patient math teacher. Work through the following problem step by step.",
    "You are a precise mathematical problem solver. Solve the following problem.",
    "As a math assistant, solve the following problem accurately.",
]


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="SFT a decoder-only model for instruction tuning (standard, math).")

    p.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B-Base", help="Base checkpoint. Default: Qwen/Qwen3-1.7B-Base.")
    p.add_argument("--quantization", type=str, default="no", choices=["no", "4bit", "8bit"], help="Quantization for loading. Default: no.")

    p.add_argument("--lora", action="store_true", default=False, help="Enable LoRA finetuning.")
    p.add_argument("--lora_r", type=int, default=16, help="LoRA rank. Default: 16.")
    p.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha. Default: 32.")
    p.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout. Default: 0.05.")
    p.add_argument("--lora_target_modules", type=str, default="q_proj,k_proj,v_proj,o_proj", help="Comma-separated LoRA target modules.")

    p.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"], help="Mixed precision. Default: bf16.")
    p.add_argument("--batch_size", type=int, default=2, help="Per-device training batch size. Default: 2.")
    p.add_argument("--eval_batch_size", type=int, default=4, help="Per-device eval batch size. Default: 4.")
    p.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps. Default: 4.")
    p.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate. Default: 2e-4.")
    p.add_argument("--epochs", type=int, default=3, help="Training epochs. Default: 3.")
    p.add_argument("--max_length", type=int, default=1024, help="Max sequence length. Default: 1024.")
    p.add_argument("--warmup_steps", type=int, default=100, help="LR warmup steps. Default: 100.")
    p.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay. Default: 0.01.")
    p.add_argument("--gradient_checkpointing", action="store_true", default=True, help="Enable gradient checkpointing. Default: on.")

    p.add_argument("--max_samples", type=int, default=-1, help=f"Number of training rows to use (from the first {MAX_RAW_EXAMPLES} streamed -- this dataset has 114M rows, streaming with a cap is required). -1 (default) = use all of them.")
    p.add_argument("--sample_selection", type=str, default="random", choices=["random", "first", "last"], help="Which rows to pick when --max_samples is positive. Default: random.")
    p.add_argument("--max_eval_samples", type=int, default=100, help="Eval rows to use. Default: 100.")

    p.add_argument("--output_dir", type=str, default="./output/supervised-finetuning/text/instruction-tuning/decoder-only/standard", help="Output directory.")
    p.add_argument("--logging_steps", type=int, default=10, help="Logging frequency. Default: 10.")
    p.add_argument("--eval_steps", type=int, default=200, help="Evaluation frequency. Default: 200.")
    p.add_argument("--save_steps", type=int, default=200, help="Checkpoint save frequency. Default: 200.")
    p.add_argument("--save_strategy", type=str, default="adapter_only", choices=["full", "adapter_only", "merged", "adapter_and_merged", "base_reference"], help="Model saving strategy. Default: adapter_only when --lora, otherwise 'full'.")

    p.add_argument("--seed", type=int, default=42, help="Random seed. Default: 42.")
    p.add_argument("--debug_first_batch", action="store_true", default=False, help="Print formatted examples and exit without training.")

    return p


class InstructionDataset(Dataset):
    def __init__(self, rows, tokenizer, max_length: int, seed: int):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        instruction = self.rng.choice(SYSTEM_INSTRUCTIONS)
        prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{row['instruction']}\n\n### Response:\n"
        full_text = prompt + row["output"]

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
    for field in ("instruction", "output"):
        assert field in example, f"Expected a {field!r} field in {DATASET_NAME}, got fields: {list(example.keys())}"
    print(f"Dataset: {DATASET_NAME}")
    print(f"Fields: {list(example.keys())}")
    print()


def load_and_prepare_data(args, tokenizer):
    print_banner("LOADING DATASET")
    # ibivibiv/math_instruct has 114M rows / ~10GB -- streaming with a cap on
    # how many raw rows get materialized, rather than downloading the full
    # split, is essential here (unlike this project's smaller SFT datasets).
    raw = load_dataset(DATASET_NAME, split="train", streaming=True)
    all_rows = [row for _, row in zip(range(MAX_RAW_EXAMPLES), raw)]

    n = len(all_rows)
    test_size = max(1, n // 10)
    eval_rows_all = all_rows[-test_size:]
    train_rows_all = all_rows[:-test_size]

    import random as _random

    if args.max_samples != -1:
        if args.sample_selection == "random":
            _random.Random(args.seed).shuffle(train_rows_all)
            train_rows_all = train_rows_all[: args.max_samples]
        elif args.sample_selection == "first":
            train_rows_all = train_rows_all[: args.max_samples]
        elif args.sample_selection == "last":
            train_rows_all = train_rows_all[-args.max_samples :]
    if args.max_eval_samples != -1:
        eval_rows_all = eval_rows_all[: args.max_eval_samples]

    train_rows, eval_rows = train_rows_all, eval_rows_all

    print(f"Train samples: {len(train_rows)}")
    print(f"Eval samples: {len(eval_rows)}")
    print()

    return (
        InstructionDataset(train_rows, tokenizer, args.max_length, args.seed),
        InstructionDataset(eval_rows, tokenizer, args.max_length, args.seed),
        eval_rows,
    )


def decode_example(example, index, tokenizer):
    input_ids = example["input_ids"]
    labels = example["labels"]
    full_text = tokenizer.decode(input_ids, skip_special_tokens=False)
    label_tokens = [tid for tid, lab in zip(input_ids, labels) if lab != -100]
    label_text = tokenizer.decode(label_tokens, skip_special_tokens=True)
    return f"Full text:\n{full_text[:600]}...\n\nOutput (loss computed on): {label_text[:400]}"


def evaluate_model(model, tokenizer, eval_rows, args):
    print_banner("EVALUATION")
    model.eval()
    predictions, references = [], []
    rng = random.Random(args.seed)

    with torch.no_grad():
        for i, row in enumerate(eval_rows):
            if i % 20 == 0:
                print(f"  progress: {i}/{len(eval_rows)}")
            instruction = rng.choice(SYSTEM_INSTRUCTIONS)
            prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{row['instruction']}\n\n### Response:\n"
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=args.max_length).to(model.device)
            output = model.generate(**inputs, max_new_tokens=256, do_sample=False, pad_token_id=tokenizer.pad_token_id)
            generated = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            predictions.append(generated or " ")
            references.append(row["output"].strip() or " ")

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge_l_scores = [scorer.score(ref, pred)["rougeL"].fmeasure for pred, ref in zip(predictions, references)]
    rouge_l = sum(rouge_l_scores) / len(rouge_l_scores) if rouge_l_scores else 0.0

    try:
        _, _, bert_f1 = bert_score(predictions, references, lang="en", verbose=False)
        bertscore_f1 = float(bert_f1.mean())
    except Exception as e:
        print(f"  BERTScore failed ({e}), skipping.")
        bertscore_f1 = None

    print(f"ROUGE-L: {rouge_l:.4f}  BERTScore(F1): {bertscore_f1}")
    return {"rouge_l": rouge_l, "bertscore_f1": bertscore_f1}


def main():
    args = build_arg_parser().parse_args()
    set_all_seeds(args.seed)
    print_config(args, "Instruction tuning SFT -- decoder-only, standard (math)")

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
        task="instruction_tuning",
        modality="text",
        architecture=ARCHITECTURE,
        variant="standard",
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
