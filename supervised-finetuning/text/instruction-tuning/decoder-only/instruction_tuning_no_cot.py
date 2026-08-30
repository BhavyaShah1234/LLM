"""Instruction tuning SFT -- decoder-only, no CoT (code instructions).

Dataset: domofon/evol-instruct-code-cot-80k. Fields verified live:
instruction, thinking, response -- the archived version of this script
(old/instruction_tuning_no_cot.py) read a nonexistent `output` field for the
target text (the real field is `response`), so every training example's
target was empty. This version uses `response` directly (CoT is simply
never included here, so there's nothing to strip). See
supervised-finetuning/README.md.

Format: ### Instruction: ... ### Input: ... ### Response: {response}
Loss: response tokens only (no reasoning).

Usage:
    python instruction_tuning_no_cot.py --debug_first_batch --max_samples 20
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

from common.data_selection import select_samples
from common.logging_utils import print_banner, print_config, print_formatted_examples
from common.model_loading import load_causal_lm, load_tokenizer
from common.model_saving import save_model
from common.peft_setup import apply_lora, build_lora_config
from common.quantization import build_quantization_config
from common.run_results import write_run_result
from common.seeding import set_all_seeds

DATASET_NAME = "domofon/evol-instruct-code-cot-80k"
ARCHITECTURE = "decoder-only"

SYSTEM_INSTRUCTIONS = [
    "You are an expert software engineer. Complete the following coding task.",
    "You are a helpful coding assistant. Write correct, clean code for the following task.",
    "Act as a senior developer. Solve the following programming task.",
    "You are a precise code generator. Complete the following task.",
    "As a programming assistant, complete the following task accurately.",
]


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for the no-CoT code-instruction-tuning run.

    Returns:
        argparse.ArgumentParser: Parser covering model/quantization, LoRA,
            optimization, data-selection, output/checkpointing, and
            seed/debug flags for this script.
    """
    p = argparse.ArgumentParser(description="SFT a decoder-only model for instruction tuning (no CoT, code).")

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

    p.add_argument("--max_samples", type=int, default=-1, help="Number of training rows to use. -1 (default) = full split.")
    p.add_argument("--sample_selection", type=str, default="random", choices=["random", "first", "last"], help="Which rows to pick when --max_samples is positive. Default: random.")
    p.add_argument("--max_eval_samples", type=int, default=100, help="Eval rows to use. Default: 100.")

    p.add_argument("--output_dir", type=str, default="./output/supervised-finetuning/text/instruction-tuning/decoder-only/no_cot", help="Output directory.")
    p.add_argument("--logging_steps", type=int, default=10, help="Logging frequency. Default: 10.")
    p.add_argument("--eval_steps", type=int, default=200, help="Evaluation frequency. Default: 200.")
    p.add_argument("--save_steps", type=int, default=200, help="Checkpoint save frequency. Default: 200.")
    p.add_argument("--save_strategy", type=str, default="adapter_only", choices=["full", "adapter_only", "merged", "adapter_and_merged", "base_reference"], help="Model saving strategy. Default: adapter_only when --lora, otherwise 'full'.")

    p.add_argument("--seed", type=int, default=42, help="Random seed. Default: 42.")
    p.add_argument("--debug_first_batch", action="store_true", default=False, help="Print formatted examples and exit without training.")

    return p


class InstructionDataset(Dataset):
    """Tokenized code-instruction dataset (response only, no CoT).

    Formats each row as an instruction/response prompt (with a randomly
    sampled system instruction) where the response is the code answer
    directly, and masks the loss on the instruction/input prefix so loss is
    computed on the response tokens only.

    Attributes:
        rows (list): Raw dataset rows (dicts with "instruction", "response").
        tokenizer: Tokenizer used to encode prompts and responses.
        max_length (int): Max token length the full prompt+response is
            truncated to.
        rng (random.Random): RNG used to sample a system instruction per
            example.
    """

    def __init__(self, rows, tokenizer, max_length: int, seed: int):
        """Initialize the dataset from raw rows.

        Args:
            rows (list): Raw dataset rows (dicts with "instruction",
                "response").
            tokenizer: Tokenizer used to encode prompts and responses.
            max_length (int): Max token length for truncation.
            seed (int): Seed for the system-instruction sampling RNG.
        """
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.rng = random.Random(seed)

    def __len__(self):
        """Return the number of rows in the dataset.

        Returns:
            int: Number of rows.
        """
        return len(self.rows)

    def __getitem__(self, idx):
        """Build the tokenized prompt/response pair for one row.

        Args:
            idx (int): Row index.

        Returns:
            dict: `input_ids`, `attention_mask`, and `labels` (with the
                instruction/input prefix masked to -100, loss on the
                response tokens only).
        """
        row = self.rows[idx]
        instruction = self.rng.choice(SYSTEM_INSTRUCTIONS)
        prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{row['instruction']}\n\n### Response:\n"
        full_text = prompt + row["response"]

        prompt_ids = self.tokenizer(prompt, add_special_tokens=True, truncation=False)["input_ids"]
        full = self.tokenizer(full_text, max_length=self.max_length, truncation=True, add_special_tokens=True)

        input_ids = full["input_ids"]
        labels = input_ids.copy()
        prompt_len = min(len(prompt_ids), len(input_ids))
        labels[:prompt_len] = [-100] * prompt_len

        return {"input_ids": input_ids, "attention_mask": full["attention_mask"], "labels": labels}


def verify_dataset() -> None:
    """Peek the dataset (streaming) and assert the expected fields exist.

    Raises:
        AssertionError: If `instruction` or `response` is missing from the
            first streamed example.
    """
    print_banner("VERIFYING DATASET")
    peek = load_dataset(DATASET_NAME, split="train", streaming=True)
    example = next(iter(peek))
    for field in ("instruction", "response"):
        assert field in example, f"Expected a {field!r} field in {DATASET_NAME}, got fields: {list(example.keys())}"
    print(f"Dataset: {DATASET_NAME}")
    print(f"Fields: {list(example.keys())}")
    print()


def load_and_prepare_data(args, tokenizer):
    """Load the dataset, carve out an eval split, and wrap both in datasets.

    Args:
        args (argparse.Namespace): Parsed CLI args; uses `max_samples`,
            `sample_selection`, `max_eval_samples`, `seed`, and `max_length`.
        tokenizer: Tokenizer passed through to the built datasets.

    Returns:
        tuple: `(train_dataset, eval_dataset, eval_rows)` where the first two
            are `InstructionDataset` instances and `eval_rows` is the raw
            (untokenized) eval rows used for generation-based evaluation.
    """
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
    print()

    train_rows = list(train_raw)
    eval_rows = list(eval_raw)
    return (
        InstructionDataset(train_rows, tokenizer, args.max_length, args.seed),
        InstructionDataset(eval_rows, tokenizer, args.max_length, args.seed),
        eval_rows,
    )


def decode_example(example, index, tokenizer):
    """Render one tokenized training example as human-readable text for debug printing.

    Args:
        example (dict): Tokenized example with `input_ids` and `labels`.
        index (int): Position of this example in the batch (unused, kept for
            the shared `decode_fn` signature used by `print_formatted_examples`).
        tokenizer: Tokenizer used to decode the token ids back to text.

    Returns:
        str: The full prompt+response text, plus the decoded response
            tokens that loss is actually computed on.
    """
    input_ids = example["input_ids"]
    labels = example["labels"]
    full_text = tokenizer.decode(input_ids, skip_special_tokens=False)
    label_tokens = [tid for tid, lab in zip(input_ids, labels) if lab != -100]
    label_text = tokenizer.decode(label_tokens, skip_special_tokens=True)
    return f"Full text:\n{full_text[:600]}...\n\nResponse (loss computed on): {label_text[:400]}\nNote: no reasoning included."


def evaluate_model(model, tokenizer, eval_rows, args):
    """Generate a code response per eval row and score it against the reference.

    Args:
        model: The (possibly LoRA-wrapped) causal LM to evaluate.
        tokenizer: Tokenizer used for prompting and decoding generations.
        eval_rows (list): Raw eval rows (dicts with "instruction",
            "response").
        args (argparse.Namespace): Parsed CLI args; uses `max_length`
            and `seed`.

    Returns:
        dict: ROUGE-L, BERTScore F1 (`None` if scoring failed), and
            `cot_enabled` (always `False`).
    """
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
            references.append(row["response"].strip() or " ")

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
    return {"rouge_l": rouge_l, "bertscore_f1": bertscore_f1, "cot_enabled": False}


def main():
    """Run the full no-CoT code-instruction-tuning SFT pipeline end to end.

    Parses CLI args, loads the tokenizer/dataset/model (optionally
    quantized/LoRA), either prints a debug batch and exits or trains,
    evaluates, saves the model, and writes a `run_result.json`.
    """
    args = build_arg_parser().parse_args()
    set_all_seeds(args.seed)
    print_config(args, "Instruction tuning SFT -- decoder-only, no CoT (code)")

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
