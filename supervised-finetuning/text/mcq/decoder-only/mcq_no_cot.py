"""Medical MCQ SFT -- decoder-only, no CoT.

Dataset: HPAI-BSC/medmcqa-cot-llama31. Fields verified live: system_prompt,
question, response. `question` is already self-contained (it embeds the
question text AND the "Options: A. ... D. ..." listing), so no separate
`options` field is needed or exists -- this fixes a bug in the archived
version of this script (old/mcq_no_cot.py), which read a nonexistent
`options` field and always fell back to placeholder dummy option text
instead of the real (already-present-in-`question`) options. The answer
letter is extracted from the end of `response` (format "... Answer: X.").
See supervised-finetuning/README.md.

Format: ### Instruction: ... ### Input: ... ### Response: {A|B|C|D}
Loss: answer-letter token(s) only (no reasoning).

Usage:
    python mcq_no_cot.py --debug_first_batch --max_samples 20
"""

import argparse
import re
import time

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
INSTRUCTION = "Answer the following medical multiple choice question by selecting the correct option."
ANSWER_RE = re.compile(r"Answer:\s*([A-D])", re.IGNORECASE)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for this training script.

    Returns:
        argparse.ArgumentParser: Parser covering model/quantization, LoRA,
            training hyperparameters, data sampling, output/checkpointing,
            and seeding/debug flags.
    """
    p = argparse.ArgumentParser(description="SFT a decoder-only model for medical MCQ (no CoT).")

    p.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B-Base", help="Base checkpoint. Default: Qwen/Qwen3-1.7B-Base.")
    p.add_argument("--quantization", type=str, default="no", choices=["no", "4bit", "8bit"], help="Quantization for loading. Default: no.")

    p.add_argument("--lora", action="store_true", default=False, help="Enable LoRA finetuning.")
    p.add_argument("--lora_r", type=int, default=16, help="LoRA rank. Default: 16.")
    p.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha. Default: 32.")
    p.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout. Default: 0.05.")
    p.add_argument("--lora_target_modules", type=str, default="q_proj,k_proj,v_proj,o_proj", help="Comma-separated LoRA target modules.")

    p.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"], help="Mixed precision. Default: bf16.")
    p.add_argument("--batch_size", type=int, default=4, help="Per-device training batch size. Default: 4.")
    p.add_argument("--eval_batch_size", type=int, default=8, help="Per-device eval batch size. Default: 8.")
    p.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps. Default: 4.")
    p.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate. Default: 2e-4.")
    p.add_argument("--epochs", type=int, default=3, help="Training epochs. Default: 3.")
    p.add_argument("--max_length", type=int, default=1024, help="Max sequence length. Default: 1024.")
    p.add_argument("--warmup_steps", type=int, default=100, help="LR warmup steps. Default: 100.")
    p.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay. Default: 0.01.")
    p.add_argument("--gradient_checkpointing", action="store_true", default=True, help="Enable gradient checkpointing. Default: on.")

    p.add_argument("--max_samples", type=int, default=-1, help="Number of training rows to use. -1 (default) = full split.")
    p.add_argument("--sample_selection", type=str, default="random", choices=["random", "first", "last"], help="Which rows to pick when --max_samples is positive. Default: random.")
    p.add_argument("--max_eval_samples", type=int, default=200, help="Eval rows to use. Default: 200.")

    p.add_argument("--output_dir", type=str, default="./output/supervised-finetuning/text/mcq/decoder-only/no_cot", help="Output directory.")
    p.add_argument("--logging_steps", type=int, default=10, help="Logging frequency. Default: 10.")
    p.add_argument("--eval_steps", type=int, default=200, help="Evaluation frequency. Default: 200.")
    p.add_argument("--save_steps", type=int, default=200, help="Checkpoint save frequency. Default: 200.")
    p.add_argument("--save_strategy", type=str, default="adapter_only", choices=["full", "adapter_only", "merged", "adapter_and_merged", "base_reference"], help="Model saving strategy. Default: adapter_only when --lora, otherwise 'full'.")

    p.add_argument("--seed", type=int, default=42, help="Random seed. Default: 42.")
    p.add_argument("--debug_first_batch", action="store_true", default=False, help="Print formatted examples and exit without training.")

    return p


def extract_answer(response: str) -> str:
    """Extract the letter answer (A-D) from a raw dataset response string.

    Args:
        response (str): Raw "response" field text, expected to end with "Answer: X.".

    Returns:
        str: The matched uppercase letter, or "A" as a fallback if no match is found.
    """
    match = ANSWER_RE.search(response)
    return match.group(1).upper() if match else "A"


class MCQDataset(Dataset):
    """Tokenized medical-MCQ dataset with answer-only supervision (no reasoning).

    Formats each row as ``### Instruction: ... ### Input: ... ### Response:
    {A|B|C|D}`` and masks the prompt out of the loss, so loss is computed on
    the answer-letter token(s) only.

    Attributes:
        rows (list): Raw dataset rows, each with "question" and "response" fields.
        tokenizer: Tokenizer used to encode prompt and full text.
        max_length (int): Max sequence length used when tokenizing the full example.
    """

    def __init__(self, rows, tokenizer, max_length: int):
        """Initialize the dataset.

        Args:
            rows (list): Raw dataset rows, each with "question" and "response" fields.
            tokenizer: Tokenizer used to encode prompt and full text.
            max_length (int): Max sequence length used when tokenizing the full example.
        """
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        """Return the number of examples in the dataset.

        Returns:
            int: Number of rows.
        """
        return len(self.rows)

    def __getitem__(self, idx):
        """Build a tokenized, loss-masked training example with answer-only supervision.

        Args:
            idx (int): Index of the row to fetch.

        Returns:
            dict: ``input_ids``, ``attention_mask``, and ``labels`` (prompt
                tokens masked with -100; only the answer letter kept).
        """
        row = self.rows[idx]
        answer = extract_answer(row["response"])
        prompt = f"### Instruction:\n{INSTRUCTION}\n\n### Input:\n{row['question']}\n\n### Response:\n"
        full_text = prompt + answer

        prompt_ids = self.tokenizer(prompt, add_special_tokens=True, truncation=False)["input_ids"]
        full = self.tokenizer(full_text, max_length=self.max_length, truncation=True, add_special_tokens=True)

        input_ids = full["input_ids"]
        labels = input_ids.copy()
        prompt_len = min(len(prompt_ids), len(input_ids))
        labels[:prompt_len] = [-100] * prompt_len

        return {"input_ids": input_ids, "attention_mask": full["attention_mask"], "labels": labels}


def verify_dataset() -> None:
    """Peek the dataset via streaming, assert expected fields, and sanity-check answer extraction.

    Not called by default in ``main`` (see the comment there) since it would
    otherwise trigger a second load of the same dataset.
    """
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
    """Load the full dataset, carve off a 10% eval split, and apply sample selection.

    Args:
        args (argparse.Namespace): Parsed CLI args; uses ``max_samples``,
            ``sample_selection``, ``max_eval_samples``, ``max_length``, and ``seed``.
        tokenizer: Tokenizer passed through to the constructed datasets.

    Returns:
        tuple: ``(train_dataset, eval_dataset, eval_rows)`` where the first two
            are :class:`MCQDataset` instances and ``eval_rows`` is the raw
            (untokenized) list of eval rows for later generation-based evaluation.
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
    return MCQDataset(train_rows, tokenizer, args.max_length), MCQDataset(eval_rows, tokenizer, args.max_length), eval_rows


def decode_example(example, index, tokenizer):
    """Render a tokenized example back to text for ``--debug_first_batch`` inspection.

    Args:
        example (dict): Tokenized example with ``input_ids`` and ``labels``.
        index (int): Position of this example in the batch being printed (unused
            in the body but kept for a uniform ``decode_fn`` signature).
        tokenizer: Tokenizer used to decode ``input_ids`` and the unmasked labels.

    Returns:
        str: Human-readable dump of the full formatted text and the answer
            letter the loss is computed on.
    """
    input_ids = example["input_ids"]
    labels = example["labels"]
    full_text = tokenizer.decode(input_ids, skip_special_tokens=False)
    label_tokens = [tid for tid, lab in zip(input_ids, labels) if lab != -100]
    label_text = tokenizer.decode(label_tokens, skip_special_tokens=True)
    return f"Full text:\n{full_text}\n\nAnswer (loss computed on): {label_text!r}\nNote: no reasoning included."


def evaluate_model(model, tokenizer, eval_rows, args):
    """Generate a short completion per eval row and score answer-letter accuracy.

    Args:
        model: Trained causal LM used for generation.
        tokenizer: Tokenizer used to build prompts and decode generations.
        eval_rows (list): Raw eval rows with "question" and "response" fields.
        args (argparse.Namespace): Parsed CLI args; uses ``max_length``.

    Returns:
        dict: ``{"accuracy": float, "f1_macro": float, "cot_enabled": False}``.
    """
    print_banner("EVALUATION")
    model.eval()
    predictions, ground_truths = [], []

    with torch.no_grad():
        for i, row in enumerate(eval_rows):
            if i % 50 == 0:
                print(f"  progress: {i}/{len(eval_rows)}")
            prompt = f"### Instruction:\n{INSTRUCTION}\n\n### Input:\n{row['question']}\n\n### Response:\n"
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=args.max_length).to(model.device)
            output = model.generate(**inputs, max_new_tokens=5, do_sample=False, pad_token_id=tokenizer.pad_token_id)
            generated = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip().upper()

            pred = next((c for c in generated[:10] if c in "ABCD"), "A")
            predictions.append(pred)
            ground_truths.append(extract_answer(row["response"]))

    accuracy = accuracy_score(ground_truths, predictions)
    f1_macro = f1_score(ground_truths, predictions, average="macro")
    print(f"Accuracy: {accuracy:.4f}  F1(macro): {f1_macro:.4f}")
    return {"accuracy": accuracy, "f1_macro": f1_macro, "cot_enabled": False}


def main():
    """Run the end-to-end MCQ-no-CoT pipeline: load, train, evaluate, save, record."""
    args = build_arg_parser().parse_args()
    set_all_seeds(args.seed)
    print_config(args, "Medical MCQ SFT -- decoder-only, no CoT")

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
