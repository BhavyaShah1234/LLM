"""Text classification SFT -- decoder-only, with CoT.

Dataset: domofon/fake_news_cot_reasoning, using the dataset's reasoning
column as Chain-of-Thought supervision. Fields verified live: title, input,
reasoning, status (status: 0=fake, 1=real) -- this fixes a bug in the
archived version of this script (old/text_classification_cot.py), which read
`text`/`label` as primary fields instead of `input`/`status` (the dataset's
actual field names, confirmed live), inconsistent with its no_cot sibling on
the *same* dataset and very likely training on empty strings as a result.
See supervised-finetuning/README.md.

Format: ### Instruction: ... ### Input: ... ### Response: <think>{reasoning}</think>{label}
Loss: on BOTH the reasoning and label tokens.

Usage:
    python text_classification_cot.py --debug_first_batch --max_samples 20
"""

import argparse
import re
import time

import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
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

DATASET_NAME = "domofon/fake_news_cot_reasoning"
ARCHITECTURE = "decoder-only"
LABEL_MAP = {0: "fake", 1: "real"}
INSTRUCTION = "Classify whether the following text is real news or fake news. Think through your reasoning step by step."


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for the CoT fake-news classification run.

    Returns:
        argparse.ArgumentParser: Parser covering model/quantization, LoRA,
            optimization, data-selection, output/checkpointing, and
            seed/debug flags for this script.
    """
    p = argparse.ArgumentParser(description="SFT a decoder-only model for fake-news classification (with CoT).")

    p.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B-Base", help="Base checkpoint to finetune. Default: Qwen/Qwen3-1.7B-Base.")
    p.add_argument("--quantization", type=str, default="no", choices=["no", "4bit", "8bit"], help="Quantization for loading. Default: no.")

    p.add_argument("--lora", action="store_true", default=False, help="Enable LoRA finetuning.")
    p.add_argument("--lora_r", type=int, default=16, help="LoRA rank. Default: 16.")
    p.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha. Default: 32.")
    p.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout. Default: 0.05.")
    p.add_argument("--lora_target_modules", type=str, default="q_proj,k_proj,v_proj,o_proj", help="Comma-separated LoRA target modules.")

    p.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"], help="Mixed precision. Default: bf16.")
    p.add_argument("--batch_size", type=int, default=2, help="Per-device training batch size (smaller than no_cot -- longer sequences). Default: 2.")
    p.add_argument("--eval_batch_size", type=int, default=4, help="Per-device eval batch size. Default: 4.")
    p.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Gradient accumulation steps. Default: 8.")
    p.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate. Default: 2e-4.")
    p.add_argument("--epochs", type=int, default=3, help="Training epochs. Default: 3.")
    p.add_argument("--max_length", type=int, default=2048, help="Max sequence length (longer, to fit reasoning). Default: 2048.")
    p.add_argument("--warmup_steps", type=int, default=100, help="LR warmup steps. Default: 100.")
    p.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay. Default: 0.01.")
    p.add_argument("--gradient_checkpointing", action="store_true", default=True, help="Enable gradient checkpointing. Default: on.")

    p.add_argument("--max_samples", type=int, default=-1, help="Number of training rows to use. -1 (default) = full split.")
    p.add_argument("--sample_selection", type=str, default="random", choices=["random", "first", "last"], help="Which rows to pick when --max_samples is positive. Default: random.")
    p.add_argument("--max_eval_samples", type=int, default=100, help="Eval rows to use (CoT generation is slow). Default: 100.")

    p.add_argument("--output_dir", type=str, default="./output/supervised-finetuning/text/classification/decoder-only/cot", help="Output directory.")
    p.add_argument("--logging_steps", type=int, default=10, help="Logging frequency. Default: 10.")
    p.add_argument("--eval_steps", type=int, default=200, help="Evaluation frequency. Default: 200.")
    p.add_argument("--save_steps", type=int, default=200, help="Checkpoint save frequency. Default: 200.")
    p.add_argument("--save_strategy", type=str, default="adapter_only", choices=["full", "adapter_only", "merged", "adapter_and_merged", "base_reference"], help="Model saving strategy. Default: adapter_only when --lora, otherwise 'full'.")

    p.add_argument("--seed", type=int, default=42, help="Random seed. Default: 42.")
    p.add_argument("--debug_first_batch", action="store_true", default=False, help="Print formatted examples and exit without training.")

    return p


class ClassificationCoTDataset(Dataset):
    """Tokenized real/fake-news classification dataset with CoT supervision.

    Formats each row as an instruction/response prompt where the response is
    a `<think>` reasoning span followed by the "real"/"fake" label, and masks
    the loss on the instruction/input prefix only (loss is computed on both
    the reasoning and the label tokens).

    Attributes:
        rows (list): Raw dataset rows (dicts with "input", "reasoning",
            "status").
        tokenizer: Tokenizer used to encode prompts and responses.
        max_length (int): Max token length the full prompt+response is
            truncated to.
    """

    def __init__(self, rows, tokenizer, max_length: int):
        """Initialize the dataset from raw rows.

        Args:
            rows (list): Raw dataset rows (dicts with "input", "reasoning",
                "status").
            tokenizer: Tokenizer used to encode prompts and responses.
            max_length (int): Max token length for truncation.
        """
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length

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
                reasoning + label tokens).
        """
        row = self.rows[idx]
        text = row["input"][:1500]
        reasoning = (row["reasoning"] or "").strip()[:500]
        if not reasoning:
            reasoning = "Based on the content and wording, I need to determine if this is real or fake news."
        label_text = LABEL_MAP[row["status"]]

        prompt = f"### Instruction:\n{INSTRUCTION}\n\n### Input:\n{text}\n\n### Response:\n"
        response = f"<think>{reasoning}</think>{label_text}"
        full_text = prompt + response

        prompt_ids = self.tokenizer(prompt, add_special_tokens=True, truncation=False)["input_ids"]
        full = self.tokenizer(full_text, max_length=self.max_length, truncation=True, add_special_tokens=True)

        input_ids = full["input_ids"]
        labels = input_ids.copy()
        prompt_len = min(len(prompt_ids), len(input_ids))
        labels[:prompt_len] = [-100] * prompt_len  # Mask prompt only; loss on BOTH reasoning and label.

        return {"input_ids": input_ids, "attention_mask": full["attention_mask"], "labels": labels}


def verify_dataset() -> None:
    """Peek the dataset (streaming) and assert the expected fields exist.

    Raises:
        AssertionError: If `input`, `status`, or `reasoning` is missing from
            the first streamed example.
    """
    print_banner("VERIFYING DATASET")
    peek = load_dataset(DATASET_NAME, split="train", streaming=True)
    example = next(iter(peek))
    for field in ("input", "status", "reasoning"):
        assert field in example, f"Expected a {field!r} field in {DATASET_NAME}, got fields: {list(example.keys())}"
    print(f"Dataset: {DATASET_NAME}")
    print(f"Fields: {list(example.keys())}")
    print(f"Sample: status={example['status']!r}, reasoning={example['reasoning'][:120]!r}...")
    print()


def load_and_prepare_data(args, tokenizer):
    """Load the dataset, carve out an eval split, and wrap both in datasets.

    Args:
        args (argparse.Namespace): Parsed CLI args; uses `max_samples`,
            `sample_selection`, `max_eval_samples`, `seed`, and `max_length`.
        tokenizer: Tokenizer passed through to the built datasets.

    Returns:
        tuple: `(train_dataset, eval_dataset, eval_rows)` where the first two
            are `ClassificationCoTDataset` instances and `eval_rows` is the
            raw (untokenized) eval rows used for generation-based evaluation.
    """
    print_banner("LOADING DATASET")
    train_raw = load_dataset(DATASET_NAME, split="train")

    n = len(train_raw)
    test_size = n // 10
    eval_raw = train_raw.select(range(n - test_size, n))
    train_raw = train_raw.select(range(n - test_size))

    train_raw = select_samples(train_raw, args.max_samples, args.sample_selection, args.seed)
    eval_raw = select_samples(eval_raw, args.max_eval_samples, "first", args.seed)

    print(f"Train samples: {len(train_raw)}")
    print(f"Eval samples: {len(eval_raw)}")
    print(f"Label map: {LABEL_MAP}")
    print("Loss computed on: BOTH reasoning and label tokens")
    print()

    train_rows = list(train_raw)
    eval_rows = list(eval_raw)
    return (
        ClassificationCoTDataset(train_rows, tokenizer, args.max_length),
        ClassificationCoTDataset(eval_rows, tokenizer, args.max_length),
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
            (reasoning + label) that loss is actually computed on.
    """
    input_ids = example["input_ids"]
    labels = example["labels"]
    full_text = tokenizer.decode(input_ids, skip_special_tokens=False)
    label_tokens = [tid for tid, lab in zip(input_ids, labels) if lab != -100]
    label_text = tokenizer.decode(label_tokens, skip_special_tokens=True)
    return f"Full text:\n{full_text[:800]}...\n\nResponse (loss computed on BOTH <think> content AND label):\n{label_text[:400]}..."


def evaluate_model(model, tokenizer, eval_rows, args):
    """Generate CoT + label predictions on the eval rows and score them.

    Parses each generation's `<think>...</think>` span for the CoT text and
    the remaining text for the "fake"/"real" label (defaulting to "real"
    when neither keyword is found), then computes classification metrics.

    Args:
        model: The (possibly LoRA-wrapped) causal LM to evaluate.
        tokenizer: Tokenizer used for prompting and decoding generations.
        eval_rows (list): Raw eval rows (dicts with "input", "status").
        args (argparse.Namespace): Parsed CLI args; uses `max_length`.

    Returns:
        dict: Accuracy, macro/weighted F1, precision, recall, AUROC (or
            `None` if undefined), CoT usage rate, and average CoT length in
            words.
    """
    print_banner("EVALUATION")
    model.eval()
    predictions, ground_truths, cot_outputs = [], [], []

    with torch.no_grad():
        for i, row in enumerate(eval_rows):
            if i % 25 == 0:
                print(f"  progress: {i}/{len(eval_rows)}")
            prompt = f"### Instruction:\n{INSTRUCTION}\n\n### Input:\n{row['input'][:1500]}\n\n### Response:\n"
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=args.max_length).to(model.device)
            output = model.generate(**inputs, max_new_tokens=256, do_sample=False, pad_token_id=tokenizer.pad_token_id)
            generated = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

            cot_match = re.search(r"<think>(.*?)</think>", generated, re.DOTALL)
            cot_text = cot_match.group(1).strip() if cot_match else ""
            remaining = (generated[cot_match.end():] if cot_match else generated).strip().lower()
            cot_outputs.append(cot_text)

            if "fake" in remaining:
                pred = 0
            elif "real" in remaining:
                pred = 1
            else:
                pred = 1
            predictions.append(pred)
            ground_truths.append(row["status"])

    accuracy = accuracy_score(ground_truths, predictions)
    f1_macro = f1_score(ground_truths, predictions, average="macro")
    f1_weighted = f1_score(ground_truths, predictions, average="weighted")
    precision = precision_score(ground_truths, predictions, average="macro")
    recall = recall_score(ground_truths, predictions, average="macro")
    try:
        auroc = roc_auc_score(ground_truths, predictions)
    except ValueError:
        auroc = None

    num_with_cot = sum(1 for c in cot_outputs if c)
    avg_cot_length = float(np.mean([len(c.split()) for c in cot_outputs if c])) if num_with_cot else 0.0
    cot_usage_rate = num_with_cot / len(eval_rows) if eval_rows else 0.0

    print(f"Accuracy: {accuracy:.4f}  F1(macro): {f1_macro:.4f}  F1(weighted): {f1_weighted:.4f}  "
          f"Precision: {precision:.4f}  Recall: {recall:.4f}  AUROC: {auroc}")
    print(f"CoT usage rate: {cot_usage_rate:.2%}  Avg CoT length: {avg_cot_length:.1f} words")

    return {
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "precision": precision,
        "recall": recall,
        "auroc": auroc,
        "cot_usage_rate": cot_usage_rate,
        "avg_cot_length_words": avg_cot_length,
    }


def main():
    """Run the full CoT fake-news classification SFT pipeline end to end.

    Parses CLI args, loads the tokenizer/dataset/model (optionally
    quantized/LoRA), either prints a debug batch and exits or trains,
    evaluates, saves the model, and writes a `run_result.json`.
    """
    args = build_arg_parser().parse_args()
    set_all_seeds(args.seed)
    print_config(args, "Text classification SFT -- decoder-only, with CoT")

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
    metrics["cot_enabled"] = True

    save_strategy = args.save_strategy if args.lora else "full"
    save_model(model, tokenizer, args.output_dir, strategy=save_strategy, base_model_name=args.model, is_lora=args.lora)

    write_run_result(
        output_dir=args.output_dir,
        stage="supervised-finetuning",
        task="text_classification",
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
