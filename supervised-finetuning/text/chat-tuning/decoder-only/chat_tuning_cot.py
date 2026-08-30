"""Chat tuning SFT -- decoder-only, with CoT.

Dataset: PJMixers-Dev/oumi-ai_lmsys_chat_1m_clean_R1-1k-think-1k-response-ShareGPT.
ShareGPT-style `conversations` field ({"from": "human"|"gpt", "value": ...}),
verified live -- matches what the archived version of this script already
assumed, so no field-name fix needed here. The `<think>...</think>` content
is already embedded directly inside each assistant turn's `value` text (this
dataset was curated from DeepSeek-R1 outputs specifically to include it), so
this script does NOT need to add `<think>` tags itself -- unlike the CoT
scripts for other tasks in this project. What IS fixed: the archived version
of this script (old/chat_tuning_cot.py) built its prompt string with a
literal `\\n` instead of a real newline (confirmed via `cat -A`), had no
evaluation function, and its --debug_first_batch returned before loading any
data.

Uses the first human/gpt exchange per conversation (single-exchange
simplification, same scope as chat_tuning_standard.py).

Format: SYSTEM: ... \\nUSER: {user}\\nASSISTANT: {response_with_embedded_think}
Loss: assistant tokens only (which already include the <think> content).

Usage:
    python chat_tuning_cot.py --debug_first_batch --max_samples 20
"""

import argparse
import random as _random
import re
import time

import numpy as np
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

DATASET_NAME = "PJMixers-Dev/oumi-ai_lmsys_chat_1m_clean_R1-1k-think-1k-response-ShareGPT"
ARCHITECTURE = "decoder-only"
SYSTEM_MESSAGE = "You are a helpful AI assistant."
MAX_RAW_EXAMPLES = 10000  # dataset is large; cap the streamed materialization


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for this script.

    Returns:
        argparse.ArgumentParser: Parser covering model/quantization/LoRA,
        optimization, data-selection, output, and debug flags.
    """
    p = argparse.ArgumentParser(description="SFT a decoder-only model for chat tuning (with CoT).")

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

    p.add_argument("--max_samples", type=int, default=-1, help="Number of training rows to use (from the first MAX_RAW_EXAMPLES streamed). -1 (default) = use all of them.")
    p.add_argument("--sample_selection", type=str, default="random", choices=["random", "first", "last"], help="Which rows to pick when --max_samples is positive. Default: random.")
    p.add_argument("--max_eval_samples", type=int, default=50, help="Eval rows to use (CoT generation is slow). Default: 50.")

    p.add_argument("--output_dir", type=str, default="./output/supervised-finetuning/text/chat-tuning/decoder-only/cot", help="Output directory.")
    p.add_argument("--logging_steps", type=int, default=10, help="Logging frequency. Default: 10.")
    p.add_argument("--eval_steps", type=int, default=200, help="Evaluation frequency. Default: 200.")
    p.add_argument("--save_steps", type=int, default=200, help="Checkpoint save frequency. Default: 200.")
    p.add_argument("--save_strategy", type=str, default="adapter_only", choices=["full", "adapter_only", "merged", "adapter_and_merged", "base_reference"], help="Model saving strategy. Default: adapter_only when --lora, otherwise 'full'.")

    p.add_argument("--seed", type=int, default=42, help="Random seed. Default: 42.")
    p.add_argument("--debug_first_batch", action="store_true", default=False, help="Print formatted examples and exit without training.")

    return p


def extract_exchange(conversations):
    """Pull the first human/gpt turn pair out of a ShareGPT-style conversation.

    Args:
        conversations: List of turn dicts with `from` (`"human"`/`"gpt"`)
            and `value` keys.

    Returns:
        tuple or None: `(user_msg, asst_msg)` for the first complete
        human/gpt exchange, or `None` if the conversation never produces
        both a human and a gpt turn.
    """
    user_msg, asst_msg = "", ""
    for turn in conversations:
        role = turn.get("from")
        if role == "human" and not user_msg:
            user_msg = turn.get("value", "")
        elif role == "gpt" and not asst_msg:
            asst_msg = turn.get("value", "")
        if user_msg and asst_msg:
            break
    if not user_msg or not asst_msg:
        return None
    return user_msg, asst_msg


class ChatCoTDataset(Dataset):
    """Torch dataset that formats extracted exchanges into CoT-wrapped chat prompts.

    Each item is tokenized, with the prompt span masked out of `labels` so
    loss is only computed on the assistant's response (which already embeds
    its own `<think>...</think>` reasoning text).

    Attributes:
        rows: List of `(user_msg, asst_msg)` tuples.
        tokenizer: The model's tokenizer.
        max_length: Max token length for truncation.
    """

    def __init__(self, rows, tokenizer, max_length: int):
        """Initialize the dataset.

        Args:
            rows: List of `(user_msg, asst_msg)` tuples.
            tokenizer: The model's tokenizer.
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
        """Build the tokenized, label-masked training example for one row.

        Args:
            idx: Index of the row to fetch.

        Returns:
            dict: `input_ids`, `attention_mask`, and `labels` (prompt span
            set to -100) for the example.
        """
        user_msg, asst_msg = self.rows[idx]
        prompt = f"SYSTEM: {SYSTEM_MESSAGE}\nUSER: {user_msg}\nASSISTANT: "
        full_text = prompt + asst_msg  # asst_msg already contains <think>...</think>

        prompt_ids = self.tokenizer(prompt, add_special_tokens=True, truncation=False)["input_ids"]
        full = self.tokenizer(full_text, max_length=self.max_length, truncation=True, add_special_tokens=True)

        input_ids = full["input_ids"]
        labels = input_ids.copy()
        prompt_len = min(len(prompt_ids), len(input_ids))
        labels[:prompt_len] = [-100] * prompt_len

        return {"input_ids": input_ids, "attention_mask": full["attention_mask"], "labels": labels}


def verify_dataset() -> None:
    """Stream one example from DATASET_NAME and sanity-check its fields.

    Asserts that a `conversations` field is present and that an exchange
    can be extracted from it, and prints a preview. Not called from
    `main()` by default (see the comment there) but kept for manual
    verification.

    Raises:
        AssertionError: If the `conversations` field is missing.
    """
    print_banner("VERIFYING DATASET")
    peek = load_dataset(DATASET_NAME, split="train", streaming=True)
    example = next(iter(peek))
    assert "conversations" in example, f"Expected a 'conversations' field in {DATASET_NAME}, got fields: {list(example.keys())}"
    exchange = extract_exchange(example["conversations"])
    print(f"Dataset: {DATASET_NAME}")
    print(f"Fields: {list(example.keys())}")
    print(f"Sample exchange extracted: {exchange is not None}")
    if exchange:
        print(f"Assistant turn contains <think>: {'<think>' in exchange[1]}")
    print()


def load_and_prepare_data(args, tokenizer):
    """Stream, extract, split, and subsample exchanges into train/eval sets.

    Args:
        args: Parsed CLI namespace; uses `max_samples`, `sample_selection`,
            `max_eval_samples`, and `seed`.
        tokenizer: The model's tokenizer, passed through to the returned
            datasets.

    Returns:
        tuple: `(train_dataset, eval_dataset, eval_rows)` where the first
        two are `ChatCoTDataset` instances and `eval_rows` is the raw list
        of eval-split `(user_msg, asst_msg)` tuples (used later for
        generation-based evaluation).
    """
    print_banner("LOADING DATASET")
    raw = load_dataset(DATASET_NAME, split="train", streaming=True)
    exchanges = []
    for _, row in zip(range(MAX_RAW_EXAMPLES), raw):
        exchange = extract_exchange(row["conversations"])
        if exchange is not None:
            exchanges.append(exchange)

    n = len(exchanges)
    test_size = max(1, n // 10)
    eval_rows = exchanges[-test_size:]
    train_rows = exchanges[:-test_size]

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

    return ChatCoTDataset(train_rows, tokenizer, args.max_length), ChatCoTDataset(eval_rows, tokenizer, args.max_length), eval_rows


def decode_example(example, index, tokenizer):
    """Decode one tokenized example into a human-readable debug string.

    Args:
        example: A tokenized example dict as returned by
            `ChatCoTDataset.__getitem__`.
        index: The example's index (unused, kept for a uniform
            `decode_fn` signature with `common.logging_utils.print_formatted_examples`).
        tokenizer: The model's tokenizer, used to decode token ids.

    Returns:
        str: Formatted preview of the full text and the loss-bearing
        assistant response.
    """
    input_ids = example["input_ids"]
    labels = example["labels"]
    full_text = tokenizer.decode(input_ids, skip_special_tokens=False)
    label_tokens = [tid for tid, lab in zip(input_ids, labels) if lab != -100]
    label_text = tokenizer.decode(label_tokens, skip_special_tokens=True)
    return f"Full text:\n{full_text[:600]}...\n\nAssistant response (loss computed on, incl. <think>): {label_text[:500]}"


def evaluate_model(model, tokenizer, eval_rows, args):
    """Generate assistant responses for the eval set and score them.

    Args:
        model: The trained (or base) causal LM to generate from.
        tokenizer: The model's tokenizer, used to build prompts and decode
            generations.
        eval_rows: Raw `(user_msg, asst_msg)` tuples (as returned by
            `load_and_prepare_data`) to evaluate on.
        args: Parsed CLI namespace; uses `max_length`.

    Returns:
        dict: `rouge_l`, `bertscore_f1` (or None if scoring failed),
        `cot_enabled`, `cot_usage_rate`, and `avg_cot_length_words`.
    """
    print_banner("EVALUATION")
    model.eval()
    predictions, references, cot_outputs = [], [], []

    with torch.no_grad():
        for i, (user_msg, asst_msg) in enumerate(eval_rows):
            if i % 10 == 0:
                print(f"  progress: {i}/{len(eval_rows)}")
            prompt = f"SYSTEM: {SYSTEM_MESSAGE}\nUSER: {user_msg}\nASSISTANT: "
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=args.max_length).to(model.device)
            output = model.generate(**inputs, max_new_tokens=384, do_sample=False, pad_token_id=tokenizer.pad_token_id)
            generated = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

            cot_match = re.search(r"<think>(.*?)</think>", generated, re.DOTALL)
            cot_text = cot_match.group(1).strip() if cot_match else ""
            remaining = (generated[cot_match.end():] if cot_match else generated).strip()
            cot_outputs.append(cot_text)

            predictions.append(remaining or " ")
            ref_match = re.search(r"</think>(.*)", asst_msg, re.DOTALL)
            references.append((ref_match.group(1).strip() if ref_match else asst_msg.strip()) or " ")

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge_l_scores = [scorer.score(ref, pred)["rougeL"].fmeasure for pred, ref in zip(predictions, references)]
    rouge_l = sum(rouge_l_scores) / len(rouge_l_scores) if rouge_l_scores else 0.0

    try:
        _, _, bert_f1 = bert_score(predictions, references, lang="en", verbose=False)
        bertscore_f1 = float(bert_f1.mean())
    except Exception as e:
        print(f"  BERTScore failed ({e}), skipping.")
        bertscore_f1 = None

    num_with_cot = sum(1 for c in cot_outputs if c)
    avg_cot_length = float(np.mean([len(c.split()) for c in cot_outputs if c])) if num_with_cot else 0.0
    cot_usage_rate = num_with_cot / len(eval_rows) if eval_rows else 0.0

    print(f"ROUGE-L: {rouge_l:.4f}  BERTScore(F1): {bertscore_f1}")
    print(f"CoT usage rate: {cot_usage_rate:.2%}  Avg CoT length: {avg_cot_length:.1f} words")
    return {
        "rouge_l": rouge_l,
        "bertscore_f1": bertscore_f1,
        "cot_enabled": True,
        "cot_usage_rate": cot_usage_rate,
        "avg_cot_length_words": avg_cot_length,
    }


def main():
    """Run the full CoT chat-tuning SFT pipeline: load, train, evaluate, save, record.

    Parses CLI args, loads and preprocesses the dataset, loads the base
    causal LM (optionally quantized/LoRA-adapted), either dumps formatted
    debug examples and exits or trains via `Trainer`, evaluates the result,
    saves the model, and writes a `run_result.json`.
    """
    args = build_arg_parser().parse_args()
    set_all_seeds(args.seed)
    print_config(args, "Chat tuning SFT -- decoder-only, with CoT")

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
        task="chat_tuning",
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
