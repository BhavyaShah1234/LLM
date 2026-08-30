#!/usr/bin/env python3
"""
Text Classification with CoT Fine-tuning Script for Qwen Models
Dataset: domofon/fake_news_cot_reasoning (with full chain-of-thought reasoning)
Format: ### Instruction: ... ### Input: ... ### Response: <think>{reasoning}</think>{label}
Loss: On BOTH thinking tokens AND label tokens
"""

import argparse
import os
import json
import random
import re
from typing import Dict, List, Optional
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import warnings
warnings.filterwarnings("ignore")


def parse_args():
    """Parse command-line arguments for the CoT text-classification fine-tuning run.

    Returns:
        argparse.Namespace: Parsed CLI arguments covering model choice,
        quantization, LoRA, training hyperparameters, data limits, and
        output/logging configuration.
    """
    parser = argparse.ArgumentParser(description="Fine-tune Qwen models for text classification with CoT")
    
    # Model arguments
    parser.add_argument("--model", type=str, required=True,
                        choices=["Qwen/Qwen3-4B-Base", "Qwen/Qwen3-4B", 
                                "Qwen/Qwen3-4B-Thinking-2507", "Qwen/Qwen3-4B-Instruct-2507"],
                        help="Qwen model to fine-tune")
    parser.add_argument("--quantization", type=str, default="4bit",
                        choices=["no", "4bit", "8bit"],
                        help="Quantization type")
    
    # LoRA arguments
    parser.add_argument("--lora", action="store_true", default=False,
                        help="Enable LoRA fine-tuning")
    parser.add_argument("--lora_r", type=int, default=16,
                        help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32,
                        help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                        help="LoRA dropout")
    parser.add_argument("--lora_target_modules", type=str, default="q_proj,k_proj,v_proj,o_proj",
                        help="Comma-separated list of target modules")
    
    # Training arguments
    parser.add_argument("--mixed_precision", type=str, default="bf16",
                        choices=["no", "fp16", "bf16"],
                        help="Mixed precision training")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Per-device training batch size (reduced for CoT)")
    parser.add_argument("--eval_batch_size", type=int, default=2,
                        help="Per-device evaluation batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8,
                        help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=2e-4,
                        help="Learning rate")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--max_length", type=int, default=2048,
                        help="Maximum sequence length (longer for CoT)")
    parser.add_argument("--warmup_steps", type=int, default=100,
                        help="Warmup steps")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay")
    
    # Optimization arguments
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True,
                        help="Enable gradient checkpointing")
    parser.add_argument("--optim", type=str, default="paged_adamw_8bit",
                        help="Optimizer")
    
    # Data arguments
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum training samples")
    
    # Output arguments
    parser.add_argument("--output_dir", type=str, default="./output/text_classification_cot",
                        help="Output directory")
    parser.add_argument("--logging_steps", type=int, default=10,
                        help="Logging frequency")
    parser.add_argument("--eval_steps", type=int, default=500,
                        help="Evaluation frequency")
    parser.add_argument("--save_steps", type=int, default=500,
                        help="Checkpoint save frequency")
    parser.add_argument("--save_total_limit", type=int, default=2,
                        help="Max checkpoints to keep")
    
    # Other arguments
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--debug_first_batch", action="store_true",
                        help="Debug mode")
    
    return parser.parse_args()


class TextClassificationCoTDataset(Dataset):
    """Torch dataset that formats fake-news examples with chain-of-thought reasoning.

    Each item is rendered as an instruction/input/response prompt whose
    response is `<think>{reasoning}</think>{label}`; the prompt span is
    masked out of the labels so loss is computed on both the reasoning and
    the final label tokens.

    Attributes:
        data (List[Dict]): Formatted examples with `"text"`, `"reasoning"`,
            and `"label"` keys.
        tokenizer: Tokenizer used to encode prompt/response text.
        max_length (int): Maximum tokenized sequence length (truncates the
            full prompt+response text).
    """

    def __init__(self, data: List[Dict], tokenizer, max_length: int):
        """Initialize the dataset.

        Args:
            data (List[Dict]): Formatted examples with `"text"`, `"reasoning"`,
                and `"label"` keys.
            tokenizer: Tokenizer used to encode prompt/response text.
            max_length (int): Maximum tokenized sequence length.
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        """Return the number of examples in the dataset.

        Returns:
            int: Number of examples.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """Format and tokenize one example, masking the prompt from the labels.

        Args:
            idx (int): Index of the example to fetch.

        Returns:
            dict: `{"input_ids": list[int], "attention_mask": list[int],
            "labels": list[int]}` with prompt tokens set to `-100` in `labels`.
        """
        item = self.data[idx]
        
        # Format prompt WITH CoT
        instruction = "Classify whether the following text is real news or fake news. Think through your reasoning step by step."
        input_text = item["text"]
        reasoning = item["reasoning"]
        label_text = item["label"]
        
        prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
        # Response includes <think> tags with reasoning AND final label
        response = f"<think>{reasoning}</think>{label_text}"
        full_text = prompt + response
        
        # Tokenize
        prompt_tokens = self.tokenizer(prompt, add_special_tokens=True, truncation=False)
        full_tokens = self.tokenizer(full_text, max_length=self.max_length, 
                                     truncation=True, add_special_tokens=True)
        
        input_ids = full_tokens["input_ids"]
        attention_mask = full_tokens["attention_mask"]
        
        # Create labels with masking (compute loss on BOTH reasoning AND label)
        labels = input_ids.copy()
        prompt_len = len(prompt_tokens["input_ids"])
        labels[:prompt_len] = [-100] * prompt_len  # Mask only the prompt
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }


def load_and_prepare_data(args):
    """Load, format, and split the fake-news CoT dataset.

    Loads `domofon/fake_news_cot_reasoning`, normalizes each example's text,
    reasoning, and label fields, carves off the last 10% as a test split, and
    optionally truncates to `args.max_samples`.

    Args:
        args (argparse.Namespace): Parsed CLI args; uses `args.max_samples`.

    Returns:
        tuple[list[dict], list[dict]]: `(train_data, test_data)`, each a list
        of dicts with `"text"`, `"reasoning"`, and `"label"` keys.
    """
    print("\n" + "="*80)
    print("LOADING DATASET")
    print("="*80)
    
    # Load dataset
    dataset = load_dataset("domofon/fake_news_cot_reasoning")
    
    def format_example(example):
        """Normalize a raw fake/real-news example's text, reasoning, and label.

        Args:
            example (dict): Raw dataset example with a text field (`text`
                or `news_text`), a reasoning field (`reasoning` or
                `explanation`), and a label field (`label` or `class`).

        Returns:
            dict: `{"text", "reasoning", "label"}`, with `text` capped to
            1500 chars, `reasoning` capped to 500 chars (defaulted to a
            generic sentence if empty), and `label` standardized to
            "fake"/"real".
        """
        # Extract text, reasoning, and label
        text = example.get("text", example.get("news_text", ""))
        reasoning = example.get("reasoning", example.get("explanation", ""))
        label = example.get("label", example.get("class", ""))
        
        # Standardize label
        if isinstance(label, int):
            label = "fake" if label == 0 else "real"
        else:
            label = str(label).lower().strip()
            if "fake" in label:
                label = "fake"
            elif "real" in label or "true" in label:
                label = "real"
        
        # Clean reasoning
        reasoning = str(reasoning).strip()
        if not reasoning:
            reasoning = "Based on the content and wording, I need to determine if this is real or fake news."
        
        return {
            "text": text[:1500],  # Limit to save space for reasoning
            "reasoning": reasoning[:500],  # Limit reasoning length
            "label": label
        }
    
    # Format data
    train_data = [format_example(ex) for ex in dataset["train"]]
    
    # Create test split
    test_size = len(train_data) // 10
    test_data = train_data[-test_size:]
    train_data = train_data[:-test_size]
    
    # Limit samples if specified
    if args.max_samples:
        train_data = train_data[:args.max_samples]
        test_data = test_data[:min(args.max_samples // 10, len(test_data))]
    
    print(f"Dataset: domofon/fake_news_cot_reasoning (WITH CoT)")
    print(f"Train samples: {len(train_data)}")
    print(f"Test samples: {len(test_data)}")
    print(f"Format: ### Instruction / ### Input / ### Response: <think>{{reasoning}}</think>{{label}}")
    print(f"Loss computed on: BOTH reasoning and label tokens")
    
    return train_data, test_data


def setup_model_and_tokenizer(args):
    """Load the tokenizer and quantized/LoRA-wrapped causal LM.

    Configures 4-bit/8-bit/no quantization per `args.quantization`, enables
    gradient checkpointing if requested, and wraps the model with LoRA if
    `args.lora` is set.

    Args:
        args (argparse.Namespace): Parsed CLI args; uses `args.model`,
            `args.quantization`, `args.mixed_precision`,
            `args.gradient_checkpointing`, `args.lora`, `args.lora_r`,
            `args.lora_alpha`, `args.lora_dropout`, `args.lora_target_modules`.

    Returns:
        tuple: `(model, tokenizer)`.
    """
    print("\n" + "="*80)
    print("LOADING MODEL AND TOKENIZER")
    print("="*80)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
        use_fast=False
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Configure quantization
    quantization_config = None
    if args.quantization == "4bit":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if args.mixed_precision == "bf16" else torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        print("Quantization: 4-bit (NF4)")
    elif args.quantization == "8bit":
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        print("Quantization: 8-bit")
    else:
        print("Quantization: None")
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if args.mixed_precision == "bf16" else torch.float16
    )
    
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        print("Gradient checkpointing: Enabled")
    
    # Apply LoRA
    if args.lora:
        model = prepare_model_for_kbit_training(model)
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=args.lora_target_modules.split(","),
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        print(f"LoRA: Enabled (r={args.lora_r}, alpha={args.lora_alpha})")
    
    print(f"Model: {args.model}")
    
    return model, tokenizer


def print_examples(train_dataset, tokenizer, num_examples=2):
    """Print a few formatted training examples for a sanity check.

    Decodes the full input text and the label-only (non-masked) span for
    each sampled example.

    Args:
        train_dataset (TextClassificationCoTDataset): Dataset to sample from.
        tokenizer: Tokenizer used to decode token ids back to text.
        num_examples (int): Number of examples to print. Defaults to 2.
    """
    print("\n" + "="*80)
    print("FORMATTED EXAMPLES (WITH CoT)")
    print("="*80)
    
    for i in range(min(num_examples, len(train_dataset))):
        example = train_dataset[i]
        input_ids = example["input_ids"]
        labels = example["labels"]
        
        full_text = tokenizer.decode(input_ids, skip_special_tokens=False)
        label_tokens = [token_id for token_id, label in zip(input_ids, labels) if label != -100]
        label_text = tokenizer.decode(label_tokens, skip_special_tokens=True)
        
        print(f"\n--- Example {i+1} ---")
        print(f"Full text:\n{full_text[:500]}...")
        print(f"\nResponse with CoT (loss computed on):\n{label_text[:300]}...")
        print(f"Note: Loss computed on BOTH <think> content AND final label")


def train_model(model, tokenizer, train_dataset, eval_dataset, args):
    """Fine-tune the model with `Trainer` and save the result.

    Builds `TrainingArguments` from `args`, trains with a seq2seq-style
    padding collator, then saves the model and tokenizer to `args.output_dir`.

    Args:
        model: Model to fine-tune.
        tokenizer: Tokenizer paired with `model`, used by the data collator
            and saved alongside the model.
        train_dataset (TextClassificationCoTDataset): Training dataset.
        eval_dataset (TextClassificationCoTDataset): Evaluation dataset.
        args (argparse.Namespace): Parsed CLI args supplying training
            hyperparameters and `args.output_dir`.

    Returns:
        Trainer: The trainer instance after training completes.
    """
    print("\n" + "="*80)
    print("TRAINING")
    print("="*80)
    
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        fp16=(args.mixed_precision == "fp16"),
        bf16=(args.mixed_precision == "bf16"),
        optim=args.optim,
        gradient_checkpointing=args.gradient_checkpointing,
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        report_to=["none"],
        seed=args.seed,
        ddp_find_unused_parameters=False if args.lora else None,
    )
    
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    
    print(f"\nStarting training with CoT...")
    trainer.train()
    
    print(f"\nSaving model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    return trainer


def evaluate_model(model, tokenizer, test_data, args):
    """Generate predictions on the test set and compute classification metrics.

    Runs greedy generation per example, extracts the `<think>...</think>`
    reasoning span and the final real/fake prediction from the generated
    text, computes accuracy/F1/precision/recall/AUROC plus CoT-usage stats,
    and writes the results to `evaluation_results.json` in `args.output_dir`.

    Args:
        model: Fine-tuned model to evaluate (used in `model.generate`).
        tokenizer: Tokenizer used to build prompts and decode outputs.
        test_data (list[dict]): Held-out examples with `"text"` and `"label"`.
        args (argparse.Namespace): Parsed CLI args; uses `args.max_length`
            and `args.output_dir`.

    Returns:
        dict: Metrics dict (accuracy, f1_macro, f1_weighted, precision,
        recall, auroc, num_test_samples, cot_enabled, cot_usage_rate,
        avg_cot_length_words) — the same dict written to disk.
    """
    print("\n" + "="*80)
    print("EVALUATION")
    print("="*80)
    
    model.eval()
    predictions = []
    ground_truths = []
    cot_outputs = []
    
    label_map = {"fake": 0, "real": 1}
    
    print(f"Running inference on {len(test_data)} test samples...")
    
    with torch.no_grad():
        for i, item in enumerate(test_data):
            if i % 100 == 0:
                print(f"  Progress: {i}/{len(test_data)}")
            
            instruction = "Classify whether the following text is real news or fake news. Think through your reasoning step by step."
            input_text = item["text"]
            prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
            
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=args.max_length)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,  # Allow more tokens for CoT
                temperature=0.1,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )
            
            generated_text = tokenizer.decode(outputs[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
            
            # Extract CoT and final answer
            cot_match = re.search(r'<think>(.*?)</think>', generated_text, re.DOTALL)
            if cot_match:
                cot_text = cot_match.group(1).strip()
                remaining = generated_text[cot_match.end():].strip()
            else:
                cot_text = ""
                remaining = generated_text.strip()
            
            cot_outputs.append(cot_text)
            
            # Extract prediction
            remaining_lower = remaining.lower()
            if "fake" in remaining_lower:
                pred_label = 0
            elif "real" in remaining_lower or "true" in remaining_lower:
                pred_label = 1
            else:
                pred_label = 1  # Default
            
            predictions.append(pred_label)
            ground_truths.append(label_map[item["label"]])
    
    # Compute metrics
    accuracy = accuracy_score(ground_truths, predictions)
    f1_macro = f1_score(ground_truths, predictions, average="macro")
    f1_weighted = f1_score(ground_truths, predictions, average="weighted")
    precision = precision_score(ground_truths, predictions, average="macro")
    recall = recall_score(ground_truths, predictions, average="macro")
    
    try:
        auroc = roc_auc_score(ground_truths, predictions)
    except:
        auroc = None
    
    # Analyze CoT quality
    num_with_cot = sum(1 for cot in cot_outputs if cot)
    avg_cot_length = np.mean([len(cot.split()) for cot in cot_outputs if cot]) if num_with_cot > 0 else 0
    
    print("\n" + "="*80)
    print("RESULTS (WITH CoT)")
    print("="*80)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 (Macro): {f1_macro:.4f}")
    print(f"F1 (Weighted): {f1_weighted:.4f}")
    print(f"Precision (Macro): {precision:.4f}")
    print(f"Recall (Macro): {recall:.4f}")
    if auroc is not None:
        print(f"AUROC: {auroc:.4f}")
    print(f"\nCoT Analysis:")
    print(f"  Samples with CoT: {num_with_cot}/{len(test_data)} ({100*num_with_cot/len(test_data):.1f}%)")
    print(f"  Avg CoT length: {avg_cot_length:.1f} words")
    
    results = {
        "accuracy": float(accuracy),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "precision": float(precision),
        "recall": float(recall),
        "auroc": float(auroc) if auroc is not None else None,
        "num_test_samples": len(test_data),
        "cot_enabled": True,
        "cot_usage_rate": float(num_with_cot / len(test_data)),
        "avg_cot_length_words": float(avg_cot_length)
    }
    
    results_path = os.path.join(args.output_dir, "evaluation_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")
    
    return results


def main():
    """Run the end-to-end CoT text-classification fine-tuning pipeline.

    Parses args, seeds RNGs, loads data and model, builds datasets, prints
    sample formatting, optionally exits early in debug mode, then trains
    and evaluates the model.
    """
    args = parse_args()

    # Set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Print configuration
    print("\n" + "="*80)
    print("CONFIGURATION")
    print("="*80)
    print(f"Task: Text Classification WITH CoT")
    print(f"Model: {args.model}")
    print(f"Quantization: {args.quantization}")
    print(f"LoRA: {args.lora}")
    if args.lora:
        print(f"  - Rank: {args.lora_r}")
        print(f"  - Alpha: {args.lora_alpha}")
    print(f"Mixed Precision: {args.mixed_precision}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Effective Batch Size: {args.batch_size * args.gradient_accumulation_steps}")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"Epochs: {args.epochs}")
    print(f"Max Length: {args.max_length}")
    
    # Load data
    train_data, test_data = load_and_prepare_data(args)
    
    # Setup model
    model, tokenizer = setup_model_and_tokenizer(args)
    
    # Create datasets
    train_dataset = TextClassificationCoTDataset(train_data, tokenizer, args.max_length)
    eval_dataset = TextClassificationCoTDataset(test_data[:50], tokenizer, args.max_length)
    
    # Print examples
    print_examples(train_dataset, tokenizer, num_examples=2)
    
    if args.debug_first_batch:
        print("\nDEBUG MODE: Exiting")
        return
    
    # Train
    trainer = train_model(model, tokenizer, train_dataset, eval_dataset, args)
    
    # Evaluate
    results = evaluate_model(model, tokenizer, test_data, args)
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()
