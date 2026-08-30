#!/usr/bin/env python3
"""
Standard Text Classification Fine-tuning Script for Qwen Models
Dataset: Tohrumi/glue_sst2_10k (sentence -> sentiment label)
Format: ### Instruction: ... ### Input: ... ### Response: {label}
Loss: Only on label tokens
"""

import argparse
import os
import json
import random
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
from sklearn.preprocessing import LabelBinarizer
import warnings
warnings.filterwarnings("ignore")


def parse_args():
    """Parse command-line arguments for the standard text-classification fine-tuning run.

    Returns:
        argparse.Namespace: Parsed CLI arguments covering model choice,
        quantization, LoRA, training hyperparameters, data limits, and
        output/logging configuration.
    """
    parser = argparse.ArgumentParser(description="Fine-tune Qwen models for text classification")
    
    # Model arguments
    parser.add_argument("--model", type=str, required=True,
                        choices=["Qwen/Qwen3-4B-Base", "Qwen/Qwen3-4B", 
                                "Qwen/Qwen3-4B-Thinking-2507", "Qwen/Qwen3-4B-Instruct-2507"],
                        help="Qwen model to fine-tune")
    parser.add_argument("--quantization", type=str, default="4bit",
                        choices=["no", "4bit", "8bit"],
                        help="Quantization type (4bit recommended for 8GB GPU)")
    
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
                        help="Comma-separated list of target modules for LoRA")
    
    # Training arguments
    parser.add_argument("--mixed_precision", type=str, default="bf16",
                        choices=["no", "fp16", "bf16"],
                        help="Mixed precision training")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Per-device training batch size")
    parser.add_argument("--eval_batch_size", type=int, default=4,
                        help="Per-device evaluation batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=2e-4,
                        help="Learning rate")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--max_length", type=int, default=512,
                        help="Maximum sequence length")
    parser.add_argument("--warmup_steps", type=int, default=100,
                        help="Warmup steps")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay")
    
    # Optimization arguments
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True,
                        help="Enable gradient checkpointing")
    parser.add_argument("--optim", type=str, default="paged_adamw_8bit",
                        help="Optimizer (paged_adamw_8bit for memory efficiency)")
    
    # Data arguments
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum training samples (for testing)")
    parser.add_argument("--test_size", type=float, default=0.1,
                        help="Test set fraction")
    
    # Output arguments
    parser.add_argument("--output_dir", type=str, default="./output/text_classification_standard",
                        help="Output directory")
    parser.add_argument("--logging_steps", type=int, default=10,
                        help="Logging frequency")
    parser.add_argument("--eval_steps", type=int, default=500,
                        help="Evaluation frequency")
    parser.add_argument("--save_steps", type=int, default=500,
                        help="Checkpoint save frequency")
    parser.add_argument("--save_total_limit", type=int, default=2,
                        help="Max number of checkpoints to keep")
    
    # Other arguments
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--debug_first_batch", action="store_true",
                        help="Debug: print first batch and exit")
    
    return parser.parse_args()


class TextClassificationDataset(Dataset):
    """Torch dataset that formats SST-2 sentiment examples with loss masking.

    Each item is rendered as an instruction/input/response prompt whose
    response is the sentiment label; the prompt span is masked out of the
    labels so loss is computed only on the label tokens.

    Attributes:
        data (List[Dict]): Formatted examples with `"sentence"` and `"label"` keys.
        tokenizer: Tokenizer used to encode prompt/response text.
        max_length (int): Maximum tokenized sequence length.
        label_map (Dict): Mapping from integer class id to label text (e.g.
            `{0: "negative", 1: "positive"}`).
    """

    def __init__(self, data: List[Dict], tokenizer, max_length: int, label_map: Dict):
        """Initialize the dataset.

        Args:
            data (List[Dict]): Formatted examples with `"sentence"` and
                `"label"` keys.
            tokenizer: Tokenizer used to encode prompt/response text.
            max_length (int): Maximum tokenized sequence length.
            label_map (Dict): Mapping from integer class id to label text.
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_map = label_map

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
        
        # Format prompt
        instruction = "Classify the sentiment of the following text as positive or negative."
        input_text = item["sentence"]
        label_text = item["label"]
        
        prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
        full_text = prompt + label_text
        
        # Tokenize
        prompt_tokens = self.tokenizer(prompt, add_special_tokens=True, truncation=False)
        full_tokens = self.tokenizer(full_text, max_length=self.max_length, 
                                     truncation=True, add_special_tokens=True)
        
        input_ids = full_tokens["input_ids"]
        attention_mask = full_tokens["attention_mask"]
        
        # Create labels with masking (only compute loss on label tokens)
        prompt_len = len(prompt_tokens["input_ids"])
        labels = [-100] * prompt_len + input_ids[prompt_len:]  # Mask prompt, keep label
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }


def load_and_prepare_data(args):
    """Load, format, and split the GLUE SST-2 sentiment dataset.

    Loads `Tohrumi/glue_sst2_10k`, maps integer labels to text, picks the
    first available eval-like split (`validation`/`eval`/`test`, falling
    back to a 90/10 split of train), and optionally truncates to
    `args.max_samples`.

    Args:
        args (argparse.Namespace): Parsed CLI args; uses `args.max_samples`.

    Returns:
        tuple[list[dict], list[dict], dict]: `(train_data, test_data,
        label_map)`, where each data list holds dicts with `"sentence"` and
        `"label"` (text) keys, and `label_map` maps int id to label text.
    """
    print("\n" + "="*80)
    print("LOADING DATASET")
    print("="*80)
    
    # Load dataset
    dataset = load_dataset("Tohrumi/glue_sst2_10k")
    
    # Map labels to text
    label_map = {0: "negative", 1: "positive"}
    
    def format_example(example):
        """Map a raw example's integer label to its class name via `label_map`.

        Args:
            example (dict): Raw dataset example with `sentence` and `label`
                (integer class index).

        Returns:
            dict: `{"sentence", "label"}` with `label` replaced by the
            class name string.
        """
        return {
            "sentence": example["sentence"],
            "label": label_map[example["label"]]
        }
    
    # Format data
    train_data = [format_example(ex) for ex in dataset["train"]]
    
    # Use eval or validation split depending on which exists
    if "validation" in dataset:
        test_data = [format_example(ex) for ex in dataset["validation"]]
    elif "eval" in dataset:
        test_data = [format_example(ex) for ex in dataset["eval"]]
    elif "test" in dataset:
        test_data = [format_example(ex) for ex in dataset["test"]]
    else:
        # If no separate test set, split training data
        split_idx = int(len(train_data) * 0.9)
        test_data = train_data[split_idx:]
        train_data = train_data[:split_idx]
    
    # Limit samples if specified
    if args.max_samples:
        train_data = train_data[:args.max_samples]
        test_data = test_data[:min(args.max_samples // 5, len(test_data))]
    
    print(f"Dataset: Tohrumi/glue_sst2_10k")
    print(f"Train samples: {len(train_data)}")
    print(f"Test samples: {len(test_data)}")
    print(f"Label mapping: {label_map}")
    
    return train_data, test_data, label_map


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
    
    # Ensure tokenizer has pad token
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
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True
        )
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
    
    # Enable gradient checkpointing
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
    print(f"Device map: {model.hf_device_map if hasattr(model, 'hf_device_map') else 'N/A'}")
    
    return model, tokenizer


def print_examples(train_dataset, tokenizer, num_examples=2):
    """Print a few formatted training examples for a sanity check.

    Decodes the full input text and the label-only (non-masked) span for
    each sampled example, and prints token-length stats.

    Args:
        train_dataset (TextClassificationDataset): Dataset to sample from.
        tokenizer: Tokenizer used to decode token ids back to text.
        num_examples (int): Number of examples to print. Defaults to 2.
    """
    print("\n" + "="*80)
    print("FORMATTED EXAMPLES")
    print("="*80)
    
    for i in range(min(num_examples, len(train_dataset))):
        example = train_dataset[i]
        input_ids = example["input_ids"]
        labels = example["labels"]
        
        # Decode full text
        full_text = tokenizer.decode(input_ids, skip_special_tokens=False)
        
        # Decode only label part (non-masked)
        label_tokens = [token_id for token_id, label in zip(input_ids, labels) if label != -100]
        label_text = tokenizer.decode(label_tokens, skip_special_tokens=True)
        
        print(f"\n--- Example {i+1} ---")
        print(f"Full text:\n{full_text}")
        print(f"\nLabel only (loss computed on): {label_text}")
        print(f"Input length: {len(input_ids)} tokens")
        print(f"Label length: {len(label_tokens)} tokens")


def train_model(model, tokenizer, train_dataset, eval_dataset, args):
    """Fine-tune the model with `Trainer` and save the result.

    Builds `TrainingArguments` from `args`, trains with a seq2seq-style
    padding collator, then saves the model and tokenizer to `args.output_dir`.

    Args:
        model: Model to fine-tune.
        tokenizer: Tokenizer paired with `model`, used by the data collator
            and saved alongside the model.
        train_dataset (TextClassificationDataset): Training dataset.
        eval_dataset (TextClassificationDataset): Evaluation dataset.
        args (argparse.Namespace): Parsed CLI args supplying training
            hyperparameters and `args.output_dir`.

    Returns:
        Trainer: The trainer instance after training completes.
    """
    print("\n" + "="*80)
    print("TRAINING")
    print("="*80)
    
    # Training arguments
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
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    
    # Train
    print(f"\nStarting training...")
    print(f"Total steps: {len(train_dataset) // (args.batch_size * args.gradient_accumulation_steps) * args.epochs}")
    print(f"Effective batch size: {args.batch_size * args.gradient_accumulation_steps * torch.cuda.device_count()}")
    
    trainer.train()
    
    # Save final model
    print(f"\nSaving model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    return trainer


def evaluate_model(model, tokenizer, test_data, label_map, args):
    """Generate predictions on the test set and compute classification metrics.

    Runs short greedy generation per example, extracts the positive/negative
    prediction from the generated text, computes accuracy/F1/precision/
    recall/AUROC, and writes the results to `evaluation_results.json` in
    `args.output_dir`.

    Args:
        model: Fine-tuned model to evaluate (used in `model.generate`).
        tokenizer: Tokenizer used to build prompts and decode outputs.
        test_data (list[dict]): Held-out examples with `"sentence"` and
            `"label"` (text) keys.
        label_map (dict): Mapping from integer class id to label text, used
            to build the reverse map back to integer ground truth.
        args (argparse.Namespace): Parsed CLI args; uses `args.max_length`
            and `args.output_dir`.

    Returns:
        dict: Metrics dict (accuracy, f1_macro, f1_weighted, precision,
        recall, auroc, num_test_samples) — the same dict written to disk.
    """
    print("\n" + "="*80)
    print("EVALUATION")
    print("="*80)
    
    model.eval()
    predictions = []
    ground_truths = []
    
    # Create reverse label map
    reverse_label_map = {v: k for k, v in label_map.items()}
    
    print(f"Running inference on {len(test_data)} test samples...")
    
    with torch.no_grad():
        for i, item in enumerate(test_data):
            if i % 100 == 0:
                print(f"  Progress: {i}/{len(test_data)}")
            
            # Format prompt
            instruction = "Classify the sentiment of the following text as positive or negative."
            input_text = item["sentence"]
            prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
            
            # Tokenize
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=args.max_length)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            # Generate
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                temperature=0.1,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )
            
            # Decode
            generated_text = tokenizer.decode(outputs[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
            
            # Extract predicted label
            generated_text = generated_text.strip().lower()
            if "positive" in generated_text:
                pred_label = 1
            elif "negative" in generated_text:
                pred_label = 0
            else:
                # Default to most common class if unclear
                pred_label = 1
            
            predictions.append(pred_label)
            ground_truths.append(reverse_label_map[item["label"]])
    
    # Compute metrics
    accuracy = accuracy_score(ground_truths, predictions)
    f1_macro = f1_score(ground_truths, predictions, average="macro")
    f1_weighted = f1_score(ground_truths, predictions, average="weighted")
    precision = precision_score(ground_truths, predictions, average="macro")
    recall = recall_score(ground_truths, predictions, average="macro")
    
    # Compute AUROC (for binary classification)
    try:
        auroc = roc_auc_score(ground_truths, predictions)
    except:
        auroc = None
    
    # Print results
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 (Macro): {f1_macro:.4f}")
    print(f"F1 (Weighted): {f1_weighted:.4f}")
    print(f"Precision (Macro): {precision:.4f}")
    print(f"Recall (Macro): {recall:.4f}")
    if auroc is not None:
        print(f"AUROC: {auroc:.4f}")
    
    # Save results
    results = {
        "accuracy": float(accuracy),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "precision": float(precision),
        "recall": float(recall),
        "auroc": float(auroc) if auroc is not None else None,
        "num_test_samples": len(test_data)
    }
    
    results_path = os.path.join(args.output_dir, "evaluation_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")
    
    return results


def main():
    """Run the end-to-end standard text-classification fine-tuning pipeline.

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
    print(f"Model: {args.model}")
    print(f"Quantization: {args.quantization}")
    print(f"LoRA: {args.lora}")
    if args.lora:
        print(f"  - Rank: {args.lora_r}")
        print(f"  - Alpha: {args.lora_alpha}")
        print(f"  - Dropout: {args.lora_dropout}")
        print(f"  - Target modules: {args.lora_target_modules}")
    print(f"Mixed Precision: {args.mixed_precision}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Gradient Accumulation: {args.gradient_accumulation_steps}")
    print(f"Effective Batch Size: {args.batch_size * args.gradient_accumulation_steps}")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"Epochs: {args.epochs}")
    print(f"Max Length: {args.max_length}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Seed: {args.seed}")
    
    # Load data
    train_data, test_data, label_map = load_and_prepare_data(args)
    
    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(args)
    
    # Create datasets
    train_dataset = TextClassificationDataset(train_data, tokenizer, args.max_length, label_map)
    eval_dataset = TextClassificationDataset(test_data[:100], tokenizer, args.max_length, label_map)  # Small eval set for speed
    
    # Print examples
    print_examples(train_dataset, tokenizer, num_examples=2)
    
    # Debug mode
    if args.debug_first_batch:
        print("\nDEBUG MODE: Exiting after printing examples")
        return
    
    # Train
    trainer = train_model(model, tokenizer, train_dataset, eval_dataset, args)
    
    # Evaluate
    results = evaluate_model(model, tokenizer, test_data, label_map, args)
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()
