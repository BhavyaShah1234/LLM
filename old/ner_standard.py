#!/usr/bin/env python3
"""
Standard NER (Named Entity Recognition) Fine-tuning Script for Qwen Models
Dataset: MorryShah/complex_ner
Format: ### Instruction: ... ### Input: ... ### Response: {entities_json}
Loss: Only on entity JSON tokens
"""

import argparse
import os
import json
import random
from typing import Dict, List
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer,
    BitsAndBytesConfig, DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
from sklearn.metrics import f1_score, precision_score, recall_score
import warnings
warnings.filterwarnings("ignore")


def parse_args():
    """Parse CLI arguments for the standard NER fine-tuning run.

    Returns:
        argparse.Namespace: Parsed arguments covering model choice, quantization/LoRA
        setup, optimization hyperparameters, and I/O paths.
    """
    parser = argparse.ArgumentParser(description="Fine-tune Qwen models for NER")
    parser.add_argument("--model", type=str, required=True,
                        choices=["Qwen/Qwen3-4B-Base", "Qwen/Qwen3-4B", 
                                "Qwen/Qwen3-4B-Thinking-2507", "Qwen/Qwen3-4B-Instruct-2507"])
    parser.add_argument("--quantization", type=str, default="4bit", choices=["no", "4bit", "8bit"])
    parser.add_argument("--lora", action="store_true", default=False)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_target_modules", type=str, default="q_proj,k_proj,v_proj,o_proj")
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--optim", type=str, default="paged_adamw_8bit")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default="./output/ner_standard")
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--debug_first_batch", action="store_true")
    return parser.parse_args()


class NERDataset(Dataset):
    """Torch dataset that formats NER examples as instruction/response pairs
    with loss masked to the extracted entity JSON only.

    Attributes:
        data (List[Dict]): Formatted examples, each with `text` and `entities`.
        tokenizer: Tokenizer used to encode prompt and full text.
        max_length (int): Max token length for the encoded example.
    """

    def __init__(self, data: List[Dict], tokenizer, max_length: int):
        """Store the formatted examples and tokenization settings.

        Args:
            data (List[Dict]): Formatted NER examples.
            tokenizer: Tokenizer used to encode prompt and full text.
            max_length (int): Max token length for the encoded example.
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
        """Build the tokenized, label-masked training example at `idx`.

        Args:
            idx (int): Index into `self.data`.

        Returns:
            dict: `input_ids`, `attention_mask`, and `labels` (prompt tokens masked
            to -100 so loss is computed on the entity JSON only).
        """
        item = self.data[idx]
        
        instruction = "Extract all named entities from the text and provide them in JSON format with entity text and label."
        input_text = item["text"]
        entities_json = json.dumps(item["entities"], ensure_ascii=False)
        
        prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
        full_text = prompt + entities_json
        
        prompt_tokens = self.tokenizer(prompt, add_special_tokens=True, truncation=False)
        full_tokens = self.tokenizer(full_text, max_length=self.max_length, 
                                     truncation=True, add_special_tokens=True)
        
        input_ids = full_tokens["input_ids"]
        attention_mask = full_tokens["attention_mask"]
        labels = input_ids.copy()
        prompt_len = len(prompt_tokens["input_ids"])
        labels[:prompt_len] = [-100] * prompt_len
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }


def load_and_prepare_data(args):
    """Load the MorryShah/complex_ner dataset and reformat its BIO-tagged tokens
    into text/entities examples for standard NER training.

    Args:
        args (argparse.Namespace): Parsed CLI args; uses `max_samples` to optionally
            cap the number of train/test examples.

    Returns:
        tuple[list[dict], list[dict]]: `(train_data, test_data)`, each a list of
        `{"text", "entities"}` dicts.
    """
    print("\n" + "="*80)
    print("LOADING DATASET")
    print("="*80)

    dataset = load_dataset("MorryShah/complex_ner")

    def format_example(example):
        """Convert BIO-tagged tokens into joined text and grouped entity spans.

        Args:
            example (dict): Raw dataset example with `tokens` and `ner_tags`
                (either string BIO tags or numeric tag ids).

        Returns:
            dict: `{"text", "entities"}` — entities are `{"entity", "label"}`
            dicts assembled from consecutive B-/I- tagged tokens, capped to 20.
        """
        # Extract entities from NER dataset format
        tokens = example.get("tokens", [])
        ner_tags = example.get("ner_tags", [])
        
        text = " ".join(tokens)
        entities = []
        
        current_entity = {"entity": "", "label": ""}
        for i, (token, tag) in enumerate(zip(tokens, ner_tags)):
            if isinstance(tag, int):
                tag_name = f"TAG_{tag}"  # Map numeric tags to names
            else:
                tag_name = str(tag)
            
            if tag_name.startswith("B-"):
                if current_entity["entity"]:
                    entities.append(current_entity)
                current_entity = {"entity": token, "label": tag_name[2:]}
            elif tag_name.startswith("I-") and current_entity["entity"]:
                current_entity["entity"] += " " + token
            else:
                if current_entity["entity"]:
                    entities.append(current_entity)
                    current_entity = {"entity": "", "label": ""}
        
        if current_entity["entity"]:
            entities.append(current_entity)
        
        return {"text": text, "entities": entities[:20]}  # Limit entities
    
    train_data = [format_example(ex) for ex in dataset["train"]]
    # Get test split with fallback
    if "test" in dataset:
        test_split = dataset["test"]
    elif "validation" in dataset:
        test_split = dataset["validation"]
    else:
        test_split = dataset["train"]
    
    # Properly handle HuggingFace Dataset slicing
    test_limit = min(500, len(test_split))
    if hasattr(test_split, 'select'):
        test_split = test_split.select(range(test_limit))
    else:
        test_split = test_split[:test_limit]
    test_data = [format_example(ex) for ex in test_split]
    
    if args.max_samples:
        train_data = train_data[:args.max_samples]
        test_data = test_data[:min(args.max_samples // 10, len(test_data))]
    
    print(f"Dataset: MorryShah/complex_ner")
    print(f"Train samples: {len(train_data)}")
    print(f"Test samples: {len(test_data)}")
    
    return train_data, test_data


def setup_model_and_tokenizer(args):
    """Load the tokenizer and base model, applying quantization and optional LoRA.

    Args:
        args (argparse.Namespace): Parsed CLI args controlling model name,
            quantization, mixed precision, gradient checkpointing, and LoRA config.

    Returns:
        tuple: `(model, tokenizer)` ready for training.
    """
    print("\n" + "="*80)
    print("LOADING MODEL AND TOKENIZER")
    print("="*80)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
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
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if args.mixed_precision == "bf16" else torch.float16
    )
    
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    
    if args.lora:
        model = prepare_model_for_kbit_training(model)
        lora_config = LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_alpha,
            target_modules=args.lora_target_modules.split(","),
            lora_dropout=args.lora_dropout, bias="none", task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    return model, tokenizer


def print_examples(train_dataset, tokenizer, num_examples=2):
    """Print a few decoded training examples for sanity-checking the formatting.

    Args:
        train_dataset (NERDataset): Dataset to sample examples from.
        tokenizer: Tokenizer used to decode `input_ids` back to text.
        num_examples (int): Number of examples to print. Defaults to 2.
    """
    print("\n" + "="*80)
    print("FORMATTED EXAMPLES")
    print("="*80)
    
    for i in range(min(num_examples, len(train_dataset))):
        example = train_dataset[i]
        full_text = tokenizer.decode(example["input_ids"], skip_special_tokens=False)
        print(f"\n--- Example {i+1} ---")
        print(full_text[:400] + "...")


def train_model(model, tokenizer, train_dataset, eval_dataset, args):
    """Run supervised fine-tuning with `Trainer` and save the resulting model.

    Args:
        model: Causal LM to fine-tune (optionally LoRA-wrapped).
        tokenizer: Tokenizer paired with `model`; also saved to `args.output_dir`.
        train_dataset (NERDataset): Training split.
        eval_dataset (NERDataset): Evaluation split used during training.
        args (argparse.Namespace): Parsed CLI args supplying all `TrainingArguments`.

    Returns:
        Trainer: The trainer instance after training completes.
    """
    print("\n" + "="*80)
    print("TRAINING")
    print("="*80)

    training_args = TrainingArguments(
        output_dir=args.output_dir, num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate, warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay, logging_steps=args.logging_steps,
        eval_steps=args.eval_steps, save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        fp16=(args.mixed_precision == "fp16"), bf16=(args.mixed_precision == "bf16"),
        optim=args.optim, gradient_checkpointing=args.gradient_checkpointing,
        eval_strategy="steps", save_strategy="steps",
        load_best_model_at_end=True, report_to=["none"], seed=args.seed
    )
    
    trainer = Trainer(
        model=model, args=training_args,
        train_dataset=train_dataset, eval_dataset=eval_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True)
    )
    
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    return trainer


def evaluate_model(model, tokenizer, test_data, args):
    """Generate entity predictions for up to 100 test examples and score
    entity-level precision, recall, and F1.

    Writes the results to `evaluation_results.json` under `args.output_dir`.

    Args:
        model: Fine-tuned causal LM to evaluate.
        tokenizer: Tokenizer paired with `model`.
        test_data (list[dict]): Test examples with `text` and `entities`.
        args (argparse.Namespace): Parsed CLI args; uses `max_length` and `output_dir`.

    Returns:
        dict: `{"f1", "precision", "recall"}` evaluation results.
    """
    print("\n" + "="*80)
    print("EVALUATION")
    print("="*80)

    model.eval()
    all_pred_entities = []
    all_true_entities = []
    
    with torch.no_grad():
        for i, item in enumerate(test_data[:100]):  # Limit for speed
            if i % 20 == 0:
                print(f"  Progress: {i}/{min(100, len(test_data))}")
            
            instruction = "Extract all named entities from the text and provide them in JSON format with entity text and label."
            prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{item['text']}\n\n### Response:\n"
            
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=args.max_length)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.1, 
                                    do_sample=False, pad_token_id=tokenizer.pad_token_id)
            
            generated_text = tokenizer.decode(outputs[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
            
            # Parse entities
            try:
                pred_entities = json.loads(generated_text)
                if not isinstance(pred_entities, list):
                    pred_entities = []
            except:
                pred_entities = []
            
            all_pred_entities.append(pred_entities)
            all_true_entities.append(item["entities"])
    
    # Simple entity-level metrics
    num_correct = 0
    num_pred = 0
    num_true = 0
    
    for pred, true in zip(all_pred_entities, all_true_entities):
        pred_set = {(e.get("entity", ""), e.get("label", "")) for e in pred if isinstance(e, dict)}
        true_set = {(e.get("entity", ""), e.get("label", "")) for e in true if isinstance(e, dict)}
        
        num_correct += len(pred_set & true_set)
        num_pred += len(pred_set)
        num_true += len(true_set)
    
    precision = num_correct / num_pred if num_pred > 0 else 0
    recall = num_correct / num_true if num_true > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"Entity-level F1: {f1:.4f}")
    print(f"Entity-level Precision: {precision:.4f}")
    print(f"Entity-level Recall: {recall:.4f}")
    
    results = {"f1": float(f1), "precision": float(precision), "recall": float(recall)}
    with open(os.path.join(args.output_dir, "evaluation_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    return results


def main():
    """Run the end-to-end standard NER fine-tuning pipeline: load data, build the
    model, train, and evaluate (or stop early if `--debug_first_batch` is set).
    """
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    print("\n" + "="*80)
    print("CONFIGURATION - Standard NER")
    print("="*80)
    print(f"Model: {args.model}")
    print(f"Quantization: {args.quantization}")
    print(f"LoRA: {args.lora}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    
    train_data, test_data = load_and_prepare_data(args)
    model, tokenizer = setup_model_and_tokenizer(args)
    
    train_dataset = NERDataset(train_data, tokenizer, args.max_length)
    eval_dataset = NERDataset(test_data[:100], tokenizer, args.max_length)
    
    print_examples(train_dataset, tokenizer)
    
    if args.debug_first_batch:
        return
    
    train_model(model, tokenizer, train_dataset, eval_dataset, args)
    evaluate_model(model, tokenizer, test_data, args)
    
    print("\nTRAINING COMPLETE!")


if __name__ == "__main__":
    main()
