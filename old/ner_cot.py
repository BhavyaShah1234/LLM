#!/usr/bin/env python3
"""
NER with CoT - Fine-tuning for Qwen Models
Dataset: zilliz/natural_questions-context-relevance-with-think (WITH CoT reasoning)
Format: ### Instruction: ... ### Input: ... ### Response: <think>{think_process}</think>{entities_json}
Loss: On BOTH thinking tokens AND entity tokens
"""

import argparse
import os
import json
import random
import re
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
import warnings
warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen for NER with CoT")
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
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--eval_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--optim", type=str, default="paged_adamw_8bit")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default="./output/ner_cot")
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--debug_first_batch", action="store_true")
    return parser.parse_args()


class NERCoTDataset(Dataset):
    def __init__(self, data: List[Dict], tokenizer, max_length: int):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        instruction = "Extract all relevant spans from the text and provide them with their relevance labels. Think through the reasoning process."
        input_text = item["text"]
        think_process = item["think_process"]
        entities_json = json.dumps(item["entities"], ensure_ascii=False)
        
        prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
        response = f"<think>{think_process}</think>{entities_json}"
        full_text = prompt + response
        
        prompt_tokens = self.tokenizer(prompt, add_special_tokens=True, truncation=False)
        full_tokens = self.tokenizer(full_text, max_length=self.max_length, 
                                     truncation=True, add_special_tokens=True)
        
        input_ids = full_tokens["input_ids"]
        attention_mask = full_tokens["attention_mask"]
        labels = input_ids.copy()
        prompt_len = len(prompt_tokens["input_ids"])
        labels[:prompt_len] = [-100] * prompt_len  # Mask only prompt, compute loss on think + entities
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }


def load_and_prepare_data(args):
    print("\n" + "="*80)
    print("LOADING DATASET")
    print("="*80)
    
    dataset = load_dataset("zilliz/natural_questions-context-relevance-with-think")
    
    def format_example(example):
        texts = example.get("texts", "")
        # Handle if texts is a list
        if isinstance(texts, list):
            texts = " ".join(str(t) for t in texts)
        texts = str(texts)  # Ensure it's a string
        context_spans = example.get("context_spans", [])
        relevance_labels = example.get("context_spans_relevance", [])
        think_process = example.get("think_process", "")
        
        sentences = re.split(r'[.!?]+', texts)
        text = sentences[0] if sentences else texts
        text = text[:1000]
        
        entities = []
        for span, label in zip(context_spans[:10], relevance_labels[:10]):
            if isinstance(span, list) and len(span) >= 2:
                # Handle nested lists - ensure we get integers
                start = span[0] if not isinstance(span[0], list) else span[0][0] if span[0] else 0
                end = span[1] if not isinstance(span[1], list) else span[1][0] if span[1] else 0
                # Convert to int if needed
                try:
                    start = int(start)
                    end = int(end)
                except (ValueError, TypeError):
                    continue
                if 0 <= start < len(text) and start < end <= len(text):
                    entity_text = text[start:end]
                    entities.append({"entity": entity_text, "label": str(label)})
        
        # Clean and limit think process
        think_process = str(think_process).strip()[:500]
        if not think_process:
            think_process = "Analyzing the text to identify relevant spans and their context relevance."
        
        return {"text": text, "entities": entities, "think_process": think_process}
    
    all_data = []
    for split in ["train"]:
        if split in dataset:
            all_data.extend([format_example(ex) for ex in dataset[split]])
    
    test_size = len(all_data) // 10
    test_data = all_data[-test_size:]
    train_data = all_data[:-test_size]
    
    if args.max_samples:
        train_data = train_data[:args.max_samples]
        test_data = test_data[:min(args.max_samples // 10, len(test_data))]
    
    print(f"Dataset: zilliz/natural_questions-context-relevance-with-think (WITH CoT)")
    print(f"Train samples: {len(train_data)}")
    print(f"Test samples: {len(test_data)}")
    print(f"Format: <think>{{reasoning}}</think>{{entities}}")
    print(f"Loss: On BOTH thinking and entities")
    
    return train_data, test_data


def setup_model_and_tokenizer(args):
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
    elif args.quantization == "8bit":
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model, quantization_config=quantization_config,
        device_map="auto", trust_remote_code=True,
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
    print("\n" + "="*80)
    print("FORMATTED EXAMPLES (WITH CoT)")
    print("="*80)
    
    for i in range(min(num_examples, len(train_dataset))):
        example = train_dataset[i]
        full_text = tokenizer.decode(example["input_ids"], skip_special_tokens=False)
        print(f"\n--- Example {i+1} ---")
        print(full_text[:500] + "...")


def train_model(model, tokenizer, train_dataset, eval_dataset, args):
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
    model.eval()
    num_correct = num_pred = num_true = 0
    cot_outputs = []
    
    with torch.no_grad():
        for i, item in enumerate(test_data[:100]):
            if i % 20 == 0:
                print(f"  Progress: {i}/100")
            
            instruction = "Extract all relevant spans from the text and provide them with their relevance labels. Think through the reasoning process."
            prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{item['text']}\n\n### Response:\n"
            
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=args.max_length)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.1, 
                                    do_sample=False, pad_token_id=tokenizer.pad_token_id)
            
            generated_text = tokenizer.decode(outputs[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
            
            # Extract CoT
            cot_match = re.search(r'<think>(.*?)</think>', generated_text, re.DOTALL)
            if cot_match:
                cot_text = cot_match.group(1).strip()
                remaining = generated_text[cot_match.end():]
            else:
                cot_text = ""
                remaining = generated_text
            
            cot_outputs.append(cot_text)
            
            try:
                pred_entities = json.loads(remaining.strip())
                if not isinstance(pred_entities, list):
                    pred_entities = []
            except:
                pred_entities = []
            
            pred_set = {(e.get("entity", ""), e.get("label", "")) for e in pred_entities if isinstance(e, dict)}
            true_set = {(e.get("entity", ""), e.get("label", "")) for e in item["entities"] if isinstance(e, dict)}
            
            num_correct += len(pred_set & true_set)
            num_pred += len(pred_set)
            num_true += len(true_set)
    
    precision = num_correct / num_pred if num_pred > 0 else 0
    recall = num_correct / num_true if num_true > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    num_with_cot = sum(1 for cot in cot_outputs if cot)
    avg_cot_len = np.mean([len(cot.split()) for cot in cot_outputs if cot]) if num_with_cot > 0 else 0
    
    print(f"\nF1: {f1:.4f}  |  Precision: {precision:.4f}  |  Recall: {recall:.4f}")
    print(f"CoT Usage: {num_with_cot}/100 ({100*num_with_cot/100:.1f}%)  |  Avg Length: {avg_cot_len:.1f} words")
    
    results = {
        "f1": float(f1), "precision": float(precision), "recall": float(recall),
        "cot_enabled": True, "cot_usage_rate": float(num_with_cot/100), 
        "avg_cot_length_words": float(avg_cot_len)
    }
    with open(os.path.join(args.output_dir, "evaluation_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    return results


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    print(f"\n{'='*80}\nCONFIGURATION - NER With CoT\n{'='*80}")
    print(f"Model: {args.model} | Quantization: {args.quantization} | LoRA: {args.lora}")
    
    train_data, test_data = load_and_prepare_data(args)
    model, tokenizer = setup_model_and_tokenizer(args)
    
    train_dataset = NERCoTDataset(train_data, tokenizer, args.max_length)
    eval_dataset = NERCoTDataset(test_data[:50], tokenizer, args.max_length)
    
    print_examples(train_dataset, tokenizer)
    
    if args.debug_first_batch:
        return
    
    train_model(model, tokenizer, train_dataset, eval_dataset, args)
    evaluate_model(model, tokenizer, test_data, args)


if __name__ == "__main__":
    main()
