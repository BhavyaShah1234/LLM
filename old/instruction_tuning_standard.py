#!/usr/bin/env python3
# Standard Instruction Tuning - ibivibiv/math_instruct
import argparse, os, json, random
from typing import Dict, List
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, BitsAndBytesConfig, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
import evaluate
import warnings
warnings.filterwarnings("ignore")

SYSTEM_INSTRUCTIONS = [
    "You are an expert mathematician. You answer users queries directly and accurately.",
    "You are a skilled mathematical problem solver who provides clear and precise solutions.",
    "You are a mathematics tutor who delivers concise and accurate answers to mathematical questions.",
    "You are a professional mathematician providing direct answers to mathematical problems.",
    "You are an experienced math expert who solves problems efficiently and accurately."
]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["Qwen/Qwen3-4B-Base", "Qwen/Qwen3-4B", "Qwen/Qwen3-4B-Thinking-2507", "Qwen/Qwen3-4B-Instruct-2507"])
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
    parser.add_argument("--output_dir", type=str, default="./output/instruction_tuning_standard")
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--debug_first_batch", action="store_true")
    return parser.parse_args()

class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        item = self.data[idx]
        prompt = f"### Instruction:\n{item['instruction']}\n\n### Input:\n{item['input']}\n\n### Response:\n"
        full_text = prompt + item['output']
        prompt_tokens = self.tokenizer(prompt, add_special_tokens=True, truncation=False)
        full_tokens = self.tokenizer(full_text, max_length=self.max_length, truncation=True, add_special_tokens=True)
        input_ids, attention_mask = full_tokens["input_ids"], full_tokens["attention_mask"]
        labels = input_ids.copy()
        labels[:len(prompt_tokens["input_ids"])] = [-100] * len(prompt_tokens["input_ids"])
        return {"input_ids": torch.tensor(input_ids), "attention_mask": torch.tensor(attention_mask), "labels": torch.tensor(labels)}

def load_and_prepare_data(args):
    print("\nLoading dataset: ibivibiv/math_instruct")
    dataset = load_dataset("ibivibiv/math_instruct")
    def format_example(ex):
        instruction = random.choice(SYSTEM_INSTRUCTIONS)
        input_text = ex.get("instruction", ex.get("query", ""))
        output_text = ex.get("output", ex.get("response", ""))
        return {"instruction": instruction, "input": input_text, "output": output_text}
    train_data = [format_example(ex) for ex in dataset["train"]]
    test_size = len(train_data) // 10
    test_data, train_data = train_data[-test_size:], train_data[:-test_size]
    if args.max_samples:
        train_data, test_data = train_data[:args.max_samples], test_data[:min(args.max_samples//10, len(test_data))]
    print(f"Train: {len(train_data)} | Test: {len(test_data)}")
    return train_data, test_data

def setup_model_and_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    quantization_config = None
    if args.quantization == "4bit":
        quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16 if args.mixed_precision=="bf16" else torch.float16, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
    elif args.quantization == "8bit":
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, quantization_config=quantization_config, device_map="auto", trust_remote_code=True, torch_dtype=torch.bfloat16 if args.mixed_precision=="bf16" else torch.float16)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    if args.lora:
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, target_modules=args.lora_target_modules.split(","), lora_dropout=args.lora_dropout, bias="none", task_type="CAUSAL_LM"))
        model.print_trainable_parameters()
    return model, tokenizer

def main():
    args = parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    print(f"\n{'='*80}\nInstruction Tuning Standard\n{'='*80}")
    if args.debug_first_batch: return
    train_data, test_data = load_and_prepare_data(args)
    model, tokenizer = setup_model_and_tokenizer(args)
    train_dataset = InstructionDataset(train_data, tokenizer, args.max_length)
    eval_dataset = InstructionDataset(test_data[:100], tokenizer, args.max_length)
    training_args = TrainingArguments(output_dir=args.output_dir, num_train_epochs=args.epochs, per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=args.eval_batch_size, gradient_accumulation_steps=args.gradient_accumulation_steps, learning_rate=args.learning_rate, warmup_steps=args.warmup_steps, weight_decay=args.weight_decay, logging_steps=args.logging_steps, eval_steps=args.eval_steps, save_steps=args.save_steps, save_total_limit=args.save_total_limit, fp16=(args.mixed_precision=="fp16"), bf16=(args.mixed_precision=="bf16"), optim=args.optim, gradient_checkpointing=args.gradient_checkpointing, eval_strategy="steps", save_strategy="steps", load_best_model_at_end=True, report_to=["none"], seed=args.seed)
    trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True))
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()
