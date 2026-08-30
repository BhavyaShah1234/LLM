#!/usr/bin/env python3
"""
MCQ with CoT - Fine-tuning for Qwen Models
Dataset: HPAI-BSC/medmcqa-cot-llama31 (WITH CoT reasoning)
Format: ### Instruction: ... ### Input: {question + options} ### Response: <think>{reasoning}</think>Answer: {letter}
Loss: On BOTH thinking tokens AND answer letter
"""

import argparse, os, json, random, re
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
from sklearn.metrics import accuracy_score, f1_score
import warnings
warnings.filterwarnings("ignore")

def parse_args():
    """Parse CLI arguments controlling model choice, quantization/LoRA, and training hyperparameters.

    Returns:
        argparse.Namespace: parsed arguments consumed by `load_and_prepare_data`,
        `setup_model_and_tokenizer`, `train_and_evaluate`, and `main`.
    """
    parser = argparse.ArgumentParser()
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
    parser.add_argument("--output_dir", type=str, default="./output/mcq_cot")
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--debug_first_batch", action="store_true")
    return parser.parse_args()

class MCQCoTDataset(Dataset):
    """Torch dataset that formats medical MCQ question/options/reasoning/answer examples into a CoT prompt, masking the prompt out of the loss.

    Attributes:
        data (list): list of `{"question", "options", "reasoning", "answer"}` examples.
        tokenizer: tokenizer used to encode prompt and full (prompt + CoT + answer) text.
        max_length (int): max token length the full sequence is truncated to.
    """
    def __init__(self, data, tokenizer, max_length):
        """Store the examples, tokenizer, and max length used at `__getitem__` time.

        Args:
            data (list): list of `{"question", "options", "reasoning", "answer"}` examples.
            tokenizer: tokenizer with a callable `__call__` interface.
            max_length (int): max sequence length passed to the tokenizer for the full text.
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    def __len__(self):
        """Return the number of examples in the dataset."""
        return len(self.data)
    def __getitem__(self, idx):
        """Tokenize example `idx` into a `<think>` CoT + answer-letter prompt, masking the prompt in labels with -100.

        Args:
            idx (int): index of the example to fetch.

        Returns:
            dict: `input_ids`, `attention_mask`, and `labels` lists for the example (not yet
            tensors — collation happens in `DataCollatorForSeq2Seq`), with label positions
            covering the prompt set to -100 so they're excluded from the loss.
        """
        item = self.data[idx]
        instruction = "Answer the medical question. Think through your reasoning step by step."
        prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{item['question']}\n\n{item['options']}\n\n### Response:\n"
        response = f"<think>{item['reasoning']}</think>Answer: {item['answer']}"
        full_text = prompt + response
        prompt_tokens = self.tokenizer(prompt, add_special_tokens=True, truncation=False)
        full_tokens = self.tokenizer(full_text, max_length=self.max_length, truncation=True, add_special_tokens=True)
        input_ids, attention_mask = full_tokens["input_ids"], full_tokens["attention_mask"]
        labels = input_ids.copy()
        labels[:len(prompt_tokens["input_ids"])] = [-100] * len(prompt_tokens["input_ids"])
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

def load_and_prepare_data(args):
    """Load the medmcqa-cot dataset, extract the answer letter and reasoning from each response, and build train/test splits.

    Args:
        args (argparse.Namespace): parsed CLI args; uses `max_samples` to optionally cap the
            train/test set sizes.

    Returns:
        tuple[list, list]: `(train_data, test_data)` lists of
        `{"question", "options", "reasoning", "answer"}` dicts.
    """
    print("\nLoading dataset: HPAI-BSC/medmcqa-cot-llama31 (WITH CoT)")
    dataset = load_dataset("HPAI-BSC/medmcqa-cot-llama31")
    def format_example(ex):
        response = ex.get("response", "")
        answer_match = re.search(r'Answer:\s*([A-D])', response, re.IGNORECASE)
        answer = answer_match.group(1).upper() if answer_match else "A"
        reasoning = re.sub(r'Answer:\s*[A-D].*', '', response, flags=re.IGNORECASE).strip()
        reasoning = reasoning.replace("<think>", "").replace("</think>", "").strip()[:500]
        return {"question": ex.get("question", ""), "options": ex.get("options", ""), "reasoning": reasoning or "Analyzing options...", "answer": answer}
    train_data = [format_example(ex) for ex in dataset["train"]]
    # Get test split with fallback
    if "test" in dataset:
        test_split = dataset["test"]
    elif "validation" in dataset:
        test_split = dataset["validation"]
    else:
        test_split = dataset["train"]
    
    # Properly handle HuggingFace Dataset slicing
    test_limit = min(1000, len(test_split))
    if hasattr(test_split, 'select'):
        test_split = test_split.select(range(test_limit))
    else:
        test_split = test_split[:test_limit]
    test_data = [format_example(ex) for ex in test_split]
    if args.max_samples:
        train_data, test_data = train_data[:args.max_samples], test_data[:min(args.max_samples//10, len(test_data))]
    print(f"Train: {len(train_data)} | Test: {len(test_data)}")
    return train_data, test_data

def setup_model_and_tokenizer(args):
    """Load the tokenizer and base model, applying quantization, gradient checkpointing, and LoRA per args.

    Args:
        args (argparse.Namespace): parsed CLI args controlling model name, quantization,
            mixed precision, gradient checkpointing, and LoRA settings.

    Returns:
        tuple: `(model, tokenizer)` ready for training.
    """
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

def train_and_evaluate(model, tokenizer, train_data, test_data, args):
    """Fine-tune the model on the CoT MCQ training set, then generate on the test set and score accuracy/F1/CoT usage.

    Trains via `Trainer`, saves the model and tokenizer to `args.output_dir`, then runs
    greedy generation over `test_data` to extract the predicted answer letter and whether
    a `<think>` block was produced, and writes `evaluation_results.json` with the metrics.

    Args:
        model: causal LM to fine-tune and evaluate.
        tokenizer: tokenizer paired with `model`.
        train_data (list): training examples as `{"question", "options", "reasoning", "answer"}` dicts.
        test_data (list): held-out examples in the same format, used for both eval-during-training
            (first 50) and post-training generation-based evaluation (all of them).
        args (argparse.Namespace): parsed CLI args controlling training hyperparameters and
            `output_dir`.
    """
    train_dataset = MCQCoTDataset(train_data, tokenizer, args.max_length)
    eval_dataset = MCQCoTDataset(test_data[:50], tokenizer, args.max_length)
    training_args = TrainingArguments(output_dir=args.output_dir, num_train_epochs=args.epochs, per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=args.eval_batch_size, gradient_accumulation_steps=args.gradient_accumulation_steps, learning_rate=args.learning_rate, warmup_steps=args.warmup_steps, weight_decay=args.weight_decay, logging_steps=args.logging_steps, eval_steps=args.eval_steps, save_steps=args.save_steps, save_total_limit=args.save_total_limit, fp16=(args.mixed_precision=="fp16"), bf16=(args.mixed_precision=="bf16"), optim=args.optim, gradient_checkpointing=args.gradient_checkpointing, eval_strategy="steps", save_strategy="steps", load_best_model_at_end=True, report_to=["none"], seed=args.seed)
    trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True))
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    # Evaluate
    model.eval()
    predictions, ground_truths, cot_outputs = [], [], []
    answer_map = {"A": 0, "B": 1, "C": 2, "D": 3}
    with torch.no_grad():
        for i, item in enumerate(test_data):
            if i % 100 == 0: print(f"  {i}/{len(test_data)}")
            prompt = f"### Instruction:\nAnswer the medical question. Think through your reasoning step by step.\n\n### Input:\n{item['question']}\n\n{item['options']}\n\n### Response:\n"
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=args.max_length)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.1, do_sample=False, pad_token_id=tokenizer.pad_token_id)
            generated_text = tokenizer.decode(outputs[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
            cot_match = re.search(r'<think>(.*?)</think>', generated_text, re.DOTALL)
            cot_text = cot_match.group(1).strip() if cot_match else ""
            cot_outputs.append(cot_text)
            remaining = generated_text[cot_match.end():] if cot_match else generated_text
            pred_letter = None
            for letter in ["A","B","C","D"]:
                if letter in remaining[:20].upper():
                    pred_letter = letter
                    break
            predictions.append(answer_map[pred_letter or "A"])
            ground_truths.append(answer_map[item["answer"]])
    accuracy = accuracy_score(ground_truths, predictions)
    f1 = f1_score(ground_truths, predictions, average="macro")
    cot_rate = sum(1 for c in cot_outputs if c) / len(cot_outputs)
    print(f"\nAccuracy: {accuracy:.4f} | F1: {f1:.4f} | CoT Rate: {cot_rate:.2%}")
    with open(os.path.join(args.output_dir, "evaluation_results.json"), "w") as f:
        json.dump({"accuracy": float(accuracy), "f1_macro": float(f1), "cot_enabled": True, "cot_usage_rate": float(cot_rate)}, f, indent=2)

def main():
    """Parse args, load data and model, and run the MCQ-with-CoT training-and-evaluation pipeline end to end."""
    args = parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    print(f"\n{'='*80}\nMCQ With CoT\n{'='*80}")
    if args.debug_first_batch:
        print("Debug mode"); return
    train_data, test_data = load_and_prepare_data(args)
    model, tokenizer = setup_model_and_tokenizer(args)
    train_and_evaluate(model, tokenizer, train_data, test_data, args)

if __name__ == "__main__":
    main()
