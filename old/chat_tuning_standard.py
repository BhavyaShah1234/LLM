#!/usr/bin/env python3
"""Chat Tuning Standard - devrev-research/MathChatSync-reasoning"""
import argparse, os, json, random, re
from typing import Dict, List
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, BitsAndBytesConfig, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
import warnings
warnings.filterwarnings("ignore")

def parse_args():
    """Parse CLI arguments controlling model choice, quantization/LoRA, and training hyperparameters.

    Returns:
        argparse.Namespace: parsed arguments consumed by `load_data`, `setup`, and `main`.
    """
    p=argparse.ArgumentParser()
    p.add_argument("--model",type=str,required=True,choices=["Qwen/Qwen3-4B-Base","Qwen/Qwen3-4B","Qwen/Qwen3-4B-Thinking-2507","Qwen/Qwen3-4B-Instruct-2507"])
    p.add_argument("--quantization",type=str,default="4bit",choices=["no","4bit","8bit"])
    p.add_argument("--lora",action="store_true")
    p.add_argument("--lora_r",type=int,default=16)
    p.add_argument("--lora_alpha",type=int,default=32)
    p.add_argument("--lora_dropout",type=float,default=0.05)
    p.add_argument("--lora_target_modules",type=str,default="q_proj,k_proj,v_proj,o_proj")
    p.add_argument("--mixed_precision",type=str,default="bf16",choices=["no","fp16","bf16"])
    p.add_argument("--batch_size",type=int,default=2)
    p.add_argument("--eval_batch_size",type=int,default=4)
    p.add_argument("--gradient_accumulation_steps",type=int,default=4)
    p.add_argument("--learning_rate",type=float,default=2e-4)
    p.add_argument("--epochs",type=int,default=3)
    p.add_argument("--max_length",type=int,default=1024)
    p.add_argument("--warmup_steps",type=int,default=100)
    p.add_argument("--weight_decay",type=float,default=0.01)
    p.add_argument("--gradient_checkpointing",action="store_true",default=True)
    p.add_argument("--optim",type=str,default="paged_adamw_8bit")
    p.add_argument("--max_samples",type=int,default=None)
    p.add_argument("--output_dir",type=str,default="./output/chat_tuning_standard")
    p.add_argument("--logging_steps",type=int,default=10)
    p.add_argument("--eval_steps",type=int,default=500)
    p.add_argument("--save_steps",type=int,default=500)
    p.add_argument("--save_total_limit",type=int,default=2)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--debug_first_batch",action="store_true")
    return p.parse_args()

class ChatDS(Dataset):
    """Torch dataset that tokenizes prompt/response chat pairs and masks the prompt tokens out of the loss.

    Attributes:
        data (list): list of `{"prompt": str, "response": str}` examples.
        tok: tokenizer used to encode prompt and full (prompt + response) text.
        ml (int): max token length the full sequence is truncated to.
    """
    def __init__(self,d,tok,ml):
        """Store the examples, tokenizer, and max length used at `__getitem__` time.

        Args:
            d (list): list of `{"prompt": str, "response": str}` examples.
            tok: tokenizer with a callable `__call__` interface.
            ml (int): max sequence length passed to the tokenizer for the full text.
        """
        self.data,self.tok,self.ml=d,tok,ml
    def __len__(self):
        """Return the number of examples in the dataset."""
        return len(self.data)
    def __getitem__(self,i):
        """Tokenize example `i` and mask prompt tokens in the labels with -100.

        Args:
            i (int): index of the example to fetch.

        Returns:
            dict: `input_ids`, `attention_mask`, and `labels` tensors for the example,
            with label positions covering the prompt set to -100 so they're excluded
            from the loss.
        """
        it=self.data[i]
        pr,rsp=it['prompt'],it['response']
        fu=pr+rsp
        pt=self.tok(pr,add_special_tokens=True,truncation=False)
        ft=self.tok(fu,max_length=self.ml,truncation=True,add_special_tokens=True)
        ii,am=ft["input_ids"],ft["attention_mask"]
        lb=ii.copy()
        lb[:len(pt["input_ids"])]=[-100]*len(pt["input_ids"])
        return {"input_ids":torch.tensor(ii),"attention_mask":torch.tensor(am),"labels":torch.tensor(lb)}

def load_data(a):
    """Load and format the MathChatSync-reasoning dataset into prompt/response pairs, splitting off a test set.

    Args:
        a (argparse.Namespace): parsed CLI args; uses `max_samples` to optionally cap the
            train/test set sizes.

    Returns:
        tuple[list, list]: `(train, test)` lists of `{"prompt": str, "response": str}` dicts.
    """
    print("\\nLoading: devrev-research/MathChatSync-reasoning")
    ds=load_dataset("devrev-research/MathChatSync-reasoning")
    def fmt(e):
        user=e.get("user",e.get("question",""))
        asst=e.get("assistant",e.get("response",""))
        pr="SYSTEM: You are a helpful AI assistant.\\nUSER: "+user+"\\nASSISTANT: "
        return {"prompt":pr,"response":asst}
    tr=[fmt(e) for e in ds["train"] if e.get("user") or e.get("question")]
    ts=len(tr)//10
    te,tr=tr[-ts:],tr[:-ts]
    if a.max_samples:
        tr,te=tr[:a.max_samples],te[:min(a.max_samples//10,len(te))]
    print(f"Train:{len(tr)} Test:{len(te)}")
    return tr,te

def setup(a):
    """Load the tokenizer and base model, applying quantization, gradient checkpointing, and LoRA per args.

    Args:
        a (argparse.Namespace): parsed CLI args controlling model name, quantization,
            mixed precision, gradient checkpointing, and LoRA settings.

    Returns:
        tuple: `(model, tokenizer)` ready for training.
    """
    tok=AutoTokenizer.from_pretrained(a.model,trust_remote_code=True,use_fast=False)
    if tok.pad_token is None:
        tok.pad_token=tok.eos_token
    qc=None
    if a.quantization=="4bit":
        qc=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_compute_dtype=torch.bfloat16 if a.mixed_precision=="bf16" else torch.float16,bnb_4bit_use_double_quant=True,bnb_4bit_quant_type="nf4")
    elif a.quantization=="8bit":
        qc=BitsAndBytesConfig(load_in_8bit=True)
    m=AutoModelForCausalLM.from_pretrained(a.model,quantization_config=qc,device_map="auto",trust_remote_code=True,torch_dtype=torch.bfloat16 if a.mixed_precision=="bf16" else torch.float16)
    if a.gradient_checkpointing:
        m.gradient_checkpointing_enable()
    if a.lora:
        m=prepare_model_for_kbit_training(m)
        m=get_peft_model(m,LoraConfig(r=a.lora_r,lora_alpha=a.lora_alpha,target_modules=a.lora_target_modules.split(","),lora_dropout=a.lora_dropout,bias="none",task_type="CAUSAL_LM"))
        m.print_trainable_parameters()
    return m,tok

def main():
    """Parse args, load data and model, and run the standard chat SFT training loop end to end."""
    a=parse_args()
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
    print(f"\\n{'='*80}\\nChat Tuning Standard\\n{'='*80}")
    if a.debug_first_batch:return
    tr,te=load_data(a)
    m,tok=setup(a)
    trd=ChatDS(tr,tok,a.max_length)
    evd=ChatDS(te[:100],tok,a.max_length)
    ta=TrainingArguments(output_dir=a.output_dir,num_train_epochs=a.epochs,per_device_train_batch_size=a.batch_size,per_device_eval_batch_size=a.eval_batch_size,gradient_accumulation_steps=a.gradient_accumulation_steps,learning_rate=a.learning_rate,warmup_steps=a.warmup_steps,weight_decay=a.weight_decay,logging_steps=a.logging_steps,eval_steps=a.eval_steps,save_steps=a.save_steps,save_total_limit=a.save_total_limit,fp16=(a.mixed_precision=="fp16"),bf16=(a.mixed_precision=="bf16"),optim=a.optim,gradient_checkpointing=a.gradient_checkpointing,eval_strategy="steps",save_strategy="steps",load_best_model_at_end=True,report_to=["none"],seed=a.seed)
    t=Trainer(model=m,args=ta,train_dataset=trd,eval_dataset=evd,data_collator=DataCollatorForSeq2Seq(tokenizer=tok,model=m,padding=True))
    t.train()
    t.save_model(a.output_dir)
    tok.save_pretrained(a.output_dir)
    print("Done!")

if __name__=="__main__":
    main()
