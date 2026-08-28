#!/usr/bin/env python3
"""Chat Tuning WITH CoT - PJMixers-Dev/oumi-ai_lmsys_chat_1m_clean_R1-1k-think-1k-response-ShareGPT"""
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
    p=argparse.ArgumentParser()
    p.add_argument("--model",type=str,required=True,choices=["Qwen/Qwen3-4B-Base","Qwen/Qwen3-4B","Qwen/Qwen3-4B-Thinking-2507","Qwen/Qwen3-4B-Instruct-2507"])
    p.add_argument("--quantization",type=str,default="4bit",choices=["no","4bit","8bit"])
    p.add_argument("--lora",action="store_true")
    p.add_argument("--lora_r",type=int,default=16)
    p.add_argument("--lora_alpha",type=int,default=32)
    p.add_argument("--lora_dropout",type=float,default=0.05)
    p.add_argument("--lora_target_modules",type=str,default="q_proj,k_proj,v_proj,o_proj")
    p.add_argument("--mixed_precision",type=str,default="bf16",choices=["no","fp16","bf16"])
    p.add_argument("--batch_size",type=int,default=1)
    p.add_argument("--eval_batch_size",type=int,default=2)
    p.add_argument("--gradient_accumulation_steps",type=int,default=8)
    p.add_argument("--learning_rate",type=float,default=2e-4)
    p.add_argument("--epochs",type=int,default=3)
    p.add_argument("--max_length",type=int,default=2048)
    p.add_argument("--warmup_steps",type=int,default=100)
    p.add_argument("--weight_decay",type=float,default=0.01)
    p.add_argument("--gradient_checkpointing",action="store_true",default=True)
    p.add_argument("--optim",type=str,default="paged_adamw_8bit")
    p.add_argument("--max_samples",type=int,default=None)
    p.add_argument("--output_dir",type=str,default="./output/chat_tuning_cot")
    p.add_argument("--logging_steps",type=int,default=10)
    p.add_argument("--eval_steps",type=int,default=500)
    p.add_argument("--save_steps",type=int,default=500)
    p.add_argument("--save_total_limit",type=int,default=2)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--debug_first_batch",action="store_true")
    return p.parse_args()

class ChatCoTDS(Dataset):
    def __init__(self,d,tok,ml):
        self.data,self.tok,self.ml=d,tok,ml
    def __len__(self):
        return len(self.data)
    def __getitem__(self,i):
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
    print("\\nLoading: PJMixers-Dev/oumi-ai_lmsys_chat_1m_clean_R1-1k-think-1k-response-ShareGPT (WITH CoT)")
    ds=load_dataset("PJMixers-Dev/oumi-ai_lmsys_chat_1m_clean_R1-1k-think-1k-response-ShareGPT")
    def fmt(e):
        convs=e.get("conversations",[])
        if len(convs)<2:return None
        user_msg,asst_msg="",""
        for c in convs:
            if c.get("from")=="human":user_msg=c.get("value","")
            elif c.get("from")=="gpt":asst_msg=c.get("value","")
        if not user_msg or not asst_msg:return None
        pr="SYSTEM: You are a helpful AI assistant.\\nUSER: "+user_msg+"\\nASSISTANT: "
        return {"prompt":pr,"response":asst_msg}
    tr=[fmt(e) for e in ds["train"] if fmt(e)][:10000]
    ts=len(tr)//10
    te,tr=tr[-ts:],tr[:-ts]
    if a.max_samples:
        tr,te=tr[:a.max_samples],te[:min(a.max_samples//10,len(te))]
    print(f"Train:{len(tr)} Test:{len(te)}")
    return tr,te

def setup(a):
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
    a=parse_args()
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
    print(f"\\n{'='*80}\\nChat Tuning WITH CoT\\n{'='*80}")
    if a.debug_first_batch:return
    tr,te=load_data(a)
    m,tok=setup(a)
    trd=ChatCoTDS(tr,tok,a.max_length)
    evd=ChatCoTDS(te[:50],tok,a.max_length)
    ta=TrainingArguments(output_dir=a.output_dir,num_train_epochs=a.epochs,per_device_train_batch_size=a.batch_size,per_device_eval_batch_size=a.eval_batch_size,gradient_accumulation_steps=a.gradient_accumulation_steps,learning_rate=a.learning_rate,warmup_steps=a.warmup_steps,weight_decay=a.weight_decay,logging_steps=a.logging_steps,eval_steps=a.eval_steps,save_steps=a.save_steps,save_total_limit=a.save_total_limit,fp16=(a.mixed_precision=="fp16"),bf16=(a.mixed_precision=="bf16"),optim=a.optim,gradient_checkpointing=a.gradient_checkpointing,eval_strategy="steps",save_strategy="steps",load_best_model_at_end=True,report_to=["none"],seed=a.seed)
    t=Trainer(model=m,args=ta,train_dataset=trd,eval_dataset=evd,data_collator=DataCollatorForSeq2Seq(tokenizer=tok,model=m,padding=True))
    t.train()
    t.save_model(a.output_dir)
    tok.save_pretrained(a.output_dir)
    print("Done!")

if __name__=="__main__":
    main()
