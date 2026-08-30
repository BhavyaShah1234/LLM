#!/usr/bin/env python3
"""VQA WITHOUT CoT - opendatalab/ChartVerse-SFT-1.8M (image+question -> answer)"""
import argparse, os, json, random, re
from typing import Dict, List
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, Qwen2VLForConditionalGeneration, Qwen2VLProcessor, TrainingArguments, Trainer,BitsAndBytesConfig, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
from PIL import Image
import io
import warnings
warnings.filterwarnings("ignore")

def parse_args():
    """Parse command-line arguments for the no-CoT VQA fine-tuning run.

    Returns:
        argparse.Namespace: Parsed CLI arguments covering model choice,
        quantization, LoRA, training hyperparameters, data limits, and
        output/logging configuration.
    """
    p=argparse.ArgumentParser()
    p.add_argument("--model",type=str,required=True,choices=["Qwen/Qwen2-VL-2B","Qwen/Qwen2-VL-2B-Instruct"])
    p.add_argument("--quantization",type=str,default="4bit",choices=["no","4bit","8bit"])
    p.add_argument("--lora",action="store_true")
    p.add_argument("--lora_r",type=int,default=16)
    p.add_argument("--lora_alpha",type=int,default=32)
    p.add_argument("--lora_dropout",type=float,default=0.05)
    p.add_argument("--lora_target_modules",type=str,default="q_proj,k_proj,v_proj,o_proj")
    p.add_argument("--mixed_precision",type=str,default="bf16",choices=["no","fp16","bf16"])
    p.add_argument("--batch_size",type=int,default=1)
    p.add_argument("--eval_batch_size",type=int,default=1)
    p.add_argument("--gradient_accumulation_steps",type=int,default=8)
    p.add_argument("--learning_rate",type=float,default=2e-4)
    p.add_argument("--epochs",type=int,default=3)
    p.add_argument("--max_length",type=int,default=512)
    p.add_argument("--warmup_steps",type=int,default=100)
    p.add_argument("--weight_decay",type=float,default=0.01)
    p.add_argument("--gradient_checkpointing",action="store_true",default=True)
    p.add_argument("--optim",type=str,default="paged_adamw_8bit")
    p.add_argument("--max_samples",type=int,default=None)
    p.add_argument("--output_dir",type=str,default="./output/vqa_no_cot")
    p.add_argument("--logging_steps",type=int,default=10)
    p.add_argument("--eval_steps",type=int,default=500)
    p.add_argument("--save_steps",type=int,default=500)
    p.add_argument("--save_total_limit",type=int,default=2)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--debug_first_batch",action="store_true")
    return p.parse_args()

class VQADS(Dataset):
    """Torch dataset formatting chart-VQA examples without reasoning.

    Each item pairs an image with an instruction/question prompt whose
    response is just the answer, processed jointly by a Qwen2-VL processor.
    Falls back to a blank white image if none is present, and skips to the
    next index on any processing error.

    Attributes:
        data: Formatted examples with `"question"`, `"answer"`, and
            `"image"` keys.
        proc: Qwen2-VL processor used to jointly encode text and image.
        ml (int): Maximum tokenized sequence length.
    """

    def __init__(self,d,proc,ml):
        """Initialize the dataset.

        Args:
            d: Formatted examples with `"question"`, `"answer"`, and
                `"image"` keys.
            proc: Qwen2-VL processor used to jointly encode text and image.
            ml (int): Maximum tokenized sequence length.
        """
        self.data,self.proc,self.ml=d,proc,ml
    def __len__(self):
        """Return the number of examples in the dataset.

        Returns:
            int: Number of examples.
        """
        return len(self.data)
    def __getitem__(self,i):
        """Format and process one example into model-ready tensors.

        Args:
            i (int): Index of the example to fetch.

        Returns:
            dict: Processor output (input_ids, attention_mask, pixel_values,
            etc.) with the batch dimension squeezed out.
        """
        it=self.data[i]
        try:
            img=it.get("image")
            if img and hasattr(img,"convert"):
                img=img.convert("RGB")
            else:
                img=Image.new("RGB",(224,224),(255,255,255))
            txt=f"### Instruction:\\nAnswer the question about the image.\\n### Input:\\n{it['question']}\\n### Response:\\n{it['answer']}"
            inputs=self.proc(text=[txt],images=[img],return_tensors="pt",padding="max_length",max_length=self.ml,truncation=True)
            return {k:v.squeeze(0) for k,v in inputs.items()}
        except Exception as e:
            print(f"Error processing sample {i}:{e}")
            return self.__getitem__((i+1)%len(self.data))

def load_data(a):
    """Stream and format up to 5000 chart-VQA examples without reasoning.

    Streams `opendatalab/ChartVerse-SFT-1.8M`, normalizes question/answer/
    image fields, carves off the last 10% as a test split, and optionally
    truncates to `a.max_samples`.

    Args:
        a (argparse.Namespace): Parsed CLI args; uses `a.max_samples`.

    Returns:
        tuple[list[dict], list[dict]]: `(train_data, test_data)`.
    """
    print("\\nLoading: opendatalab/ChartVerse-SFT-1.8M (NO CoT)")
    ds=load_dataset("opendatalab/ChartVerse-SFT-1.8M",split="train",streaming=True)
    def fmt(e):
        q=e.get("query",e.get("question",""))
        a=e.get("answer",e.get("response",""))
        img=e.get("image",e.get("chart",None))
        return {"question":q,"answer":a,"image":img}
    tr=[fmt(e) for _,e in zip(range(5000),ds)]
    ts=len(tr)//10
    te,tr=tr[-ts:],tr[:-ts]
    if a.max_samples:
        tr,te=tr[:a.max_samples],te[:min(a.max_samples//10,len(te))]
    print(f"Train:{len(tr)} Test:{len(te)}")
    return tr,te

def setup(a):
    """Load the Qwen2-VL processor and quantized/LoRA-wrapped model.

    Configures 4-bit/8-bit/no quantization per `a.quantization`, enables
    gradient checkpointing if requested, and wraps the model with LoRA if
    `a.lora` is set.

    Args:
        a (argparse.Namespace): Parsed CLI args; uses `a.model`,
            `a.quantization`, `a.mixed_precision`, `a.gradient_checkpointing`,
            `a.lora`, `a.lora_r`, `a.lora_alpha`, `a.lora_dropout`,
            `a.lora_target_modules`.

    Returns:
        tuple: `(model, processor)`.
    """
    proc=Qwen2VLProcessor.from_pretrained(a.model,trust_remote_code=True)
    qc=None
    if a.quantization=="4bit":
        qc=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_compute_dtype=torch.bfloat16 if a.mixed_precision=="bf16" else torch.float16,bnb_4bit_use_double_quant=True,bnb_4bit_quant_type="nf4")
    elif a.quantization=="8bit":
        qc=BitsAndBytesConfig(load_in_8bit=True)
    m=Qwen2VLForConditionalGeneration.from_pretrained(a.model,quantization_config=qc,device_map="auto",trust_remote_code=True,torch_dtype=torch.bfloat16 if a.mixed_precision=="bf16" else torch.float16)
    if a.gradient_checkpointing:
        m.gradient_checkpointing_enable()
    if a.lora:
        m=prepare_model_for_kbit_training(m)
        m=get_peft_model(m,LoraConfig(r=a.lora_r,lora_alpha=a.lora_alpha,target_modules=a.lora_target_modules.split(","),lora_dropout=a.lora_dropout,bias="none",task_type="CAUSAL_LM"))
        m.print_trainable_parameters()
    return m,proc

def main():
    """Run the end-to-end no-CoT VQA fine-tuning pipeline.

    Parses args, seeds RNGs, and (unless `--debug_first_batch` is set)
    streams/loads data, sets up the model, builds datasets, trains with
    `Trainer`, and saves the model and processor to `a.output_dir`.
    """
    a=parse_args()
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
    print(f"\\n{'='*80}\\nVQA WITHOUT CoT\\n{'='*80}")
    if a.debug_first_batch:return
    tr,te=load_data(a)
    m,proc=setup(a)
    trd=VQADS(tr,proc,a.max_length)
    evd=VQADS(te[:50],proc,a.max_length)
    ta=TrainingArguments(output_dir=a.output_dir,num_train_epochs=a.epochs,per_device_train_batch_size=a.batch_size,per_device_eval_batch_size=a.eval_batch_size,gradient_accumulation_steps=a.gradient_accumulation_steps,learning_rate=a.learning_rate,warmup_steps=a.warmup_steps,weight_decay=a.weight_decay,logging_steps=a.logging_steps,eval_steps=a.eval_steps,save_steps=a.save_steps,save_total_limit=a.save_total_limit,fp16=(a.mixed_precision=="fp16"),bf16=(a.mixed_precision=="bf16"),optim=a.optim,gradient_checkpointing=a.gradient_checkpointing,eval_strategy="steps",save_strategy="steps",load_best_model_at_end=True,report_to=["none"],seed=a.seed)
    t=Trainer(model=m,args=ta,train_dataset=trd,eval_dataset=evd,data_collator=DataCollatorForSeq2Seq(proc.tokenizer,model=m))
    t.train()
    t.save_model(a.output_dir)
    proc.save_pretrained(a.output_dir)
    print("Done!")

if __name__=="__main__":
    main()
