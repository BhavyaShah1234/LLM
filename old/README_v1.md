# Qwen Model Fine-Tuning Scripts Collection

A comprehensive collection of 16 standalone Python scripts for fine-tuning Qwen models (text and multimodal) on diverse NLP and VQA tasks, with and without Chain-of-Thought (CoT) reasoning.

## 📋 Overview

This repository contains fully self-sufficient fine-tuning scripts supporting:
- **6 Qwen Models**: 4 text models (Qwen3-4B variants), 2 multimodal models (Qwen2-VL-2B variants)
- **8 Task Types**: Text Classification, NER, MCQ, Instruction Tuning, Chat Tuning, and VQA
- **3 Optimization Techniques**: Quantization (4-bit/8-bit), LoRA, Mixed Precision
- **Flexible Hardware**: Optimized for 8GB GPU (RTX 5060) with scaling to multi-GPU A100 setups

## 🗂️ Script Matrix

| # | Script | Task | Dataset | CoT | Models | Output Format |
|---|--------|------|---------|-----|--------|---------------|
| 1 | [text_classification_standard.py](text_classification_standard.py) | Text Classification | `Tohrumi/glue_sst2_10k` | ❌ | Qwen3-4B* | `{label}` |
| 2 | [text_classification_no_cot.py](text_classification_no_cot.py) | Text Classification | `domofon/fake_news_cot_reasoning` | ❌ | Qwen3-4B* | `{label}` |
| 3 | [text_classification_cot.py](text_classification_cot.py) | Text Classification | `domofon/fake_news_cot_reasoning` | ✅ | Qwen3-4B* | `<think>{reasoning}</think>{label}` |
| 4 | [ner_standard.py](ner_standard.py) | NER | `MorryShah/complex_ner` | ❌ | Qwen3-4B* | `[{entity, label}...]` |
| 5 | [ner_no_cot.py](ner_no_cot.py) | NER | `zilliz/natural_questions...` | ❌ | Qwen3-4B* | `[{entity, label}...]` |
| 6 | [ner_cot.py](ner_cot.py) | NER | `zilliz/natural_questions...` | ✅ | Qwen3-4B* | `<think>{reasoning}</think>[...]` |
| 7 | [mcq_standard.py](mcq_standard.py) | MCQ | `araag2/MedMCQA` | ❌ | Qwen3-4B* | `{A/B/C/D}` |
| 8 | [mcq_no_cot.py](mcq_no_cot.py) | MCQ | `HPAI-BSC/medmcqa-cot-llama31` | ❌ | Qwen3-4B* | `{A/B/C/D}` |
| 9 | [mcq_cot.py](mcq_cot.py) | MCQ | `HPAI-BSC/medmcqa-cot-llama31` | ✅ | Qwen3-4B* | `<think>{reasoning}</think>Answer: {letter}` |
| 10 | [instruction_tuning_standard.py](instruction_tuning_standard.py) | Instruction | `ibivibiv/math_instruct` | ❌ | Qwen3-4B* | `{output}` |
| 11 | [instruction_tuning_no_cot.py](instruction_tuning_no_cot.py) | Instruction | `domofon/evol-instruct-code-cot-80k` | ❌ | Qwen3-4B* | `{output}` |
| 12 | [instruction_tuning_cot.py](instruction_tuning_cot.py) | Instruction | `domofon/evol-instruct-code-cot-80k` | ✅ | Qwen3-4B* | `<think>{thinking}</think>{output}` |
| 13 | [chat_tuning_standard.py](chat_tuning_standard.py) | Chat | `devrev-research/MathChatSync-reasoning` | ❌ | Qwen3-4B* | `{assistant_response}` |
| 14 | [chat_tuning_cot.py](chat_tuning_cot.py) | Chat | `PJMixers-Dev/oumi-ai_lmsys...` | ✅ | Qwen3-4B* | `<think>...</think>{response}` |
| 15 | [vqa_no_cot.py](vqa_no_cot.py) | VQA | `opendatalab/ChartVerse-SFT-1.8M` | ❌ | Qwen2-VL-2B* | `{answer}` |
| 16 | [vqa_cot.py](vqa_cot.py) | VQA | `opendatalab/ChartVerse-SFT-1.8M` | ✅ | Qwen2-VL-2B* | `<think>{visual_reasoning}</think>{answer}` |

*Each script supports all model variants within its category (4 Qwen3-4B models for text tasks, 2 Qwen2-VL-2B models for VQA)

## 🚀 Quick Start

### Installation

```bash
# Clone and navigate
cd /home/bhavya-shah/Projects/LLM

# Install dependencies
pip install transformers datasets accelerate peft bitsandbytes torch torchvision \
            evaluate rouge_score bert_score scikit-learn Pillow
```

### Basic Usage

```bash
# Example 1: Text Classification (8GB GPU, 4-bit + LoRA)
python3 text_classification_standard.py \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --quantization 4bit \
  --lora \
  --lora_r 16 \
  --batch_size 2 \
  --epochs 3

# Example 2: MCQ with CoT (A100, no quantization, larger batch)
python3 mcq_cot.py \
  --model Qwen/Qwen3-4B-Thinking-2507 \
  --quantization no \
  --mixed_precision bf16 \
  --batch_size 16 \
  --epochs 3

# Example 3: VQA without CoT (multi-GPU)
accelerate launch --num_processes 4 vqa_no_cot.py \
  --model Qwen/Qwen2-VL-2B-Instruct \
  --quantization 4bit \
  --lora \
  --batch_size 1 \
  --epochs 3
```

## 🎯 Common CLI Arguments

All scripts share these arguments:

### Model Selection
- `--model`: Choose from supported Qwen models (text: Qwen3-4B variants, VQA: Qwen2-VL-2B variants)

### Optimization
- `--quantization {no|4bit|8bit}`: Quantization type (default: `4bit` for 8GB GPU)
- `--lora`: Enable LoRA fine-tuning (highly recommended for memory efficiency)
- `--lora_r INT`: LoRA rank (default: 16)
- `--lora_alpha INT`: LoRA alpha scaling (default: 32)
- `--lora_dropout FLOAT`: LoRA dropout (default: 0.05)
- `--lora_target_modules STR`: Comma-separated modules (default: `q_proj,k_proj,v_proj,o_proj`)
- `--mixed_precision {no|fp16|bf16}`: Mixed precision training (default: `bf16`)

### Training Hyperparameters
- `--batch_size INT`: Per-device training batch size (default: 2 for text, 1 for VQA)
- `--gradient_accumulation_steps INT`: Gradient accumulation (default: 4-8 depending on task)
- `--learning_rate FLOAT`: Learning rate (default: 2e-4)
- `--epochs INT`: Number of training epochs (default: 3)
- `--max_length INT`: Maximum sequence length (default: 512-2048 depending on task)
- `--warmup_steps INT`: Warmup steps (default: 100)
- `--weight_decay FLOAT`: Weight decay (default: 0.01)

### System
- `--gradient_checkpointing`: Enable gradient checkpointing (default: True)
- `--optim STR`: Optimizer (default: `paged_adamw_8bit` for memory efficiency)
- `--seed INT`: Random seed (default: 42)

### Data & Output
- `--max_samples INT`: Limit training samples (useful for testing)
- `--output_dir STR`: Output directory for checkpoints
- `--logging_steps INT`: Logging frequency (default: 10)
- `--eval_steps INT`: Evaluation frequency (default: 500)
- `--save_steps INT`: Checkpoint save frequency (default: 500)

### Debugging
- `--debug_first_batch`: Print formatted examples and exit (useful for debugging)

## 📊 Dataset Selection Rationale

### Text Classification (Scripts 2 & 3)
**Chosen**: `domofon/fake_news_cot_reasoning`  
**Why**: 10k rows, clean structure with explicit CoT reasoning, practical real-world task (fake news detection)  
**Alternative considered**: `Syghmon/rich-cot` (rejected due to more complex preprocessing)

### NER (Scripts 5 & 6)
**Chosen**: `zilliz/natural_questions-context-relevance-with-think`  
**Why**: Contains explicit `think_process` column for CoT, context-span annotations for entities, relevance labels  
**Format**: Extracts entities from `context_spans` (start/end indices) + `context_spans_relevance` (labels)

### MCQ (Scripts 8 & 9)
**Chosen**: `HPAI-BSC/medmcqa-cot-llama31`  
**Why**: Medical MCQ with explicit CoT in `response` field, structured answer format  
**Format**: Parses `Answer: X` from response, extracts reasoning before answer for CoT version

### Instruction Tuning (Scripts 11 & 12)
**Chosen**: `domofon/evol-instruct-code-cot-80k`  
**Why**: 80k coding tasks with `thinking` column for CoT, diverse instruction complexity  
**Format**: Renames `instruction` → `input`, adds rotating system instructions, wraps `thinking` in `<think>` tags for CoT

### Chat Tuning (Script 14)
**Chosen**: `PJMixers-Dev/oumi-ai_lmsys_chat_1m_clean_R1-1k-think-1k-response-ShareGPT`  
**Why**: ShareGPT format with CoT annotations in assistant messages, cleaned multi-turn conversations  
**Format**: Extracts user/assistant turns from `conversations` field

### VQA (Scripts 15 & 16)
**Chosen**: `opendatalab/ChartVerse-SFT-1.8M`  
**Why**: 1.8M chart images with questions/answers, supports vision-language tasks  
**Note**: CoT reasoning synthesized during training (dataset may not have explicit reasoning column)

## 🔧 Training Format Details

### Prompt Templates

**Instruction Tuning Format**:
```
### Instruction:
{system_instruction}

### Input:
{input_text}

### Response:
{output}
```

**Chat Format** (Vicuna template fallback):
```
SYSTEM: {system_message}
USER: {user_message}
ASSISTANT: {assistant_response}
```

**VQA Format**:
```
### Instruction:
Answer the question about the image.

### Input:
{question}

### Response:
{answer}
```

### Loss Masking Rules

| Task Type | Loss Computed On | Masked Tokens |
|-----------|------------------|---------------|
| **Text Classification** | Label tokens only | Instruction + Input |
| **NER** | Entity JSON only | Instruction + Input |
| **MCQ** | Answer letter only | Instruction + Input + Options |
| **Instruction (no CoT)** | Output only | Instruction + Input |
| **Instruction (CoT)** | `<think>` content + Output | Instruction + Input |
| **Chat** | Assistant messages only | System + User messages |
| **Chat (CoT)** | Assistant (incl. `<think>`) | System + User messages |
| **VQA (no CoT)** | Answer only | Instruction + Input + Image embeddings |
| **VQA (CoT)** | `<think>` + Answer | Instruction + Input + Image embeddings |

Implementation: Set `labels[i] = -100` for all masked token positions before passing to model.

## 💾 Hardware Recommendations

### 8GB GPU (RTX 5060, RTX 3080 12GB)
```bash
--quantization 4bit \
--lora --lora_r 16 \
--batch_size 1-2 \
--gradient_accumulation_steps 8 \
--gradient_checkpointing \
--mixed_precision bf16
```
**Memory**: ~7GB VRAM with 4-bit + LoRA  
**Effective batch size**: 8-16 (with gradient accumulation)

### 24GB GPU (RTX 3090, RTX 4090, A5000)
```bash
--quantization 8bit \
--lora --lora_r 32 \
--batch_size 4-8 \
--gradient_accumulation_steps 4 \
--mixed_precision bf16
```
**Memory**: ~18GB VRAM with 8-bit + LoRA  
**Effective batch size**: 16-32

### 40GB+ GPU (A100, H100)
```bash
--quantization no \
--lora --lora_r 64 \  # or full fine-tuning without --lora
--batch_size 16-32 \
--gradient_accumulation_steps 1-2 \
--mixed_precision bf16
```
**Memory**: ~35GB VRAM (full precision + LoRA)  
**Effective batch size**: 32-64

### Multi-GPU Setup
```bash
# 4x A100 example
accelerate launch --num_processes 4 \
  --multi_gpu --mixed_precision bf16 \
  {script}.py \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --quantization no \
  --batch_size 8 \
  --gradient_accumulation_steps 1
```
**Effective batch size**: 32 (8 per GPU × 4 GPUs)

## 📈 Evaluation Metrics

### Classification Tasks (Text Classification, MCQ)
- **Accuracy**: Correct predictions / Total predictions
- **F1 Score**: Harmonic mean of precision and recall
  - Macro F1: Unweighted average across classes
  - Weighted F1: Weighted by class support
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **AUROC**: Area under ROC curve (binary classification only)

### NER Tasks
- **Entity-level F1**: Exact match F1 for (entity_text, label) pairs
- **Entity-level Precision**: Correctly predicted entities / All predicted entities
- **Entity-level Recall**: Correctly predicted entities / All ground truth entities

### Generation Tasks (Instruction Tuning, Chat, VQA)
- **ROUGE**: Recall-oriented n-gram overlap (ROUGE-1, ROUGE-2, ROUGE-L)
- **BLEU**: Precision-oriented n-gram overlap
- **METEOR**: Alignment-based metric with synonyms and stemming
- **BERTScore**: Semantic similarity using BERT embeddings
- **Perplexity**: Exponentiated cross-entropy loss

### CoT-Specific Metrics
- **CoT Usage Rate**: % of predictions with `<think>` tags
- **Avg CoT Length**: Average word count in `<think>` content
- **CoT vs No-CoT Accuracy**: Compare same dataset with/without CoT

## 🔍 Startup Configuration Display

All scripts print configuration at startup:

```
================================================================================
CONFIGURATION
================================================================================
Task: Text Classification WITH CoT
Model: Qwen/Qwen3-4B-Thinking-2507
Quantization: 4bit
LoRA: True
  - Rank: 16
  - Alpha: 32
  - Dropout: 0.05
  - Target modules: q_proj,k_proj,v_proj,o_proj
Mixed Precision: bf16
Batch Size: 1
Gradient Accumulation: 8
Effective Batch Size: 8
Learning Rate: 0.0002
Epochs: 3
Max Length: 2048
Output Directory: ./output/text_classification_cot
Seed: 42

================================================================================
LOADING DATASET
================================================================================
Dataset: domofon/fake_news_cot_reasoning (WITH CoT)
Train samples: 9000
Test samples: 1000
Format: ### Instruction / ### Input / ### Response: <think>{reasoning}</think>{label}
Loss: On BOTH thinking and label tokens

================================================================================
FORMATTED EXAMPLES
================================================================================

--- Example 1 ---
Full text:
### Instruction:
Classify whether the following text is real news or fake news. Think through your reasoning step by step.

### Input:
[News article text...]

### Response:
<think>The article contains sensational language and lacks credible sources...</think>fake

Note: Loss computed on BOTH <think> content AND final label

--- Example 2 ---
[...]
```

## 🛠️ Troubleshooting

### Out of Memory (OOM)

**Symptoms**: `CUDA out of memory` error

**Solutions**:
1. Enable 4-bit quantization: `--quantization 4bit`
2. Reduce batch size: `--batch_size 1`
3. Increase gradient accumulation: `--gradient_accumulation_steps 16`
4. Enable gradient checkpointing: `--gradient_checkpointing` (default: enabled)
5. Reduce max length: `--max_length 512`
6. For VQA: Reduce image resolution (edit script's processor config)

### Slow Training

**Symptoms**: <10 samples/second

**Solutions**:
1. Disable gradient checkpointing if memory allows (remove `--gradient_checkpointing`)
2. Use mixed precision: `--mixed_precision bf16` (default)
3. Increase batch size if memory allows
4. Use faster optimizer: `--optim adamw_torch` (instead of paged_adamw_8bit)
5. Reduce eval frequency: `--eval_steps 1000`

### Loss Not Decreasing

**Symptoms**: Training loss plateaus or increases

**Solutions**:
1. Check data formatting (use `--debug_first_batch`)
2. Verify loss masking (inspect printed examples)
3. Lower learning rate: `--learning_rate 1e-4`
4. Increase LoRA rank: `--lora_r 32`
5. Try different model variant (e.g., Instruct vs Base)

### Dataset Loading Errors

**Symptoms**: `DatasetNotFoundError` or streaming issues

**Solutions**:
1. Authenticate with HuggingFace: `huggingface-cli login`
2. For streaming datasets (VQA), check internet connection
3. Limit samples for testing: `--max_samples 100`

## 📁 Output Structure

After training, each script creates:

```
{output_dir}/
├── checkpoint-{step}/          # Training checkpoints
│   ├── adapter_config.json     # LoRA config (if --lora enabled)
│   ├── adapter_model.bin       # LoRA weights
│   ├── trainer_state.json      # Training state
│   └── ...
├── pytorch_model.bin           # Final model weights
├── config.json                 # Model config
├── tokenizer_config.json       # Tokenizer config
├── tokenizer.json              # Tokenizer vocabulary
└── evaluation_results.json     # Evaluation metrics
```

**evaluation_results.json** example:
```json
{
  "accuracy": 0.8750,
  "f1_macro": 0.8621,
  "f1_weighted": 0.8789,
  "precision": 0.8534,
  "recall": 0.8712,
  "auroc": 0.9123,
  "num_test_samples": 1000,
  "cot_enabled": true,
  "cot_usage_rate": 0.9800,
  "avg_cot_length_words": 45.2
}
```

## 🔬 Advanced Usage

### Debug Mode

Print formatted examples without training:

```bash
python3 text_classification_cot.py \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --debug_first_batch
```

### Custom LoRA Configuration

Target specific modules for domain-specific tasks:

```bash
# Target all linear layers
python3 instruction_tuning_cot.py \
  --model Qwen/Qwen3-4B-Thinking-2507 \
  --lora \
  --lora_r 32 \
  --lora_alpha 64 \
  --lora_target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj
```

### Resume from Checkpoint

```bash
# Training automatically resumes from latest checkpoint if output_dir exists
python3 {script}.py \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --output_dir ./output/my_training \
  ...
```

### Inference with Trained Model

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-4B-Instruct-2507",
    device_map="auto",
    trust_remote_code=True
)

# Load LoRA weights
model = PeftModel.from_pretrained(base_model, "./output/text_classification_cot")
tokenizer = AutoTokenizer.from_pretrained("./output/text_classification_cot")

# Inference
prompt = "### Instruction:\nClassify sentiment...\n### Input:\n...\n### Response:\n"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## 📚 Citation

If you use these scripts in your research, please cite:

```bibtex
@software{qwen_finetuning_scripts_2026,
  title = {Comprehensive Qwen Fine-Tuning Scripts Collection},
  author = {Your Name},
  year = {2026},
  url = {https://github.com/yourusername/qwen-finetuning}
}
```

## 🤝 Contributing

Contributions welcome! Please:
1. Test on at least 2 datasets
2. Maintain standalone script structure
3. Update README with changes
4. Follow existing code style

## 📄 License

MIT License - See LICENSE file for details

## 🙏 Acknowledgments

- **Qwen Team** (Alibaba Cloud) for Qwen models
- **HuggingFace** for transformers, datasets, PEFT libraries
- **Dataset Creators**: Tohrumi, domofon, MorryShah, zilliz, HPAI-BSC, araag2, ibivibiv, devrev-research, PJMixers-Dev, opendatalab

## 📞 Support

For issues or questions:
- Open an issue on GitHub
- Check troubleshooting section above
- Review script-specific comments in source code

---

**Last Updated**: February 2026  
**Version**: 1.0.0  
**Status**: ✅ All 16 scripts tested and functional
