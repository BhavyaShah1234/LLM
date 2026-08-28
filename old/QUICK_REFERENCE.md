# Quick Reference: Qwen Fine-Tuning Scripts

## 📊 Technical Feasibility Matrix

| Requirement | Feasibility | Complexity | Time Estimate | Blockers |
|-------------|-------------|------------|---------------|----------|
| Model Loading (Text) | ✅ High | Low | 1 day | None |
| Model Loading (VL) | ✅ High | Medium | 1 day | None |
| Chat Template + Fallback | ✅ High | Medium | 2 days | Need testing |
| Loss Masking (Assistant) | ⚠️ Medium | High | 3 days | Off-by-one errors |
| Loss Masking (Answer Span) | ✅ High | Medium | 2 days | Token alignment |
| Loss Masking (CoT) | ✅ High | Medium | 2 days | Special token handling |
| 4-bit Quantization | ✅ High | Low | 0.5 days | None |
| LoRA Setup | ✅ High | Low | 0.5 days | None |
| Accelerate/Mixed Precision | ✅ High | Low | 1 day | None |
| VL Processing | ✅ High | Medium | 2 days | Memory optimization |
| Dataset Formatting | ✅ High | Medium | 3 days | Format variations |
| Evaluation Metrics | ✅ High | Medium | 2 days | Library dependencies |

**Overall: HIGH FEASIBILITY (90%)** - All requirements are implementable with standard tools.

---

## 🎯 16 Scripts Breakdown (Proposed)

### Text Models (14 scripts)

#### Using Qwen3-4B-Base (2 scripts)
1. `finetune_classification_base.py` - Text classification from scratch
2. `finetune_instruction_base.py` - Instruction tuning from base model

#### Using Qwen3-4B-Instruct-2507 (8 scripts)
3. `finetune_classification_instruct.py` - Text classification (IMDb, AG News)
4. `finetune_ner_instruct.py` - Named Entity Recognition (CoNLL-2003)
5. `finetune_mcq_instruct.py` - Multiple Choice QA (MMLU, ARC)
6. `finetune_instruction_instruct.py` - Instruction following (Alpaca)
7. `finetune_chat_instruct.py` - Multi-turn chat (ShareGPT)
8. `finetune_cot_instruct.py` - Chain-of-Thought reasoning (GSM8K)
9. `finetune_qa_instruct.py` - Question answering (SQuAD)
10. `finetune_summarization_instruct.py` - Text summarization (CNN/DM)

#### Using Qwen3-4B-Thinking-2507 (2 scripts)
11. `finetune_thinking_reasoning.py` - Advanced CoT with <think> tags
12. `finetune_thinking_math.py` - Mathematical reasoning with explicit thinking

#### Custom Format (2 scripts)
13. `finetune_custom_text.py` - Generic script with custom dataset format
14. `finetune_preference_tuning.py` - DPO/Reward modeling (bonus)

### Multimodal Models (2 scripts)

#### Using Qwen2-VL-2B-Instruct
15. `finetune_vqa_instruct.py` - Visual Question Answering (VQAv2)
16. `finetune_image_captioning.py` - Image captioning (COCO)

---

## 🧩 Code Reuse Map

```
                    ┌─────────────────────────────────┐
                    │   utils/model_utils.py          │
                    │  - load_model_and_tokenizer()   │
                    │  - setup_lora()                 │
                    │  - setup_chat_template()        │
                    └────────────┬────────────────────┘
                                 │
                    ┌────────────┴────────────────────┐
                    │                                 │
         ┌──────────▼──────────┐         ┌──────────▼──────────┐
         │  Text Scripts (14)   │         │  VL Scripts (2)     │
         │  - Classification    │         │  - VQA              │
         │  - NER               │         │  - Captioning       │
         │  - MCQ               │         │                     │
         │  - Instruction       │         │  Use Processor      │
         │  - Chat              │         │  instead of         │
         │  - CoT               │         │  Tokenizer          │
         │  - Thinking          │         └─────────────────────┘
         └──────────┬───────────┘
                    │
         ┌──────────▼──────────────────────────────────┐
         │   utils/masking_utils.py                    │
         │  - mask_prompt_tokens()                     │
         │  - mask_user_messages()                     │
         │  - mask_cot_tags()                          │
         └──────────┬──────────────────────────────────┘
                    │
         ┌──────────▼──────────────────────────────────┐
         │   utils/training_utils.py                   │
         │  - train_epoch()                            │
         │  - evaluate_epoch()                         │
         │  - save_checkpoint()                        │
         └──────────┬──────────────────────────────────┘
                    │
         ┌──────────▼──────────────────────────────────┐
         │   utils/eval_utils.py                       │
         │  - evaluate_classification()                │
         │  - evaluate_ner()                           │
         │  - evaluate_generation()                    │
         │  - evaluate_vqa()                           │
         └─────────────────────────────────────────────┘
```

**Reuse Statistics:**
- `model_utils.py`: Used by **all 16 scripts** (100%)
- `training_utils.py`: Used by **all 16 scripts** (100%)
- `masking_utils.py`: Used by **14 scripts** (87%)
- `eval_utils.py`: Used by **all 16 scripts** (100%)
- `data_utils.py`: Used by **16 scripts** with task-specific formatters (100%)

**Total lines of code saved:** ~60-70% through utilities

---

## 🔧 Common CLI Arguments (All Scripts)

```bash
# Model configuration
--model_name Qwen/Qwen3-4B-Instruct-2507
--trust_remote_code  # Flag, no value needed

# Quantization (mutually exclusive)
--load_in_4bit       # Recommended for 4B models
--load_in_8bit       # Alternative

# LoRA configuration
--use_lora           # Flag to enable LoRA
--lora_r 16          # LoRA rank
--lora_alpha 32      # LoRA alpha (usually 2*r)
--lora_dropout 0.05
--lora_target_modules q_proj k_proj v_proj o_proj  # Space-separated

# Data configuration
--dataset_name imdb  # HuggingFace dataset
--dataset_config default  # Dataset configuration (if applicable)
--max_length 512     # Max sequence length
--train_split train  # Training split name
--eval_split test    # Eval split name

# Training configuration
--output_dir ./outputs/my_run
--num_epochs 3
--batch_size 4
--gradient_accumulation_steps 4  # Effective batch = batch_size * accumulation
--learning_rate 2e-4
--weight_decay 0.01
--warmup_steps 100
--warmup_ratio 0.1   # Alternative to warmup_steps

# Mixed precision
--bf16               # BFloat16 (recommended for Ampere+ GPUs)
--fp16               # Float16 (alternative)

# Logging & checkpointing
--logging_steps 10
--eval_steps 500
--save_steps 500
--save_total_limit 3  # Keep only last 3 checkpoints

# Other
--seed 42
--resume_from_checkpoint ./outputs/checkpoint-1000
```

---

## 📈 Memory & Performance Estimates

### Text Models (Qwen3-4B)

| Configuration | VRAM | Training Speed | Batch Size | Quality |
|---------------|------|----------------|------------|---------|
| Full FP32 | 48GB | 1x | 1 | Best |
| Full FP16 | 24GB | 2x | 2-4 | Best |
| Full BF16 | 24GB | 2x | 2-4 | Best |
| 8-bit + LoRA | 12GB | 1.5x | 4-8 | Very Good |
| 4-bit + LoRA | 8GB | 0.7x | 8-16 | Good |

**Recommended:** 4-bit + LoRA (r=16) for single GPU

### Multimodal Models (Qwen2-VL-2B)

| Configuration | VRAM | Batch Size | Notes |
|---------------|------|------------|-------|
| Full FP16 | 16GB + images | 2-4 | Image batch memory varies |
| 8-bit + LoRA | 8GB + images | 4-8 | Recommended |
| 4-bit + LoRA | 6GB + images | 8-16 | Best for consumer GPUs |

**Image memory:** ~100-500MB per image depending on resolution
**Recommended:** Resize to 384x384 and use 4-bit + LoRA

---

## 🎨 Loss Masking Patterns Quick Reference

### 1. Classification/MCQ/NER (Mask Prompt, Keep Answer)

```python
# Input:  "Question: What is 2+2?\nAnswer: 4"
# Mask:   "======= All masked ======   Keep"
# Labels: [-100, -100, -100, ..., 4_token_id]

def apply_masking(input_ids, labels, prompt_length):
    labels[:prompt_length] = -100
    return labels
```

**Used in:** Scripts 1, 3, 4, 5, 9

---

### 2. Chat (Mask User, Keep Assistant)

```python
# Input:  "USER: Hello ASSISTANT: Hi! USER: How are you? ASSISTANT: Great!"
# Mask:   "==== All === ========  Keep === All ======= ========  Keep"
# Labels: [-100, ..., -100, Hi_tokens, -100, ..., Great_tokens]

def apply_masking(input_ids, labels, tokenizer):
    # Find ASSISTANT: tokens
    # Set labels to -100 for everything except assistant responses
    return masked_labels
```

**Used in:** Scripts 7, 10

---

### 3. Instruction (Mask Instruction, Keep Response)

```python
# Input:  "### Instruction:\nWrite a poem\n### Response:\nRoses are red..."
# Mask:   "======= All masked ========= ====      Keep all response"
# Labels: [-100, -100, ..., -100, response_tokens...]

def apply_masking(input_ids, labels, instruction_length):
    labels[:instruction_length] = -100
    return labels
```

**Used in:** Scripts 2, 6, 8

---

### 4. CoT/Thinking (Keep Reasoning + Answer)

```python
# Input:  "Question: 2+2?\n<think>Add 2 and 2</think>\nAnswer: 4"
# Mask:   "==== Masked === Keep reasoning content  Keep answer"
# Labels: [-100, ..., think_tags_kept_or_masked, reasoning_tokens, answer_tokens]

def apply_masking(input_ids, labels, question_length):
    labels[:question_length] = -100
    # Optionally mask <think> and </think> tokens themselves
    return labels
```

**Used in:** Scripts 8, 11, 12

---

## 🧪 Critical Testing Checklist

Before running full training, verify:

- [ ] **Model loads correctly** with quantization
  ```python
  model = load_model_and_tokenizer(model_name, load_in_4bit=True)
  print(model.dtype)  # Should show qint4 or similar
  ```

- [ ] **LoRA applied to correct layers**
  ```python
  model = setup_lora(model, r=16)
  model.print_trainable_parameters()  # Should show <1% trainable
  ```

- [ ] **Chat template exists or fallback applied**
  ```python
  print(tokenizer.chat_template)  # Should not be None
  ```

- [ ] **Dataset loads and formats correctly**
  ```python
  dataset = load_dataset(dataset_name)
  print(dataset[0])  # Verify structure
  formatted = format_function(dataset[0])
  print(formatted)  # Verify formatting
  ```

- [ ] **Tokenization works properly**
  ```python
  tokens = tokenizer(formatted["text"], return_tensors="pt")
  print(tokens.input_ids.shape)  # Should be reasonable length
  ```

- [ ] **Loss masking is correct** ⚠️ CRITICAL
  ```python
  # Manually check first batch
  batch = next(iter(dataloader))
  print("Input IDs:", batch["input_ids"][0])
  print("Labels:", batch["labels"][0])
  print("Decoded input:", tokenizer.decode(batch["input_ids"][0]))
  # Verify that -100 appears in the right places
  ```

- [ ] **Forward pass works**
  ```python
  outputs = model(**batch)
  print("Loss:", outputs.loss)  # Should be positive float
  ```

- [ ] **Training step completes**
  ```python
  loss.backward()
  optimizer.step()
  print("Step completed")
  ```

- [ ] **GPU memory usage acceptable**
  ```python
  print(torch.cuda.memory_allocated() / 1e9, "GB")
  ```

---

## 🐛 Common Issues & Solutions

### Issue 1: "CUDA out of memory"
**Solutions:**
1. Reduce batch_size: `--batch_size 1`
2. Increase gradient_accumulation: `--gradient_accumulation_steps 16`
3. Enable gradient checkpointing in model_utils: `model.gradient_checkpointing_enable()`
4. Reduce max_length: `--max_length 256`
5. Use more aggressive quantization: `--load_in_4bit` instead of `--load_in_8bit`

### Issue 2: "Loss is NaN"
**Solutions:**
1. Check loss masking (likely all labels are -100)
2. Reduce learning rate: `--learning_rate 1e-4`
3. Use bf16 instead of fp16: `--bf16` (no --fp16)
4. Add gradient clipping: `clip_grad_norm_(model.parameters(), 1.0)`

### Issue 3: "Model not learning (loss not decreasing)"
**Solutions:**
1. Verify loss masking is correct (print first batch)
2. Check learning rate isn't too low: `--learning_rate 2e-4` or higher
3. Verify dataset formatting (prompt + completion structure)
4. Check that LoRA is actually enabled: `model.print_trainable_parameters()`

### Issue 4: "Chat template not found"
**Solutions:**
1. Verify fallback is being applied: Add logging
2. Manually set template in code if needed
3. Check model variant (base models usually don't have chat template)

### Issue 5: "Tokenizer padding issues"
**Solutions:**
1. Set pad token explicitly: `tokenizer.pad_token = tokenizer.eos_token`
2. Ensure padding in collator: `padding=True` in tokenizer call
3. For batch generation, set: `model.generation_config.pad_token_id = tokenizer.eos_token_id`

---

## 📦 Complete requirements.txt

```
torch>=2.0.0
transformers>=4.36.0
peft>=0.7.0
accelerate>=0.25.0
bitsandbytes>=0.41.0
datasets>=2.16.0
sentencepiece>=0.1.99
protobuf>=3.20.0
pillow>=10.0.0
scipy>=1.11.0

# Evaluation metrics
scikit-learn>=1.3.0
seqeval>=1.2.2
rouge-score>=0.1.2
nltk>=3.8.0

# Logging & utilities
wandb>=0.16.0  # Optional, for experiment tracking
tensorboard>=2.15.0  # Optional, alternative to wandb
tqdm>=4.66.0
```

Install with:
```bash
pip install -r requirements.txt
```

---

## 🚀 Quick Start Commands

### Script 1: Text Classification
```bash
python scripts/text/finetune_classification.py \
    --model_name Qwen/Qwen3-4B-Instruct-2507 \
    --dataset_name imdb \
    --output_dir ./outputs/classification \
    --load_in_4bit --use_lora --lora_r 16 \
    --num_epochs 3 --batch_size 4 --gradient_accumulation_steps 4 \
    --learning_rate 2e-4 --bf16 --seed 42
```

### Script 5: MCQ
```bash
python scripts/text/finetune_mcq.py \
    --model_name Qwen/Qwen3-4B-Instruct-2507 \
    --dataset_name allenai/ai2_arc --dataset_config ARC-Easy \
    --output_dir ./outputs/mcq \
    --load_in_4bit --use_lora --lora_r 16 \
    --num_epochs 5 --batch_size 2 --gradient_accumulation_steps 8 \
    --learning_rate 1e-4 --bf16 --seed 42
```

### Script 7: Chat
```bash
python scripts/text/finetune_chat.py \
    --model_name Qwen/Qwen3-4B-Instruct-2507 \
    --dataset_name OpenAssistant/oasst1 \
    --output_dir ./outputs/chat \
    --load_in_4bit --use_lora --lora_r 32 \
    --num_epochs 3 --batch_size 2 --gradient_accumulation_steps 8 \
    --learning_rate 2e-4 --bf16 --seed 42
```

### Script 11: Thinking/CoT
```bash
python scripts/text/finetune_thinking.py \
    --model_name Qwen/Qwen3-4B-Thinking-2507 \
    --dataset_name gsm8k --dataset_config main \
    --output_dir ./outputs/thinking \
    --load_in_4bit --use_lora --lora_r 32 \
    --num_epochs 5 --batch_size 2 --gradient_accumulation_steps 8 \
    --learning_rate 1e-4 --bf16 --seed 42
```

### Script 15: VQA
```bash
python scripts/multimodal/finetune_vqa.py \
    --model_name Qwen/Qwen2-VL-2B-Instruct \
    --dataset_name HuggingFaceM4/VQAv2 \
    --output_dir ./outputs/vqa \
    --load_in_4bit --use_lora --lora_r 16 \
    --num_epochs 3 --batch_size 1 --gradient_accumulation_steps 16 \
    --learning_rate 1e-4 --bf16 --seed 42
```

---

## 📊 Success Criteria

### Minimum Viable Implementation
- [ ] All 16 scripts can be invoked without errors
- [ ] Model loads with quantization and LoRA
- [ ] Training loop completes for at least 10 steps
- [ ] Loss decreases over training
- [ ] Checkpoints can be saved and loaded
- [ ] Basic evaluation runs and produces metrics

### Good Implementation
- [ ] Loss masking tested and verified correct
- [ ] Chat template fallback works
- [ ] Evaluation metrics match expected baselines
- [ ] Scripts documented with example commands
- [ ] Utilities are well-tested
- [ ] Memory usage optimized

### Excellent Implementation
- [ ] Comprehensive unit tests for critical functions
- [ ] Multiple dataset formats supported per script
- [ ] Experiment tracking (wandb/tensorboard) integrated
- [ ] Multi-GPU support tested
- [ ] Error handling and helpful error messages
- [ ] Performance baselines documented

---

## ⏰ Time Estimates

| Task | Time | Cumulative |
|------|------|------------|
| Setup environment | 0.5 days | 0.5 days |
| Core utilities (model, args) | 1 day | 1.5 days |
| Script 1 (Classification) | 1 day | 2.5 days |
| Testing & debugging Script 1 | 0.5 days | 3 days |
| Masking utilities | 1 day | 4 days |
| Scripts 2-4 (NER, MCQ, Instruction) | 2 days | 6 days |
| Scripts 5-8 (Chat, CoT, variants) | 2 days | 8 days |
| Evaluation utilities | 1 day | 9 days |
| Scripts 9-10 (Thinking) | 1 day | 10 days |
| VL utilities | 1 day | 11 days |
| Scripts 11-12 (VQA) | 2 days | 13 days |
| Scripts 13-16 (Variants) | 2 days | 15 days |
| Testing & bug fixes | 2 days | 17 days |
| Documentation | 1 day | 18 days |
| Buffer for unexpected issues | 2 days | **20 days** |

**Total: ~4 weeks for single developer**

---

## 🎯 Next Immediate Actions

1. **Verify model access**
   ```bash
   python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('Qwen/Qwen3-4B-Instruct-2507')"
   ```

2. **Create project structure**
   ```bash
   mkdir -p scripts/text scripts/multimodal utils configs tests examples
   touch utils/__init__.py
   ```

3. **Install dependencies**
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install torch transformers peft accelerate bitsandbytes datasets
   ```

4. **Start with utils/model_utils.py**
   - Implement `load_model_and_tokenizer()`
   - Test with Qwen3-4B-Instruct-2507
   - Verify 4-bit loading works

5. **Create first script: finetune_classification.py**
   - Use simple prompt-completion format
   - Test with 100 IMDb examples
   - Verify training loop works

6. **Iterate based on results**
   - Fix any issues found
   - Expand to other scripts
   - Build utilities incrementally
