# Optimization

Status: **done (Waves 4-6)** -- every subfolder in the taxonomy is now real (11 techniques via 11 scripts, plus `distillation/{execution-time,storage}` and `other/{memory,storage}` cross-referencing sibling scripts per this folder's own "one primary placement" policy rather than duplicating them).

See root README.md section 9b ("optimization/ design note") for the full rationale. Short version: every technique here also exists as a --flag on relevant task scripts elsewhere (via common/) for getting a task done efficiently; the scripts in this folder instead study each technique in isolation (VRAM/speed/accuracy/disk measured directly). Categorized training/ | inference/ | distillation/ | other/, each split into execution-time/ | memory/ | storage/ by primary resource optimized. Dual-purpose techniques (e.g. mixed precision, FlashAttention, PTQ) get one primary placement with the secondary benefit cross-referenced, not duplicated -- see the technique-to-resource-axis matrix in the root README once populated.

## What's built

| Folder | Technique | Real finding | Status |
|---|---|---|---|
| `training/memory/quantization-aware-training/` + `inference/memory/post-training-quantization/` | QAT vs. PTQ | At 8-bit, PTQ is already near-lossless -- QAT has nothing to recover. At 4-bit, QAT roughly halves PTQ's degradation. | **done** |
| `training/execution-time/flash-attention/` | eager vs. SDPA attention | 1.05x speedup / 6.1% memory reduction at seq_len=256; both implementations OOM at seq_len=1024 for a reason unrelated to attention (vocab-size-driven loss memory). | **done** |
| `training/memory/mixed-precision/` | fp32 vs. fp16 vs. bf16 | fp16/bf16: ~2.5x speedup, 27.6% memory reduction vs fp32, essentially identical to each other. | **done** |
| `training/memory/gradient-checkpointing/` | with vs. without checkpointing | 1.13x slower, 23.5% peak memory reduction -- a real memory-for-time trade, the only technique in this wave that isn't a free win. | **done** |
| `training/memory/lora-qlora/` | full fine-tune vs. LoRA vs. QLoRA | Full fine-tuning of a 1.7B model genuinely OOMs on 8GB hardware (optimizer state, not activations). QLoRA uses 44% less memory than LoRA but is 2.24x slower (dequantization overhead). | **done** |
| `training/memory/quantization-4bit-8bit/` | bnb 4-bit vs. 8-bit for LoRA training | 4-bit beats 8-bit on BOTH memory (-45.7% vs -34.9%) AND speed (2.05x vs 3.21x slowdown) -- bnb's 8-bit kernel path is less optimized than its 4-bit path. | **done (Wave 6)** |
| `training/memory/deepspeed-zero-fsdp/` | ZeRO Stage 2 CPU offload (single-GPU -- sharding across GPUs doesn't apply here) | Only 7.8% memory reduction, 3.30x slower -- honest result for a small model where optimizer state is a small slice of total footprint; the technique is for larger models where it dominates. | **done (Wave 6)** |
| `training/storage/checkpoint-compression/` | fp32 vs. bf16 vs. gzip(bf16) | bf16 is a free 49.1% size win over fp32. gzip adds 23.2% more but costs ~22x the load-plain-bf16 time in decompression -- good for archival, bad for repeated loads. | **done (Wave 6)** |
| `inference/execution-time/speculative-decoding/` | draft + target generation | 1.48x SLOWER at full scale with independently-pretrained draft/target models -- a real negative result: speculative decoding needs a well-aligned (e.g. distilled) draft to pay off, not just "any smaller model." | **done (Wave 6)** |
| `inference/execution-time/kv-cache-paged-attention/` | use_cache=True vs. False | 17.83x speedup -- the largest in this folder. Paged attention itself (multi-request KV-cache management) is cross-referenced to `production/serve_and_benchmark.py`'s vLLM-engine numbers. | **done (Wave 6)** |
| `inference/storage/quantized-checkpoint-storage/` | real (not estimated) bnb-4bit/8bit saved checkpoint size | 4-bit: 35.3% smaller on disk (capped below the naive 4x ceiling by this checkpoint's 50.6% embedding-weight share, which bnb doesn't quantize). | **done (Wave 6)** |
| `distillation/memory/` (+ `execution-time/`, `storage/` cross-referenced) | teacher-student ViT distillation | 29.1x faster, 99.3% smaller -- but only 26.3% accuracy vs. the teacher's 82.3% at this toy training budget; an honest efficiency-vs-accuracy tradeoff, not oversold. | **done (Wave 6)** |
| `other/execution-time/pruning/` (+ `memory/`, `storage/` cross-referenced) | structured MLP-width pruning | 1.22x speedup, 16.4% fewer params, +10.61 perplexity cost with no recovery finetuning -- unstructured (mask-based) pruning was deliberately avoided since it gives no real speedup on dense hardware. | **done (Wave 6)** |

Every technique in the original taxonomy now has a real script (or, for
techniques spanning multiple resource axes, one real script with the
other axis folders cross-referencing it rather than duplicating the run).
