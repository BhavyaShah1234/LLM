# 4-bit / 8-bit Quantization for training, i.e. QLoRA-style (training, memory)

Status: **done (Wave 6)**.

`quantization_bits_benchmark.py` compares bitsandbytes `no` (bf16) vs.
`8bit` vs. `4bit` quantization of the FROZEN base model during LoRA
training, holding the LoRA config fixed throughout. This isolates a
different variable than `../lora-qlora/lora_qlora_benchmark.py`, which
held quantization fixed and varied whether LoRA was used at all (full vs.
LoRA vs. QLoRA) -- here LoRA is used in all three arms; only the base
model's bit-width changes. Real forward + backward + `optimizer.step()`,
same reasoning as the LoRA/QLoRA benchmark.

## Real result (`Qwen/Qwen3-1.7B-Base`, batch_size=2, seq_len=256)

| Quantization | Peak memory | Avg step time |
|---|---|---|
| bf16 (no quant) | 6330.9 MB | 0.2354s |
| 8-bit | 4119.9 MB (-34.9%) | 0.7554s (3.21x slower) |
| 4-bit | 3434.8 MB (-45.7%) | 0.4832s (2.05x slower) |

**4-bit beats 8-bit on BOTH axes** -- less memory (45.7% vs. 34.9%
reduction) AND faster (2.05x vs. 3.21x slowdown relative to bf16). This is
real and counter to the naive intuition that fewer bits should always cost
more compute for the same speed benefit. The likely explanation:
bitsandbytes' 4-bit NF4 path (`Params4bit`) is a more heavily-optimized,
newer CUDA kernel than its 8-bit path (`MatMul8bitLt`, which the library
itself warns casts inputs through fp16 during quantization -- visible
directly in this run's logs) -- a known characteristic of bitsandbytes,
not something specific to this project's setup. Both quantized options are
slower than bf16 (real dequantization overhead on every forward pass, as
also found in `../lora-qlora/README.md`), but if you're choosing a
bit-width specifically for training-time quantization on this kind of
hardware, this result argues for 4-bit over 8-bit on both memory and speed
grounds -- 8-bit isn't a "safer, slightly slower" middle ground here.

## Usage

```bash
python quantization_bits_benchmark.py --debug_first_batch
python quantization_bits_benchmark.py --batch_size 2 --seq_len 256 --num_steps 5
```

No dataset needed -- see `../../execution-time/flash-attention/README.md`
for why synthetic batches are appropriate for a compute/memory benchmark
like this one.
