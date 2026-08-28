# DeepSpeed ZeRO / FSDP (training, memory)

Status: **done (Wave 6)** for ZeRO Stage 2 CPU offload on a single GPU; FSDP/multi-GPU sharding remains genuinely out of scope on this hardware (see below).

`deepspeed_zero_benchmark.py` compares plain AdamW (all optimizer state on
GPU) against DeepSpeed ZeRO Stage 2 with the optimizer state offloaded to
CPU RAM, on a real forward + backward + optimizer step.

## Why this isn't a sharding benchmark

ZeRO's headline use case -- and FSDP's -- is sharding optimizer/gradient/
parameter state **across multiple GPUs**. This project's hardware is a
single 8GB GPU (`world_size=1`); there's nothing to shard across, so that
specific benefit genuinely doesn't apply here, and no benchmark on this
hardware can honestly demonstrate it. What DOES apply on a single GPU is
**ZeRO-Offload**: moving optimizer state to CPU RAM instead of GPU memory,
trading GPU memory for host RAM + PCIe transfer time -- a real, testable
technique, and what this script measures instead.

Uses this project's own from-scratch CLM checkpoint
(`./output/pretraining/clm`, 51M params) rather than a larger model:
ZeRO-Offload's benefit is proportional to optimizer state SIZE, and this
machine's available CPU RAM (~13GB free, checked via `free -h` before
writing this script) can't comfortably hold a 1.7B-parameter model's full
Adam state (~13.6GB) alongside everything else already resident. The
small CLM checkpoint's Adam state (~408MB) is a size CPU-offload can
actually demonstrate cleanly here without becoming the bottleneck itself.

## Real environment bug found and worked around

DeepSpeed's CPU-offloaded optimizer defaults to `DeepSpeedCPUAdam`, a
fused CPU/CUDA kernel that JIT-compiles on first use. That compilation
failed here with `CUDAMismatchException: Installed CUDA version 13.3 does
not match the version torch was compiled with 13.0` -- a real toolkit/
PyTorch-build mismatch on this machine, not fixable by changing this
project's code alone. Worked around via DeepSpeed's own documented escape
hatch (found because DeepSpeed's error message for the *converse* failure
mode names this exact flag): `zero_force_ds_cpu_optimizer: false` plus
`torch_adam: true` in the optimizer config, which falls back to plain
`torch.optim.AdamW` for the CPU-offloaded optimizer instead of the
JIT-compiled kernel. This works correctly but is slower than the fused
kernel would have been -- the slowdown number below is pessimistically
biased by this environment constraint, not purely inherent to CPU
offloading itself.

## Real result (`./output/pretraining/clm`, batch_size=8, seq_len=256)

| | Peak GPU memory | Avg step time |
|---|---|---|
| baseline (plain AdamW, GPU) | 2266.0 MB | 0.0607s |
| ZeRO Stage 2 (CPU-offloaded optimizer) | 2088.5 MB (-7.8%) | 0.2005s (3.30x slower) |

A modest memory win, a substantial time cost -- both genuinely expected
here, not surprising: this small model's Adam state (~408MB) is only a
slice of the ~2.2GB total GPU footprint (weights + activations +
gradients + optimizer state combined), so offloading just that slice
removes a modest fraction of total memory, while paying a real CPU<->GPU
transfer cost every step (worsened by the non-fused `torch_adam` fallback
above). **The takeaway this result actually supports**: ZeRO-Offload is
worth it when the model is large enough that optimizer state dominates
GPU memory (which is exactly the regime where offloading it away matters
most) -- for a small model like this one, the technique's overhead isn't
worth its modest memory return, and QLoRA-style base-model quantization
(see `../quantization-4bit-8bit/README.md`, 34-46% memory reduction with
a smaller speed penalty) is the better lever on this hardware for the
model scales this project actually trains at.

## Usage

```bash
python deepspeed_zero_benchmark.py --debug_first_batch
python deepspeed_zero_benchmark.py --batch_size 8 --seq_len 256 --num_steps 10
```

No dataset needed -- see `../../execution-time/flash-attention/README.md`
for why synthetic batches are appropriate for a compute/memory benchmark
like this one.
