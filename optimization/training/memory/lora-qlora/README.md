# LoRA / QLoRA (training, memory)

Status: **done (Wave 5)**.

`lora_qlora_benchmark.py` compares three ways of training the same model
on a real optimizer step (forward + backward + `optimizer.step()`, unlike
the other benchmarks in this wave, which only time forward+backward -- see
the module docstring for why a real optimizer step is the point here):
`full` (every parameter trainable), `lora` (frozen bf16 base + small
trainable adapter), `qlora` (frozen 4-bit base + trainable adapter, this
project's `--quantization 4bit --lora` combination used throughout
`supervised-finetuning/` and `rlhf/`).

## Real result (batch_size=1, seq_len=256, `Qwen/Qwen3-1.7B-Base`)

| Config | Trainable params | Peak memory | Avg step time |
|---|---|---|---|
| `full` | 1,720,574,976 (100%) | **OOM** | -- |
| `lora` | 6,422,528 (0.372%) | 4952.5 MB | 0.1332s |
| `qlora` | 6,422,528 (0.628%\*) | 2751.9 MB | 0.2980s |

\* `qlora`'s `total_params` denominator (1,022,354,432) is smaller than
`full`/`lora`'s (~1.72B) -- not because the model has fewer logical
parameters, but because bitsandbytes' 4-bit `Params4bit` tensors pack
multiple logical weights per stored element, and `sum(p.numel())` counts
storage elements. Treat `qlora`'s raw `total_params` as an artifact of the
packed representation, not a directly comparable parameter count -- the
`trainable_params` count (LoRA adapter only, never quantized) IS directly
comparable across all three configs.

**`full` genuinely OOMs on this project's 8GB target hardware** -- not a
bug, the concrete demonstration of why this project's model-selection
philosophy treats `--lora`/`--quantization` as effectively required (not
optional headroom) for full-parameter training at this model scale, a
convention already documented from earlier waves (see
`rlhf/README.md`) and now backed by
a direct number here: AdamW's per-trainable-parameter optimizer state
(~12-16 bytes/param) makes 1.7B trainable parameters cost tens of GB
regardless of how efficient the forward/backward pass itself is.

**`qlora` uses 44.4% less peak memory than `lora`, but is 2.24x slower.**
This is the real, well-known QLoRA tradeoff, now measured directly: 4-bit
NF4 weights must be dequantized on the fly for every matmul in the forward
pass (and again for the backward pass), which is real extra compute the
plain-bf16 `lora` config doesn't pay. QLoRA is worth it when memory, not
speed, is the binding constraint -- e.g. it's what makes
`rlhf/dpo/dpo.py`'s policy+reference
model pair fit on this hardware at all (see that folder's README), where
`lora` alone still OOM'd.

## Real bug found and fixed: OOM cleanup left GPU memory pinned across configs

The first version of this script's per-config `try/except
torch.OutOfMemoryError` didn't `del model` in the exception branch. After
`full`'s (expected) OOM, the half-allocated model stayed referenced by the
loop variable, so `torch.cuda.empty_cache()` had nothing to reclaim --
confirmed via a live cascading failure where `lora` (which fits on its own)
then OOM'd too, and `qlora` failed for a third, unrelated reason
(`device_map="auto"`'s memory probing saw almost no free GPU memory and
defensively offloaded some layers to CPU, which bitsandbytes 4-bit
rejects). Fixed with an explicit `del model` in a `finally` block before
`torch.cuda.empty_cache()`, guaranteeing cleanup runs whether or not the
config OOM'd. Verified fixed: `lora` and `qlora` both ran cleanly
immediately after `full`'s OOM in the same process, using the numbers
above.

## Usage

```bash
python lora_qlora_benchmark.py --debug_first_batch
python lora_qlora_benchmark.py --batch_size 1 --seq_len 256 --num_steps 5
```

No dataset needed -- see `../../execution-time/flash-attention/README.md`
for why synthetic batches are appropriate for a compute/memory benchmark
like this one.
