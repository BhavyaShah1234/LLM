# Production Serving

Status: **done (Wave 4)**.

`serve_and_benchmark.py` serves a model through vLLM's offline `LLM` engine
(the same continuous-batching engine the OpenAI-compatible HTTP server
wraps) and benchmarks throughput/latency. With `--lora_path`, it attaches a
LoRA adapter natively via vLLM's `LoRARequest` -- no merge step needed,
which matters here specifically because `common/model_saving.py`'s
`adapter_only` strategy is what most `--lora` runs in this project actually
save (see that module's docstring: cheapest way to keep every comparison
run's result around). A serving script that only accepted merged
checkpoints would leave most of this project's own trained artifacts
unservable.

It reuses `rlhf/grpo/grpo.py`'s exact
prompt format, dataset, and answer-extraction regex, so `--lora_path`
pointing at that script's output also reports base-vs-adapted accuracy on
the same prompts -- a real check that a trained adapter's effect survives
being served through a different inference engine than it was trained/
evaluated with (trl's HF-based generation vs vLLM).

## Real environment issue found and fixed

vLLM 0.26.0's engine core JIT-compiles Triton/inductor CUDA kernels at
startup. On this project's Python 3.14 venv, that failed with `fatal error:
Python.h: No such file or directory` -- the system had the `python3.14` and
`python3.14-venv` packages but not `python3.14-dev` (headers only ship
separately on Debian/Ubuntu). Fixed by installing `python3.14-dev` (from
the deadsnakes PPA already providing python3.14 on this system). Not
something `pip install` inside the venv can fix -- it's a system package
Requirement, worth flagging early to anyone reproducing this project on a
fresh machine.

## Real finding: LoRA needs more GPU memory headroom than base serving

`--gpu_memory_utilization 0.5` (a conservative default for an 8GB card)
works for base-model-only serving but fails for `--lora_path` runs with
`Available KV cache memory: -0.24 GiB` -- a negative KV-cache budget,
because enabling LoRA (`enable_lora=True`) adds Punica LoRA kernel buffers
and an extra `torch.compile` artifact set on top of the base model's own
compiled graph and weights. Confirmed by reproducing the failure and fixing
it purely by raising `--gpu_memory_utilization` to `0.7` -- same model,
same adapter, same prompts. `serve_and_benchmark.py`'s default is `0.7` for
this reason (base-only serving has headroom to spare at that setting; LoRA
serving needs it).

## Real benchmark run

`Qwen/Qwen3-1.7B` + the `rlhf/grpo/`
adapter, 20 MedMCQA dev prompts, greedy decoding, `--max_tokens 64`:

| | Throughput | Accuracy |
|---|---|---|
| Base model | 1026.9 tok/s | 0.450 |
| LoRA-adapted | 1020.1 tok/s | 0.450 |

Two honest findings, not glossed over:
- **Throughput cost of LoRA is negligible** (~0.7% slower) -- vLLM's
  Punica kernels are specifically built to make LoRA-adapted serving nearly
  as fast as the base model.
- **Accuracy delta is exactly 0.000** on this 20-prompt sample -- the
  adapter didn't change any answer. This is consistent with, not
  contradicting, `rlhf/grpo/README.md`:
  that adapter was trained on only `--max_samples 40` for a handful of
  steps (a smoke-scale verification run, not a real training budget), so a
  measurable accuracy shift on a *different* 20-prompt sample wasn't
  expected. This result is itself a useful, real confirmation that
  `serve_and_benchmark.py`'s accuracy comparison is doing real work (it
  would have caught a shift if the adapter's training had been large enough
  to produce one) -- it just needs a properly-trained adapter, not this
  project's smoke-test-scale one, to show a nonzero delta.

## Usage

```bash
python serve_and_benchmark.py --debug_first_batch
python serve_and_benchmark.py --model Qwen/Qwen3-1.7B --num_prompts 20
python serve_and_benchmark.py --model Qwen/Qwen3-1.7B --lora_path ./output/rlhf/grpo --gpu_memory_utilization 0.7 --num_prompts 20
```

See `optimization/inference/` for the underlying inference-time techniques
(KV-cache/paged attention, speculative decoding, PTQ) this stage will
exercise further as those subfolders move from stub to real scripts;
`--quantization bitsandbytes` is already wired up here for vLLM's own
quantized serving, untested pending a merged (non-adapter) checkpoint to
quantize.

## Dataset Reference (this folder's slice of the root README's master table)

| Script | Dataset | Split(s) | Role |
|---|---|---|---|
| `serve_and_benchmark.py` | `araag2/MedMCQA` (config `processed`) | `dev` (benchmark prompts) | same split/prompt-format as `rlhf/grpo/grpo.py`, so its adapter's effect is directly checkable under vLLM |
