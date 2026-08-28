# KV-Cache / Paged Attention (inference, execution-time)

Status: **done (Wave 6)**.

`kv_cache_benchmark.py` compares generation with `use_cache=True` (the
default -- each step reuses cached key/value projections from all
previous tokens) against `use_cache=False` (every step recomputes
attention over the entire sequence so far from scratch), via real
generation and wall-clock timing on `Qwen/Qwen3-1.7B-Base`.

**Paged attention specifically** (solving KV-cache memory *fragmentation*
across many concurrent requests -- vLLM's core contribution) is a serving
-infrastructure technique that only matters once you're managing a
KV-cache pool across many simultaneous requests of different lengths; a
single-request `transformers.generate()` call can't demonstrate it in
isolation. That's exactly what `production/serve_and_benchmark.py`
already exercises (vLLM's engine uses paged attention internally for
every request it serves) -- see that script's README for real serving
throughput. This script measures the more fundamental, framework-agnostic
piece underneath: does caching itself matter, and by how much.

## Real result (3 MedMCQA dev prompts, greedy, max_new_tokens=32)

| | Tokens | Wall-clock | Throughput |
|---|---|---|---|
| `use_cache=True` | 77 | 14.70s | 5.2 tok/s |
| `use_cache=False` | 77 | 262.11s | 0.3 tok/s |

**17.83x speedup from caching alone.** This is the largest speedup found
by any benchmark in this project's `optimization/` folder so far -- and
expected to be: without caching, generating token *N* recomputes attention
over all *N-1* previous tokens from scratch, making a `max_new_tokens`-long
generation cost `O(max_new_tokens^2)` total work instead of `O(max_new_tokens)`.
`--num_prompts`/`--max_new_tokens` default lower than most benchmarks in
this project specifically because the uncached arm is this expensive --
an early full-scale run (10 prompts, 64 tokens) was killed after 590s
without finishing the uncached arm alone, which is itself a real data
point about how unusable naive uncached generation is at any realistic
length, not a benchmark-script problem to fix.

## Usage

```bash
python kv_cache_benchmark.py --debug_first_batch
python kv_cache_benchmark.py --num_prompts 3 --max_new_tokens 32
```

## Dataset Reference (this folder's slice of the root README's master table)

| Script | Dataset | Split(s) | Role |
|---|---|---|---|
| `kv_cache_benchmark.py` | `araag2/MedMCQA` (config `processed`) | `dev` (generation prompts only) | same split/prompt-format as `rlhf/grpo/grpo.py` |
