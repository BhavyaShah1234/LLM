# Speculative Decoding (inference, execution-time)

Status: **done (Wave 6)**.

`speculative_decoding_benchmark.py` compares plain autoregressive
generation against HF `generate()`'s built-in `assistant_model` speculative
decoding, using `Qwen/Qwen3-1.7B-Base` as target and `Qwen/Qwen3-0.6B-Base`
as draft (same tokenizer/vocab_size, confirmed before implementing --
speculative decoding requires the draft and target to share a vocabulary).

## Real result (10 MedMCQA dev prompts, greedy, max_new_tokens=64)

| | Tokens generated | Wall-clock | Throughput |
|---|---|---|---|
| Plain generation | 410 | 46.12s | 8.9 tok/s |
| Speculative decoding | 369 | 68.49s | 5.4 tok/s |

**Speculative decoding was 1.48x SLOWER here, not faster** -- a genuine,
measured result, not a bug in this benchmark. A single-prompt
`--debug_first_batch` check earlier looked like a speedup (3.67s vs.
5.69s), which is exactly why this script runs the full 10-prompt benchmark
rather than trusting a 1-sample debug run's timing -- at that small a
sample, noise can (and did) point the wrong direction.

**Why this is a real, expected-in-hindsight outcome, not a bug**:
speculative decoding's speedup depends entirely on the target model
*accepting* most of what the draft proposes -- when it does, one target
forward pass verifies several tokens at once; when it doesn't, you pay for
both the draft's forward passes AND the target's verification passes
without saving any target-side work. `Qwen/Qwen3-0.6B-Base` and
`Qwen/Qwen3-1.7B-Base` are two **independently pretrained** checkpoints,
not a draft *distilled from* the target the way production speculative-decoding
setups usually pair them -- there's no guarantee their token-level
predictions agree often enough for a high acceptance rate, and this
result suggests they don't, on this task/prompt style. This is a known,
realistic caveat of the technique, not specific to this project's
implementation: speculative decoding is a real win only when draft and
target are well-aligned (ideally the same model family with the draft
distilled from the target, or the target's own quantized/pruned self),
not merely "any smaller model from the same family."

## Usage

```bash
python speculative_decoding_benchmark.py --debug_first_batch
python speculative_decoding_benchmark.py --num_prompts 10 --max_new_tokens 64
```

## Dataset Reference (this folder's slice of the root README's master table)

| Script | Dataset | Split(s) | Role |
|---|---|---|---|
| `speculative_decoding_benchmark.py` | `araag2/MedMCQA` (config `processed`) | `dev` (generation prompts only) | same split/prompt-format as `rlhf/grpo/grpo.py` |
