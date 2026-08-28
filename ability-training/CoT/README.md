# Chain-of-Thought / Reasoning Induction

Status: **done (Wave 4)**.

Induces `<think>...</think>` reasoning via RL with Verifiable Rewards
(RLVR), DeepSeek-R1-Zero-style: no SFT-based reasoning demonstrations at
all, just RL pressure from a correctness reward plus a format reward for
producing a well-formed think block. `reasoning_induction_grpo.py` reuses
the exact GRPO mechanism and dataset
(`rlhf/grpo/grpo.py`,
`araag2/MedMCQA`) but adds a second reward component and tracks whether
`<think>` usage emerges over training via `evaluate_cot_usage_rate()`.

## Reward design

- `correctness_reward(prompts, completions, answer, **kwargs)` -- same
  verifiable letter-match reward as `grpo/grpo.py`, but reads the answer
  letter from *after* the `</think>` tag if one is present (so a model that
  reasons before answering isn't penalized for the reasoning text itself).
- `format_reward(prompts, completions, **kwargs)` -- `1.0` if a
  `<think>...</think>` block is present with >= 3 words of content inside,
  else `0.0`. The word-count floor exists specifically so the model can't
  farm format reward with an empty or filler think block (e.g. `<think>hmm</think>`)
  -- verified via `--debug_first_batch`, see "Real findings" below.
- Combined via `GRPOConfig(reward_funcs=[correctness_reward, format_reward],
  reward_weights=[1.0, args.format_reward_weight])`, default
  `--format_reward_weight 0.3`.

## Real findings from a real training run

Ran `--max_samples 32 --max_eval_samples 20 --lora --quantization 4bit
--batch_size 4 --gradient_accumulation_steps 2 --num_generations 4
--max_completion_length 150` (16 optimizer steps, ~3.5 min):

- `correctness_reward` improved to a final eval mean of **0.475** --
  comparable to plain `grpo/grpo.py`'s 0.425 on the same dataset/model,
  confirming the added format reward doesn't come at the expense of
  correctness learning even at this tiny scale.
- `format_reward` stayed at exactly **0.0** the entire run --
  `evaluate_cot_usage_rate()` reported **0.00% `<think>` usage after
  training**. The model never spontaneously discovered `<think>` tag usage
  in 16 steps. This is a genuine (not a bug) negative result: DeepSeek-R1-Zero-style
  emergent reasoning is reported to take thousands of RL steps before format
  reward gets meaningful gradient signal -- 16 steps on a `--max_samples 32`
  toy run is nowhere near that regime. Treat this run as a pipeline
  correctness check (rewards computed correctly, training loop runs,
  adapter saves), not as a claim that RLVR-induced reasoning doesn't work
  -- it would need a much longer run (thousands of steps, full dataset) to
  actually test emergence.
- `eval_completions/clipped_ratio` was `0.9` -- most completions hit
  `--max_completion_length` without terminating. Expected at this
  undertrained a checkpoint; worth raising `--max_completion_length` and/or
  training longer before drawing conclusions from completion length stats.

This result is exactly the kind of question `experiments/` exists to
formalize (does CoT-inducing RL actually beat no-CoT on this task, and at
what step count does `<think>` usage emerge) -- a longer run here is a
natural candidate to extend `experiments/cot-vs-no-cot-classification/` or
a new experiment comparing RLVR-induced CoT against the existing
SFT-based `*_cot.py` scripts in `supervised-finetuning/`.

## Usage

```bash
python reasoning_induction_grpo.py --debug_first_batch --max_samples 8
python reasoning_induction_grpo.py --lora --quantization 4bit --max_samples 500   # needed to fit 8GB VRAM, see RLHF parent README
```

## Dataset Reference (this folder's slice of the root README's master table)

| Script | Dataset | Split(s) | Role |
|---|---|---|---|
| `reasoning_induction_grpo.py` | `araag2/MedMCQA` (config `processed`) | `train` (training), `dev` (eval) | primary + only corpus |
