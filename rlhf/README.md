# Reinforcement Learning with Human Feedback

Status: **DPO, GRPO, KTO, and reward-modeling all done (Waves 4-5)**.

## Prerequisite

DPO/GRPO/KTO assume the policy model already follows the target response
format/behavior -- running them directly on a raw pretrained model is not
workable in practice (there's nothing coherent yet to align). All three
default `--model` to `Qwen/Qwen3-1.7B` (the vendor
**instruction-tuned** sibling of this project's usual `Qwen/Qwen3-1.7B-Base`
default), NOT the base model. **`reward-modeling/` is the one exception**:
it trains a classifier head, not the policy itself, so it has no such
prerequisite and defaults `--model` to `Qwen/Qwen3-1.7B-Base` instead --
see that folder's README. This project's own `supervised-finetuning/`
instruction-tuning/chat-tuning scripts had only been smoke-tested
(`--debug_first_batch`), not actually trained to completion, at the time
this wave was built -- once one of them has been trained for real, point
`--model` at its output directory instead to see the "our own
instruction-tuning" starting point rather than the vendor one.

`experiments/rlhf-pretrained-vs-sft-init/` (done, Wave 4) empirically tests
the prerequisite itself: `grpo.py` run identically on `Qwen/Qwen3-1.7B`
(instruct) vs `Qwen/Qwen3-1.7B-Base` (raw pretrained). The reward metric
alone looks only slightly worse for the pretrained-init run (0.375 vs
0.425) -- but direct inspection of its generations shows it converged to a
degenerate shortcut, echoing option A's text back almost every time,
rather than learning the task. `eval_entropy` (~4x higher) and
`eval_frac_reward_zero_std` (much lower -- less consistency across sampled
completions) are the metrics that actually catch this; the reward number
alone would not have. See that experiment's README for the full
generations and analysis.

Uses `trl` (`DPOTrainer`/`DPOConfig`, `GRPOTrainer`/`GRPOConfig`,
`KTOTrainer`/`KTOConfig`, `RewardTrainer`/`RewardConfig`) --
`transformers` has no native RLHF trainer support, confirmed via
introspection.

## What's built

| Folder | Algorithm | What it needs | Status |
|---|---|---|---|
| `dpo/` | Direct Preference Optimization | Paired (chosen, rejected) preference data | **done** |
| `grpo/` | Group Relative Policy Optimization | A verifiable/rule-based reward function (RLVR) -- no learned reward model needed | **done** |
| `kto/` | Kahneman-Tversky Optimization | Unpaired binary (prompt, response, good/bad) feedback | **done (Wave 5)** |
| `reward-modeling/` | Reward model training | Paired preference data (same shape as DPO's) | **done (Wave 5)** |

## Real findings from building this wave (not anticipated during planning)

- **Memory**: `DPOTrainer` keeps a policy model AND a reference model in
  memory. Even with `--lora`, a real training run of `Qwen/Qwen3-1.7B`
  (fp16 ≈4GB) OOM'd on this project's 8GB GPU without `--quantization 4bit`
  -- two fp16 copies alone exceed 8GB before any activations/optimizer
  state. Unlike `supervised-finetuning/`, where `--quantization` is optional
  headroom, it's effectively required here at this model scale. GRPO's
  generation-heavy rollout (sampling `--num_generations` completions per
  prompt every step) has the same practical requirement.
- **API drift**: this project's pinned `trl==1.9.2` has no `max_prompt_length`
  field on either `DPOConfig` or `GRPOConfig` (only a combined `max_length`
  for DPO, and `max_completion_length` alone for GRPO) -- found via
  `TypeError` when following what looked like the expected API, confirmed by
  introspecting the actual dataclass fields.
- **A real, non-obvious dataset bug**: `araag2/MedMCQA`'s `test` split has
  `Label = None` for every row (labels withheld, standard practice to
  prevent leaderboard cheating). `grpo/grpo.py` originally evaluated against
  it, silently producing exactly 0.0 reward for every eval example
  regardless of model quality -- confirmed not to be a training-loop bug by
  directly instrumenting `GRPOTrainer`'s reward function during a live
  training step (which showed real, varying rewards on the `train` split).
  Fixed by evaluating against `dev` instead. **This same bug was found to
  already exist in `supervised-finetuning/text/mcq/decoder-only/mcq_standard.py`**
  (built in Wave 2, but never actually trained for real there, only
  smoke-tested -- which is why it went undetected until GRPO's live reward
  signal surfaced it) and fixed there too.
- **`KTOTrainer` requires actual (not effective) per-device batch size > 1**
  -- `--batch_size 1` with gradient accumulation (which works fine for
  DPO/GRPO) raises a `ValueError`, because KTO estimates its KL term from
  the real mini-batch; a batch of 1 gives a degenerate estimate. Fixed by
  using `--batch_size 2` with fewer accumulation steps for the same
  effective batch size. See `kto/README.md`.
- **Wrong PEFT `task_type` for reward modeling**: `common/peft_setup.py`'s
  `build_lora_config()` default (`task_type="CAUSAL_LM"`) crashes
  `reward_modeling.py` with `AttributeError: ... object has no attribute
  'prepare_inputs_for_generation'`, because `RewardTrainer` swaps the model
  for `AutoModelForSequenceClassification`, which isn't generation-capable.
  Fixed by passing `task_type="SEQ_CLS"` explicitly in that one script --
  `common/`'s default stays `CAUSAL_LM` since every other script in this
  project does generate text. See `reward-modeling/README.md`.
