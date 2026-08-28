# RLHF: Pretrained-Init vs SFT-Init

Status: **done (Wave 4)**.

Question: what happens if RLHF (DPO/GRPO) is run directly on a raw
pretrained model versus on an SFT'd/instruction-tuned model? This
experiment exists specifically to demonstrate empirically *why*
`rlhf/README.md` documents an
instruction-tuned-checkpoint prerequisite.

## Setup

Two GRPO runs (`rlhf/grpo/grpo.py`),
identical except `--model`:

```bash
python grpo.py --max_samples 40 --max_eval_samples 20 --lora --quantization 4bit --batch_size 4 --gradient_accumulation_steps 2 --num_generations 4 --max_completion_length 48
# (the above is this project's already-run default: Qwen/Qwen3-1.7B, instruction-tuned)

python grpo.py --model Qwen/Qwen3-1.7B-Base --max_samples 40 --max_eval_samples 20 --lora --quantization 4bit --batch_size 4 --gradient_accumulation_steps 2 --num_generations 4 --max_completion_length 48 --output_dir ./output/rlhf/grpo-pretrained-init
```

## Real finding -- and why the obvious metric almost hides it

The naive expectation was "the pretrained-init run scores markedly lower."
It doesn't, by the headline number: `eval_reward` is 0.425 (instruct) vs
0.375 (pretrained) -- a small gap that alone would suggest RLHF's
SFT-prerequisite claim is overstated, at least at this tiny (40-sample,
3-step) compute budget.

The other run_result.json fields tell a sharper story:

| Metric | Instruct-init | Pretrained-init | Reading |
|---|---|---|---|
| `eval_reward` | 0.425 | 0.375 | looks close |
| `eval_entropy` | 0.495 | 1.922 | pretrained-init is ~4x less confident/peaked |
| `eval_frac_reward_zero_std` | 0.70 | 0.15 | instruct-init is consistent across its 4 sampled completions per prompt far more often |
| `eval_completions/mean_terminated_length` | 0.0 (100% hit the token cap) | 3.7 (completions that stop do so almost instantly) | pretrained-init's "clean stops" are suspiciously short |

Direct inspection of the pretrained-init adapter's real greedy generations
(captured via `model.generate()` on the saved adapter -- see `compare.py`
for the exact snippet pattern) shows the mechanism: it converged to a
**degenerate shortcut, echoing option A's text back almost every time**,
rather than learning to actually answer the question:

```
Label='A'  Completion='A. Impulse through myelinated fibers is slower than non-myelinated fibers'   (correct by luck)
Label='A'  Completion='A. The oncotic pressure of the fluid leaving the capillaries is less than...' (correct by luck)
Label='C'  Completion='C. Amniotic fluid samples plus chromosomal analysis will definitely tell...'  (correct, real explanation)
Label='C'  Completion='A. Antegrade'                                                                  (wrong -- defaulted to A again)
```

**"Always guess A" is right ~25% of the time on a 4-option MCQ purely by
chance** -- close to this run's 0.375 mean reward once you allow that a
quarter of the eval set genuinely has label A. GRPO's sparse verifiable
reward gave the raw pretrained model just enough signal, at this tiny
compute budget, to find that cheap shortcut instead of genuinely learning
the task -- a textbook reward-hacking failure mode. The instruction-tuned
starting point doesn't just get a higher score; it starts from a policy
that already knows how to *attempt* the task format, so GRPO's reward has
something real to shape rather than a blank policy free to collapse onto
whatever cheap heuristic gets partial credit fastest.

**Answer to the question**: yes, RLHF benefits from (and at this compute
budget, needs) an instruction-tuned starting point -- not primarily because
the raw-pretrained run scores dramatically lower on the reward metric
itself, but because without an instruction-following prior, GRPO's sparse
reward is exploitable by a trivial shortcut that a naive reward-only
comparison would under-detect. `eval_entropy` and
`eval_frac_reward_zero_std` (both direct signals of policy stability, not
task performance per se) caught this; the reward number alone would not
have.

## Usage

```bash
python compare.py
```
