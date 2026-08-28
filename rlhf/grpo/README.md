# GRPO

Status: **done (Wave 4)**.

Group Relative Policy Optimization with a **verifiable, rule-based reward**
(RLVR: RL with Verifiable Rewards) -- no learned reward model needed. See
the parent folder's README for the RLHF prerequisite and real memory/API
findings, and root README's "RLHF algorithm family" for the theory. This
script's reward function and dataset framing are shared almost verbatim with
`ability-training/CoT/`, which reuses this exact approach to induce
reasoning rather than align preferences -- see that folder's README for how
the two relate.

## Dataset

**`araag2/MedMCQA`** (config `processed`) -- the same dataset
`supervised-finetuning/text/mcq/decoder-only/mcq_standard.py` uses. Its
`Label` field (already a letter A-D) is exactly the ground truth a
verifiable reward function needs: `mcq_correctness_reward()` extracts the
first standalone A/B/C/D letter from a completion and compares it to
`Label`, returning `1.0`/`0.0`.

**Real bug found**: the `test` split's `Label` field is `None` for every
row (labels withheld). Evaluating against it silently produces exactly 0.0
reward regardless of model quality -- confirmed via direct instrumentation
of the reward function during a live training step (nonzero on `train`,
exactly zero on `test`). Fixed by evaluating against `dev` instead; the same
bug was found (and fixed) in `mcq_standard.py`, which had the identical
`split="test"` choice but had never been trained for real to surface it.

## Usage

```bash
python grpo.py --debug_first_batch --max_samples 8
python grpo.py --lora --quantization 4bit --max_samples 500   # needed to fit 8GB VRAM, see parent README
```

If `completions/clipped_ratio` stays at `1.0` (every completion hits
`--max_completion_length` without producing a parseable letter), increase
`--max_completion_length` -- this model's natural response style
("The correct answer is X. ...") needs more headroom than a bare letter.

## Dataset Reference (this folder's slice of the root README's master table)

| Script | Dataset | Split(s) | Role |
|---|---|---|---|
| `grpo.py` | `araag2/MedMCQA` (config `processed`) | `train` (training), `dev` (eval) | primary + only corpus |
