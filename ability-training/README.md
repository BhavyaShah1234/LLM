# Ability Training

Status: **CoT/reasoning-induction done (Wave 4)**.

Introduces abilities that a model doesn't reliably have off-the-shelf, via
a dedicated post-hoc training stage. Currently one ability is built out:

| Folder | Ability | Approach | Status |
|---|---|---|---|
| `CoT/` | Chain-of-thought reasoning | RLVR (GRPO with correctness + format reward), DeepSeek-R1-Zero-style -- no SFT rationale data | **done** |

See `CoT/README.md` for the reward design and real findings from a
training run (correctness reward improves; `<think>` usage did not emerge
in a 16-step toy run -- documented as an expected/explained negative
result, not a bug).

An SFT-based alternative (distilling a teacher's rationales instead of
inducing reasoning via RL) already exists as the `*_cot.py` variants under
`supervised-finetuning/` (e.g. `text/classification/decoder-only/text_classification_cot.py`,
`text/mcq/decoder-only/mcq_cot.py`) -- those take CoT as given training
data rather than inducing it, so they're filed under `supervised-finetuning/`
rather than here. Comparing the two approaches head-to-head is a natural
future `experiments/` entry.
