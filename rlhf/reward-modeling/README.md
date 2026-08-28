# Reward Modeling

Status: **done (Wave 5)**.

Trains a scalar reward model via a Bradley-Terry pairwise loss on
(chosen, rejected) pairs -- the component classic reward-model-then-PPO
RLHF needs, and the thing DPO/KTO are specifically designed to let you
skip. Not consumed by another script in this project yet (no PPO trainer
built), but the trained model's scores are a general-purpose reward
signal -- see the module docstring for how this could feed a future GRPO
run as a learned alternative to `grpo/grpo.py`'s hand-written rule-based
reward.

**No RLHF prerequisite here**, unlike `dpo.py`/`kto.py`/`grpo.py`: this
trains a classifier, not the policy, so `--model` defaults to this
project's usual base-model default (`Qwen/Qwen3-1.7B-Base`), not an
instruction-tuned checkpoint.

## Dataset

Reuses **`trl-lib/ultrafeedback_binarized`** (same dataset `dpo.py` uses,
deliberately -- reward modeling needs the identical (chosen, rejected)
pair shape, and `RewardTrainer` consumes the same conversational format
`DPOTrainer` does with no reformatting, confirmed via source inspection).

## Real bug found and fixed: wrong PEFT task_type for a classification head

`trl`'s `RewardTrainer` swaps the given `--model` string for
`AutoModelForSequenceClassification` internally (`num_labels=1`, confirmed
via source inspection) -- a fundamentally different model than the causal
LM `common/peft_setup.py`'s `build_lora_config()` defaults to
(`task_type="CAUSAL_LM"`). Using that default here crashed with:

```
AttributeError: 'Qwen3ForSequenceClassification' object has no attribute 'prepare_inputs_for_generation'
```

`peft`'s `get_peft_model()` assumes `task_type="CAUSAL_LM"` models are
generation-capable and tries to wire up
`prepare_inputs_for_generation`, which a sequence-classification model
doesn't have. Fixed by passing `task_type="SEQ_CLS"` explicitly in this
script's `build_lora_config()` call -- `common/peft_setup.py` itself
wasn't changed, since `CAUSAL_LM` is the correct default for every other
script in this project; this is the one script attaching a classification
head instead of generating text.

## Verified working run

```bash
python reward_modeling.py --lora --quantization 4bit --batch_size 4 --gradient_accumulation_steps 4 --max_length 256 --max_samples 60 --max_eval_samples 20
```

`eval_accuracy: 1.0`, `eval_margin: 0.496` (chosen scored meaningfully
higher than rejected on every eval pair). Treat `1.0` as a small-sample
result (20 eval pairs after a handful of optimizer steps), not a claim of
a well-calibrated reward model -- it confirms the training mechanism works
correctly (real loss, real reward spread between min/mean/max), not that
this reward model is production-ready.

## Usage

```bash
python reward_modeling.py --debug_first_batch --max_samples 20
python reward_modeling.py --lora --quantization 4bit --max_samples 2000
```

## Dataset Reference (this folder's slice of the root README's master table)

| Script | Dataset | Split(s) | Role |
|---|---|---|---|
| `reward_modeling.py` | `trl-lib/ultrafeedback_binarized` | `train` (training), `test` (eval) | same dataset as `dpo/dpo.py` -- identical (chosen, rejected) shape |
