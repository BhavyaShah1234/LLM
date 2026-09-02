# Domain Adaptation: LoRA-Weighted vs Base-Weight-Updated

Status: **planned** (not yet run).

Question: given the *same amount of data* for both the domain-adaptation
stage and the downstream finetuning stage, does a domain-adapted model
whose **base weights** were updated (`domain_adaptation.py --full_finetune`)
outperform one where only a **LoRA adapter** was trained
(`domain_adaptation.py`'s default) -- measured by downstream finetuning
performance on a task in the same domain?

This follows directly from `domain-adaptation/domain_adaptation.py`'s
`--full_finetune` flag: canonical Domain-Adaptive Pretraining (DAPT,
Gururangan et al. 2020, "Don't Stop Pretraining") updates the base model's
own weights; LoRA-based domain adaptation is a later, resource-efficient
variant of the same idea, not the original approach. Whether the
resource-efficient variant actually costs anything in downstream quality is
an open, empirically-answerable question -- this experiment answers it for
this project's model/hardware/data budget.

## Setup

Four runs total, in two stages. Data budget held fixed within each stage
across both arms.

**Stage 1 -- domain-adaptation** (`domain-adaptation/domain_adaptation.py`,
`araag2/MedMCQA` "processed" config, `Explanation` field), identical
`--max_samples`/`--epochs` on both arms:

```bash
# LoRA arm -- --save_strategy merged so the output is a plain loadable
# checkpoint, directly comparable to the full-finetune arm's output (the
# default adapter_only strategy needs the base model present separately).
python domain_adaptation.py --max_samples 50000 --epochs 3 --save_strategy merged --output_dir ./output/domain-adaptation/lora

# Full-finetune arm -- same data budget. NOT confirmed to fit in 8GB VRAM
# (see the flag's own help text / supervised-finetuning/README.md's OOM
# finding for this model size); --optimizer paged_adamw_8bit and a small
# --batch_size are the mitigations available in this script. If it still
# doesn't fit on this hardware, this arm needs to run wherever it does
# (more VRAM, or with further offloading this project doesn't implement,
# e.g. DeepSpeed ZeRO), or with a reduced --max_samples matched on both arms.
python domain_adaptation.py --full_finetune --optimizer paged_adamw_8bit --batch_size 1 --max_samples 50000 --epochs 3 --output_dir ./output/domain-adaptation/full-finetune
```

**Stage 2 -- downstream finetuning** on a task in the same domain
(`supervised-finetuning/text/mcq/decoder-only/mcq_standard.py`, MedMCQA's
question/options/answer structure -- the same dataset family, different
field usage, so this is a genuine downstream task rather than more of the
same objective), `--model` pointed at each Stage 1 output, identical
`--max_samples`/`--epochs` on both arms:

```bash
python mcq_standard.py --model ../../../output/domain-adaptation/lora --max_samples <N> --epochs <E> --lora --output_dir ./output/supervised-finetuning/mcq-from-domain-adapted-lora
python mcq_standard.py --model ../../../output/domain-adaptation/full-finetune --max_samples <N> --epochs <E> --lora --output_dir ./output/supervised-finetuning/mcq-from-domain-adapted-full
```

(`--lora` on the Stage 2 finetuning calls is about *that* stage's own
compute budget -- keep it identical across both arms, it is not what this
experiment is testing.)

## What this isolates -- and what it doesn't

This isolates "does the domain-adaptation weight-update strategy matter for
downstream performance," holding the downstream finetuning method and data
budget fixed across arms. It does **not** control for the full-finetune
arm needing a different `--batch_size`/`--optimizer` to fit in VRAM (a real
optimization confound noted above) -- if that arm needs a meaningfully
different effective batch size to run at all, note that alongside the
result rather than reading a difference in the final comparison as purely
about "LoRA vs full weights."

## Result

Not yet run -- see Setup above to reproduce. Once both Stage 2 runs
complete, `compare.py` reads their `run_result.json` files and prints the
comparison.
