"""Configurable model-saving strategies, shared by every training script.

Strategies:
  - "full":              save the complete model as-is (no LoRA involved).
                          This is the *only* mode pretraining/ scripts use --
                          there's no adapter/base distinction when you just
                          trained every parameter from scratch.
  - "adapter_only":       (Wave 2+, LoRA runs) save just the adapter weights;
                          smallest on disk, needs the base model present at
                          inference time (PeftModel.from_pretrained(base, dir)).
  - "merged":             (Wave 2+, LoRA runs) merge the adapter into the base
                          and save one standalone model directory -- larger on
                          disk, loads with plain from_pretrained(), no PEFT
                          dependency at inference/serving time.
  - "adapter_and_merged": save both.
  - "base_reference":     (Wave 2+, LoRA runs) write a small pointer file
                          recording the exact base checkpoint id/revision used,
                          instead of duplicating multi-GB base weights, so an
                          adapter_only save stays fully reproducible.
"""

import json
import os
import subprocess
from typing import Optional

_VALID_STRATEGIES = {"full", "adapter_only", "merged", "adapter_and_merged", "base_reference"}


def _dir_size_human(path: str) -> str:
    """Get a directory's on-disk size as a human-readable string (via `du -sh`).

    Args:
        path (str): Directory to measure.

    Returns:
        str: Human-readable size (e.g. "1.2G"), or "unknown" if `du` fails.
    """
    try:
        out = subprocess.run(["du", "-sh", path], capture_output=True, text=True, check=True)
        return out.stdout.split()[0]
    except Exception:
        return "unknown"


def save_model(
    model,
    tokenizer,
    output_dir: str,
    strategy: str = "full",
    base_model_name: Optional[str] = None,
    is_lora: bool = False,
) -> None:
    """Save a trained model according to one of the module's save strategies.

    See the module docstring for what each strategy value does.

    Args:
        model: The trained model (or PeftModel, if `is_lora`) to save.
        tokenizer: Tokenizer to save alongside the model.
        output_dir (str): Directory to write the saved model(s) into.
        strategy (str): One of "full", "adapter_only", "merged",
            "adapter_and_merged", or "base_reference".
        base_model_name (Optional[str]): Base checkpoint id/path, required
            for the "base_reference" strategy and any LoRA strategy that
            writes a base-reference pointer file.
        is_lora (bool): Whether `model` is a LoRA-adapted PeftModel.

    Raises:
        ValueError: If `strategy` is not a recognized strategy, if
            `strategy='full'` is used with `is_lora=True`, or if a
            LoRA-only strategy is used with `is_lora=False`.
    """
    if strategy not in _VALID_STRATEGIES:
        raise ValueError(f"Unknown save strategy {strategy!r}; expected one of {sorted(_VALID_STRATEGIES)}")

    if strategy == "full":
        if is_lora:
            raise ValueError("save_strategy='full' is only valid for non-LoRA (full-parameter) runs")
        _save_full(model, tokenizer, output_dir)
        return

    if not is_lora:
        raise ValueError(f"save_strategy={strategy!r} requires a LoRA-adapted model (is_lora=True)")

    if strategy == "base_reference":
        _save_base_reference(output_dir, base_model_name)
        return

    if strategy in ("adapter_only", "adapter_and_merged"):
        adapter_dir = os.path.join(output_dir, "adapter") if strategy == "adapter_and_merged" else output_dir
        _save_adapter_only(model, tokenizer, adapter_dir)
        _save_base_reference(adapter_dir, base_model_name)

    if strategy in ("merged", "adapter_and_merged"):
        merged_dir = os.path.join(output_dir, "merged") if strategy == "adapter_and_merged" else output_dir
        _save_merged(model, tokenizer, merged_dir)


def _save_full(model, tokenizer, output_dir: str) -> None:
    """Save the complete model and tokenizer as-is.

    Args:
        model: Model to save.
        tokenizer: Tokenizer to save.
        output_dir (str): Directory to save into (created if missing).
    """
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"[model_saving] saved full model weights to {output_dir} ({_dir_size_human(output_dir)} on disk)")


def _save_adapter_only(model, tokenizer, output_dir: str) -> None:
    """Save just the LoRA adapter weights (not the base model).

    Args:
        model: PeftModel whose adapter weights should be saved.
        tokenizer: Tokenizer to save.
        output_dir (str): Directory to save into (created if missing).
    """
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"[model_saving] saved LoRA adapter to {output_dir} ({_dir_size_human(output_dir)} on disk)")


def _save_merged(model, tokenizer, output_dir: str) -> None:
    """Merge the LoRA adapter into the base model and save the result standalone.

    Args:
        model: PeftModel to merge and save.
        tokenizer: Tokenizer to save.
        output_dir (str): Directory to save into (created if missing).
    """
    os.makedirs(output_dir, exist_ok=True)
    merged = model.merge_and_unload()
    merged.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"[model_saving] saved merged (adapter+base) model to {output_dir} ({_dir_size_human(output_dir)} on disk)")


def _save_base_reference(output_dir: str, base_model_name: Optional[str]) -> None:
    """Write a small JSON pointer file recording the base checkpoint used.

    Args:
        output_dir (str): Directory to write `base_model.json` into (created
            if missing).
        base_model_name (Optional[str]): Base checkpoint id/path to record.
    """
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "base_model.json"), "w") as f:
        json.dump({"base_model_name": base_model_name}, f, indent=2)
