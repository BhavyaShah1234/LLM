"""Minimal distributed/batch-size helpers.

Deliberately thin: real DeepSpeed ZeRO / FSDP config plumbing belongs to the
optimization/ stage (a future wave), not here. This module only holds the
couple of values every script's TrainingArguments construction needs.
"""

from typing import Optional


def get_effective_batch_size(per_device_batch_size: int, grad_accum_steps: int, world_size: int = 1) -> int:
    """Compute the true global batch size across accumulation and devices.

    Args:
        per_device_batch_size (int): Batch size processed per device per step.
        grad_accum_steps (int): Number of gradient accumulation steps before
            an optimizer step.
        world_size (int): Number of distributed processes/devices training in
            parallel.

    Returns:
        int: The effective (global) batch size seen by one optimizer step.
    """
    return per_device_batch_size * grad_accum_steps * world_size


def ddp_find_unused_parameters(use_lora: bool) -> Optional[bool]:
    """Decide the `ddp_find_unused_parameters` TrainingArguments value.

    LoRA freezes most of the base model's parameters, which confuses DDP's
    unused-parameter detection unless this is explicitly set.

    Args:
        use_lora (bool): Whether the run uses a LoRA-adapted model.

    Returns:
        Optional[bool]: True if LoRA is in use (DDP must scan for unused
            params); None otherwise, letting DDP use its own default.
    """
    # LoRA freezes most of the base model's parameters, which confuses DDP's
    # unused-parameter detection unless this is explicitly set.
    return True if use_lora else None
