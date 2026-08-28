"""Minimal distributed/batch-size helpers.

Deliberately thin: real DeepSpeed ZeRO / FSDP config plumbing belongs to the
optimization/ stage (a future wave), not here. This module only holds the
couple of values every script's TrainingArguments construction needs.
"""

from typing import Optional


def get_effective_batch_size(per_device_batch_size: int, grad_accum_steps: int, world_size: int = 1) -> int:
    return per_device_batch_size * grad_accum_steps * world_size


def ddp_find_unused_parameters(use_lora: bool) -> Optional[bool]:
    # LoRA freezes most of the base model's parameters, which confuses DDP's
    # unused-parameter detection unless this is explicitly set.
    return True if use_lora else None
