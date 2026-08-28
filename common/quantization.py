"""Training-time quantization config (QLoRA-style loading of a pretrained base).

Not used by pretraining/ scripts (there's no pretrained base to quantize --
they train full parameters from a random init). This is Wave-2+ plumbing,
built now because it's identical, boilerplate logic across every script that
will load a pretrained checkpoint.
"""

from typing import Optional

import torch
from transformers import BitsAndBytesConfig


def _torch_dtype(mixed_precision: str) -> torch.dtype:
    if mixed_precision == "bf16":
        return torch.bfloat16
    if mixed_precision == "fp16":
        return torch.float16
    return torch.float32


def build_quantization_config(
    quantization: str,
    mixed_precision: str = "bf16",
) -> Optional[BitsAndBytesConfig]:
    """quantization: 'no' | '4bit' | '8bit'."""
    compute_dtype = _torch_dtype(mixed_precision)
    if quantization == "4bit":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )
    if quantization == "8bit":
        return BitsAndBytesConfig(load_in_8bit=True)
    if quantization == "no":
        return None
    raise ValueError(f"Unknown quantization mode: {quantization!r} (expected 'no', '4bit', or '8bit')")
