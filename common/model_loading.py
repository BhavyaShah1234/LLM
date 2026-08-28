"""Model/tokenizer loading, shared across every stage.

Two distinct modes:
  - `build_model_from_scratch`: random-init a model from a config (no weights
    downloaded). This is what pretraining/ uses -- there's no pretrained
    checkpoint yet, we're creating the first one.
  - `load_causal_lm` / `load_vision_language_model` / `load_tokenizer`:
    `from_pretrained`-based loading of an existing checkpoint (local path or
    HF Hub id), optionally quantized. This is Wave-2+ plumbing (SFT, RLHF,
    ...) -- built now since it's identical boilerplate across every script
    that starts from a pretrained model.
"""

from typing import Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

_ARCHITECTURE_TO_AUTOMODEL = {
    "decoder-only": AutoModelForCausalLM,
    "encoder-only": AutoModelForMaskedLM,
    "encoder-decoder": AutoModelForSeq2SeqLM,
}


def build_model_from_scratch(
    architecture_family: str,
    config: PretrainedConfig,
    gradient_checkpointing: bool = True,
) -> PreTrainedModel:
    """Random-init a model matching `config`'s architecture. No pretrained
    weights are downloaded or loaded -- this is genuinely training from
    scratch, which is the entire point of the pretraining/ stage.
    """
    try:
        automodel_cls = _ARCHITECTURE_TO_AUTOMODEL[architecture_family]
    except KeyError:
        raise ValueError(
            f"Unknown architecture_family {architecture_family!r}; "
            f"expected one of {sorted(_ARCHITECTURE_TO_AUTOMODEL)}"
        )
    model = automodel_cls.from_config(config)
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
    return model


def load_tokenizer(model_name: str, use_fast: bool = True, ensure_pad_token: bool = True) -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=use_fast, trust_remote_code=True)
    if ensure_pad_token and tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_causal_lm(
    model_name: str,
    quantization_config: Optional[BitsAndBytesConfig] = None,
    torch_dtype: torch.dtype = torch.bfloat16,
    gradient_checkpointing: bool = True,
    trust_remote_code: bool = True,
) -> PreTrainedModel:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        dtype=torch_dtype,
        device_map="auto" if quantization_config is not None else None,
        trust_remote_code=trust_remote_code,
    )
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
    return model


def load_vision_language_model(
    model_name: str,
    quantization_config: Optional[BitsAndBytesConfig] = None,
    torch_dtype: torch.dtype = torch.bfloat16,
    gradient_checkpointing: bool = True,
    model_class=None,
) -> PreTrainedModel:
    if model_class is None:
        from transformers import Qwen2VLForConditionalGeneration

        model_class = Qwen2VLForConditionalGeneration
    model = model_class.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        dtype=torch_dtype,
        device_map="auto" if quantization_config is not None else None,
        trust_remote_code=True,
    )
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
    return model
