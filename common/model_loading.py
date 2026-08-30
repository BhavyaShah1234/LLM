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

    Args:
        architecture_family (str): One of "decoder-only", "encoder-only", or
            "encoder-decoder"; selects which AutoModel class to instantiate.
        config (PretrainedConfig): Model config to build the random-init
            model from.
        gradient_checkpointing (bool): Whether to enable gradient
            checkpointing on the resulting model.

    Returns:
        PreTrainedModel: A freshly initialized (untrained) model.

    Raises:
        ValueError: If `architecture_family` isn't a recognized key.
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
    """Load a tokenizer from a local path or HF Hub id.

    Args:
        model_name (str): Local path or HF Hub model id to load the
            tokenizer from.
        use_fast (bool): Whether to prefer the fast (Rust-backed) tokenizer.
        ensure_pad_token (bool): If True and the tokenizer has no pad token,
            set it to the EOS token (needed by many decoder-only models).

    Returns:
        PreTrainedTokenizerBase: The loaded tokenizer.
    """
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
    """Load a pretrained causal LM, optionally quantized.

    Args:
        model_name (str): Local path or HF Hub model id to load.
        quantization_config (Optional[BitsAndBytesConfig]): bitsandbytes
            config for 4-bit/8-bit loading; None loads in full precision.
        torch_dtype (torch.dtype): Compute dtype for unquantized weights.
        gradient_checkpointing (bool): Whether to enable gradient
            checkpointing on the loaded model.
        trust_remote_code (bool): Passed through to `from_pretrained`; allows
            model repos with custom modeling code.

    Returns:
        PreTrainedModel: The loaded causal LM.
    """
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
    """Load a pretrained vision-language model, optionally quantized.

    Args:
        model_name (str): Local path or HF Hub model id to load.
        quantization_config (Optional[BitsAndBytesConfig]): bitsandbytes
            config for 4-bit/8-bit loading; None loads in full precision.
        torch_dtype (torch.dtype): Compute dtype for unquantized weights.
        gradient_checkpointing (bool): Whether to enable gradient
            checkpointing on the loaded model.
        model_class: AutoModel-style class to instantiate; defaults to
            `Qwen2VLForConditionalGeneration` when not given.

    Returns:
        PreTrainedModel: The loaded vision-language model.
    """
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
