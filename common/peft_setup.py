"""LoRA/PEFT config construction. Wave-2+ plumbing (nothing pretrained to adapt
during from-scratch pretraining), built now since it's identical boilerplate
across every future SFT/RLHF script.
"""

from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training


def build_lora_config(
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: str = "q_proj,k_proj,v_proj,o_proj",
    task_type: str = "CAUSAL_LM",
) -> LoraConfig:
    """Build a `peft.LoraConfig` from the project's standard LoRA CLI flags.

    Args:
        lora_r (int): LoRA rank.
        lora_alpha (int): LoRA scaling factor.
        lora_dropout (float): Dropout probability applied to LoRA layers.
        target_modules (str): Comma-separated module names to attach LoRA
            adapters to (e.g. "q_proj,k_proj,v_proj,o_proj").
        task_type (str): PEFT task type string (e.g. "CAUSAL_LM").

    Returns:
        LoraConfig: The constructed LoRA configuration.
    """
    return LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules.split(","),
        task_type=task_type,
        bias="none",
    )


def apply_lora(model, lora_config: LoraConfig, prepare_for_kbit: bool = True, print_trainable: bool = True) -> PeftModel:
    """Wrap a base model with LoRA adapters.

    Args:
        model: Base model to adapt.
        lora_config (LoraConfig): LoRA configuration to apply.
        prepare_for_kbit (bool): Whether to run
            `prepare_model_for_kbit_training` first (needed when `model` was
            loaded quantized).
        print_trainable (bool): Whether to print the trainable-parameter
            count/percentage after wrapping.

    Returns:
        PeftModel: The LoRA-adapted model.
    """
    if prepare_for_kbit:
        model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    if print_trainable:
        model.print_trainable_parameters()
    return model
