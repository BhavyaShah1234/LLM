"""Qualitative evaluation of the three pretraining/ checkpoints on novel prompts.

Loads whatever is currently saved in each of clm.py's, mlm.py's, and
span_corruption.py's --output_dir and runs it against hand-written prompts
that do not appear in TinyStories, to sanity-check "did this model actually
learn something" beyond the eval-loss/perplexity numbers written to each
run_result.json. Each architecture is tested the way it's actually able to
be tested:
  - clm.py (decoder-only): free-form greedy continuation.
  - span_corruption.py (encoder-decoder): sentinel-masked span infill, with
    the predicted span spliced back into the sentence for readability.
  - mlm.py (encoder-only): top-5 [MASK] fill-in predictions.

Usage:
    python test_trained_models.py
    python test_trained_models.py --clm_dir ./output/pretraining/clm --skip_mlm
"""

import argparse
import re

import torch

from common.model_loading import load_model_from_checkpoint, load_tokenizer

CLM_PROMPTS = [
    "The astronaut opened the spaceship door and",
    "Deep in the forest, a wise old owl",
    "On her birthday, Emma received a mysterious box that",
    "The robot looked at the broken toy and decided to",
    "It was raining hard, so the two friends decided to",
]

SPAN_CORRUPTION_INPUTS = [
    "The little boy grabbed his <extra_id_0> and ran outside to play.",
    "After the rain stopped, a bright <extra_id_0> appeared in the sky.",
    "She carefully wrapped the gift in <extra_id_0> before giving it to her friend.",
    "The old dog slowly walked to its <extra_id_0> and fell asleep.",
    "Every morning, the baker would <extra_id_0> fresh bread for the whole town.",
]

MLM_SENTENCES = [
    "The children were laughing and playing in the [MASK].",
    "He picked up the [MASK] and threw it across the yard.",
    "The [MASK] was so loud that everyone covered their ears.",
    "She wrapped herself in a warm [MASK] to stay cozy.",
    "The little kitten chased the [MASK] around the garden.",
]


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for this script.

    Returns:
        argparse.ArgumentParser: Parser covering each model's checkpoint
        directory, generation length, and per-architecture skip flags.
    """
    p = argparse.ArgumentParser(description="Qualitatively test the trained clm/mlm/span_corruption checkpoints on novel prompts.")
    p.add_argument("--clm_dir", type=str, default="./output/pretraining/clm", help="Path to the saved clm.py checkpoint. Default: ./output/pretraining/clm.")
    p.add_argument("--mlm_dir", type=str, default="./output/pretraining/mlm", help="Path to the saved mlm.py checkpoint. Default: ./output/pretraining/mlm.")
    p.add_argument("--span_dir", type=str, default="./output/pretraining/span_corruption", help="Path to the saved span_corruption.py checkpoint. Default: ./output/pretraining/span_corruption.")
    p.add_argument("--max_new_tokens", type=int, default=80, help="Max tokens to generate for clm/span_corruption. Default: 80.")
    p.add_argument("--skip_clm", action="store_true", default=False, help="Skip the decoder-only (clm.py) test.")
    p.add_argument("--skip_mlm", action="store_true", default=False, help="Skip the encoder-only (mlm.py) test.")
    p.add_argument("--skip_span", action="store_true", default=False, help="Skip the encoder-decoder (span_corruption.py) test.")
    return p


def test_clm(model_dir: str, max_new_tokens: int) -> None:
    """Load the decoder-only checkpoint and greedily continue each CLM_PROMPTS entry.

    Args:
        model_dir (str): Path to the saved clm.py checkpoint.
        max_new_tokens (int): Max number of tokens to generate per prompt.
    """
    print("=" * 80)
    print(f"DECODER-ONLY (clm.py) -- loading from {model_dir}")
    print("=" * 80)
    tokenizer = load_tokenizer(model_dir)
    model = load_model_from_checkpoint("decoder-only", model_dir, gradient_checkpointing=False)
    model.eval()
    for prompt in CLM_PROMPTS:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        print(f"\nPrompt: {prompt!r}\n-> {text!r}")
    print()


def test_span_corruption(model_dir: str, max_new_tokens: int) -> None:
    """Load the encoder-decoder checkpoint, infill each sentinel-masked span,
    and splice the prediction back into the original sentence for readability.

    Args:
        model_dir (str): Path to the saved span_corruption.py checkpoint.
        max_new_tokens (int): Max number of tokens to generate per input.
    """
    print("=" * 80)
    print(f"ENCODER-DECODER (span_corruption.py) -- loading from {model_dir}")
    print("=" * 80)
    tokenizer = load_tokenizer(model_dir)
    model = load_model_from_checkpoint("encoder-decoder", model_dir, gradient_checkpointing=False)
    model.eval()
    for text in SPAN_CORRUPTION_INPUTS:
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        raw_infill = tokenizer.decode(output_ids[0], skip_special_tokens=False)
        # The target format is "<pad><extra_id_0> ... text ... <extra_id_1>...";
        # take just the text between the first two sentinels as the filled span.
        match = re.search(r"<extra_id_0>\s*(.*?)\s*(?:<extra_id_1>|</s>|$)", raw_infill, re.DOTALL)
        filled_span = match.group(1).strip() if match else "(could not parse infill)"
        reconstructed = text.replace("<extra_id_0>", filled_span)
        print(f"\nInput: {text!r}")
        print(f"Raw model output: {raw_infill!r}")
        print(f"Reconstructed: {reconstructed!r}")
    print()


def test_mlm(model_dir: str) -> None:
    """Load the encoder-only checkpoint and predict the top-5 fill-ins for
    each [MASK] token in MLM_SENTENCES.

    Args:
        model_dir (str): Path to the saved mlm.py checkpoint.
    """
    print("=" * 80)
    print(f"ENCODER-ONLY (mlm.py) -- loading from {model_dir}")
    print("=" * 80)
    tokenizer = load_tokenizer(model_dir)
    model = load_model_from_checkpoint("encoder-only", model_dir, gradient_checkpointing=False)
    model.eval()
    for sentence in MLM_SENTENCES:
        inputs = tokenizer(sentence, return_tensors="pt").to(model.device)
        mask_positions = (inputs["input_ids"][0] == tokenizer.mask_token_id).nonzero(as_tuple=True)[0]
        if len(mask_positions) == 0:
            print(f"\nSentence: {sentence!r}\n-> [no MASK token found in tokenized input]")
            continue
        with torch.no_grad():
            logits = model(**inputs).logits
        mask_logits = logits[0, mask_positions[0]]
        top5_ids = torch.topk(mask_logits, 5).indices.tolist()
        top5_tokens = [tokenizer.decode([tok_id]).strip() for tok_id in top5_ids]
        print(f"\nSentence: {sentence!r}\n-> Top-5 predictions for [MASK]: {top5_tokens}")
    print()


def main():
    """Parse CLI args and run whichever architecture tests aren't skipped."""
    args = build_arg_parser().parse_args()
    if not args.skip_clm:
        test_clm(args.clm_dir, args.max_new_tokens)
    if not args.skip_span:
        test_span_corruption(args.span_dir, args.max_new_tokens)
    if not args.skip_mlm:
        test_mlm(args.mlm_dir)
    print("Done.")


if __name__ == "__main__":
    main()
