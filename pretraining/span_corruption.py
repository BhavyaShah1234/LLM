"""Span-corruption pretraining -- encoder-decoder architecture (T5-style).

Trains a small, T5-style transformer *from scratch* (random initialization,
no pretrained weights). The objective: randomly mask contiguous spans of the
input with sentinel tokens (<extra_id_0>, <extra_id_1>, ...); the encoder
sees the corrupted text, the decoder must reconstruct exactly the dropped
spans (each preceded by its sentinel). This is the pretraining objective
T5/BART-family encoder-decoder models use, and structurally different from
both CLM (causal, decoder-only) and MLM (bidirectional in-place prediction,
encoder-only) -- here the encoder is bidirectional but the decoder is causal
and cross-attends to the encoder output.

Toy-scale by design -- see pretraining/README.md.

Usage:
    python span_corruption.py --debug_first_batch --max_samples 20
    python span_corruption.py --max_steps 500 --output_dir ./output/pretraining/span_corruption
"""

import argparse
import math
import time

import numpy as np
from datasets import load_dataset
from transformers import DataCollatorForSeq2Seq, T5Config, Trainer, TrainingArguments

from common.data_selection import select_samples
from common.logging_utils import print_banner, print_config, print_formatted_examples
from common.model_loading import build_model_from_scratch, load_tokenizer
from common.model_saving import save_model
from common.run_results import write_run_result
from common.seeding import set_all_seeds

DATASET_NAME = "roneneldan/TinyStories"
TOKENIZER_NAME = "t5-base"
ARCHITECTURE = "encoder-decoder"


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for this script.

    Returns:
        argparse.ArgumentParser: Parser covering architecture sizing,
        span-corruption noise parameters, data, training hyperparameters,
        system, and debug options.
    """
    p = argparse.ArgumentParser(description="Pretrain a small encoder-decoder model from scratch (span-corruption objective).")

    p.add_argument("--hidden_size", type=int, default=512, help="Transformer hidden size (d_model). Default: 512.")
    p.add_argument("--num_encoder_layers", type=int, default=4, help="Encoder depth. Default: 4.")
    p.add_argument("--num_decoder_layers", type=int, default=4, help="Decoder depth. Default: 4 (total depth 8, comparable to clm.py/mlm.py's 8 layers).")
    p.add_argument("--num_attention_heads", type=int, default=8, help="Number of attention heads. Default: 8.")
    p.add_argument("--block_size", type=int, default=256, help="Input sequence length in tokens, before span corruption shrinks it. Default: 256.")
    p.add_argument("--noise_density", type=float, default=0.15, help="Fraction of input tokens corrupted into spans. Default: 0.15 (the T5 paper default).")
    p.add_argument("--mean_noise_span_length", type=float, default=3.0, help="Average corrupted span length in tokens. Default: 3.0 (the T5 paper default).")

    p.add_argument("--max_samples", type=int, default=-1, help="Number of raw dataset rows (stories) to use before tokenization/packing. -1 (default) = use the full split.")
    p.add_argument("--sample_selection", type=str, default="random", choices=["random", "first", "last"], help="Which rows to pick when --max_samples is positive. Default: random.")
    p.add_argument("--max_eval_samples", type=int, default=2000, help="Number of validation rows to use for evaluation. Default: 2000.")

    p.add_argument("--batch_size", type=int, default=16, help="Per-device training batch size. Default: 16.")
    p.add_argument("--gradient_accumulation_steps", type=int, default=2, help="Gradient accumulation steps. Default: 2.")
    p.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate. Default: 3e-4.")
    p.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay. Default: 0.01.")
    p.add_argument("--warmup_steps", type=int, default=200, help="LR warmup steps. Default: 200.")
    p.add_argument("--epochs", type=float, default=1.0, help="Training epochs over the (possibly truncated) train split. Ignored if --max_steps is set. Default: 1.0.")
    p.add_argument("--max_steps", type=int, default=-1, help="If set (>0), overrides --epochs. Default: -1 (unset).")

    p.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"], help="Mixed precision mode. Default: bf16.")
    p.add_argument("--gradient_checkpointing", action="store_true", default=False, help="Enable gradient checkpointing. Default: off.")
    p.add_argument("--seed", type=int, default=42, help="Random seed. Default: 42.")
    p.add_argument("--output_dir", type=str, default="./output/pretraining/span_corruption", help="Where to save the trained model, tokenizer, and run_result.json.")
    p.add_argument("--logging_steps", type=int, default=25, help="Logging frequency. Default: 25.")
    p.add_argument("--eval_steps", type=int, default=250, help="Evaluation frequency. Default: 250.")
    p.add_argument("--save_steps", type=int, default=500, help="Checkpoint save frequency. Default: 500.")

    p.add_argument("--debug_first_batch", action="store_true", default=False, help="Load data, build the model, print formatted examples and the model's real parameter count, then exit without training.")

    return p


def verify_dataset() -> None:
    """Peek at the training split via streaming and assert a "text" field exists.

    Raises:
        AssertionError: If the first streamed example has no "text" field.
    """
    print_banner("VERIFYING DATASET")
    peek = load_dataset(DATASET_NAME, split="train", streaming=True)
    example = next(iter(peek))
    assert "text" in example, f"Expected a 'text' field in {DATASET_NAME}, got fields: {list(example.keys())}"
    print(f"Dataset: {DATASET_NAME}")
    print(f"Fields: {list(example.keys())}")
    print(f"Sample text: {example['text'][:200]!r}")
    print()


def tokenize_and_pack(dataset, tokenizer, block_size: int, desc: str):
    """Tokenize raw story text and pack it into fixed-length blocks (pre-corruption).

    Args:
        dataset (datasets.Dataset): Raw dataset with a "text" column.
        tokenizer: Tokenizer to encode text with.
        block_size (int): Number of tokens per packed block, before span
            corruption shrinks the encoder input.
        desc (str): Short label used in progress-bar descriptions (e.g.
            "train" or "eval").

    Returns:
        datasets.Dataset: Dataset of fixed-length "input_ids" blocks, with
        any trailing tokens shorter than block_size dropped.
    """
    def tokenize_fn(examples):
        """Tokenize a batch of stories without adding special tokens."""
        return tokenizer(examples["text"], add_special_tokens=False)

    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=dataset.column_names, desc=f"Tokenizing ({desc})")

    def group_texts(examples):
        """Concatenate a tokenized batch and split it into block_size chunks."""
        concatenated = sum(examples["input_ids"], [])
        total_length = (len(concatenated) // block_size) * block_size
        return {"input_ids": [concatenated[i : i + block_size] for i in range(0, total_length, block_size)]}

    return tokenized.map(group_texts, batched=True, remove_columns=tokenized.column_names, desc=f"Packing into {block_size}-token blocks ({desc})")


def random_spans_noise_mask(length: int, noise_density: float, mean_noise_span_length: float, rng: np.random.Generator) -> np.ndarray:
    """Build a boolean noise mask arranged as alternating non-noise/noise spans.

    Boolean mask over `length` positions marking which tokens are corrupted,
    arranged as alternating non-noise/noise spans (T5 paper, Section 3.1.4).

    Args:
        length (int): Number of positions to generate a mask for.
        noise_density (float): Target fraction of positions marked noisy.
        mean_noise_span_length (float): Target average length of each noisy
            span.
        rng (numpy.random.Generator): Random generator used to shuffle span
            boundaries.

    Returns:
        numpy.ndarray: Boolean array of length `length`; True marks a
        corrupted (noise) position.
    """
    num_noise_tokens = int(round(length * noise_density))
    num_noise_tokens = min(max(num_noise_tokens, 1), length - 1)
    num_noise_spans = max(int(round(num_noise_tokens / mean_noise_span_length)), 1)
    num_nonnoise_tokens = length - num_noise_tokens

    def random_segmentation(num_items: int, num_segments: int) -> np.ndarray:
        """Split num_items into num_segments randomly-sized, non-empty, ordered pieces.

        Args:
            num_items (int): Total number of items to distribute.
            num_segments (int): Number of segments to split into.

        Returns:
            numpy.ndarray: Length-`num_segments` array of segment lengths
            summing to `num_items`.
        """
        mask_indices = np.arange(num_items - 1) < (num_segments - 1)
        rng.shuffle(mask_indices)
        first_in_segment = np.pad(mask_indices, [[1, 0]], constant_values=0)
        segment_id = np.cumsum(first_in_segment)
        segment_length = np.zeros(num_segments, dtype=np.int32)
        np.add.at(segment_length, segment_id, 1)
        return segment_length

    noise_span_lengths = random_segmentation(num_noise_tokens, num_noise_spans)
    nonnoise_span_lengths = random_segmentation(num_nonnoise_tokens, num_noise_spans)
    interleaved = np.reshape(np.stack([nonnoise_span_lengths, noise_span_lengths], axis=1), [num_noise_spans * 2])
    span_starts = np.cumsum(interleaved)[:-1]
    span_start_indicator = np.zeros(length, dtype=np.int32)
    span_start_indicator[span_starts] = 1
    span_num = np.cumsum(span_start_indicator)
    return (span_num % 2) == 1


def apply_span_corruption(token_ids, tokenizer, noise_density: float, mean_noise_span_length: float, rng: np.random.Generator):
    """Replace noisy spans with sentinel tokens and build the reconstruction target.

    Each contiguous noisy span in `token_ids` is collapsed to a single
    `<extra_id_N>` sentinel in the encoder input; the decoder target
    interleaves the same sentinels with the dropped tokens they replaced.

    Args:
        token_ids (list[int]): Token ids for one packed block.
        tokenizer: Tokenizer used to look up sentinel token ids and the EOS
            token id.
        noise_density (float): Target fraction of positions marked noisy.
        mean_noise_span_length (float): Target average length of each noisy
            span.
        rng (numpy.random.Generator): Random generator used to place spans.

    Returns:
        tuple[list[int], list[int]]: Encoder input ids (corrupted, sentinel-
        substituted, EOS-terminated) and decoder target ids (sentinels plus
        the dropped tokens, EOS-terminated).
    """
    noise_mask = random_spans_noise_mask(len(token_ids), noise_density, mean_noise_span_length, rng)
    encoder_input_ids, target_ids = [], []
    sentinel_idx = 0
    prev_noise = False
    for tok, is_noise in zip(token_ids, noise_mask):
        if is_noise:
            if not prev_noise:
                sentinel_id = tokenizer.convert_tokens_to_ids(f"<extra_id_{sentinel_idx}>")
                encoder_input_ids.append(sentinel_id)
                target_ids.append(sentinel_id)
                sentinel_idx += 1
            target_ids.append(tok)
        else:
            encoder_input_ids.append(tok)
        prev_noise = is_noise
    encoder_input_ids.append(tokenizer.eos_token_id)
    target_ids.append(tokenizer.eos_token_id)
    return encoder_input_ids, target_ids


def load_and_prepare_data(args, tokenizer):
    """Load TinyStories, pack into blocks, and apply span corruption to each.

    Args:
        args (argparse.Namespace): Parsed CLI args; uses max_samples,
            sample_selection, max_eval_samples, seed, block_size,
            noise_density, and mean_noise_span_length.
        tokenizer: Tokenizer to encode text and look up sentinel/EOS ids.

    Returns:
        tuple[datasets.Dataset, datasets.Dataset]: Train and eval datasets
        with "input_ids" (corrupted encoder input) and "labels"
        (reconstruction target) columns.
    """
    print_banner("LOADING DATASET")
    raw_train = load_dataset(DATASET_NAME, split="train")
    raw_eval = load_dataset(DATASET_NAME, split="validation")

    raw_train = select_samples(raw_train, args.max_samples, args.sample_selection, args.seed)
    raw_eval = select_samples(raw_eval, args.max_eval_samples, "first", args.seed)

    print(f"Train rows (before packing): {len(raw_train)}")
    print(f"Eval rows (before packing): {len(raw_eval)}")

    train_blocks = tokenize_and_pack(raw_train, tokenizer, args.block_size, "train")
    eval_blocks = tokenize_and_pack(raw_eval, tokenizer, args.block_size, "eval")
    print(f"Train blocks (after packing to {args.block_size} tokens): {len(train_blocks)}")
    print(f"Eval blocks (after packing to {args.block_size} tokens): {len(eval_blocks)}")

    rng = np.random.default_rng(args.seed)

    def corrupt(example):
        """Apply span corruption to one packed block, returning encoder/target ids."""
        enc, tgt = apply_span_corruption(example["input_ids"], tokenizer, args.noise_density, args.mean_noise_span_length, rng)
        return {"input_ids": enc, "labels": tgt}

    train_dataset = train_blocks.map(corrupt, desc="Applying span corruption (train)")
    eval_dataset = eval_blocks.map(corrupt, desc="Applying span corruption (eval)")
    print()
    return train_dataset, eval_dataset


def build_model(args, tokenizer):
    """Construct a randomly initialized T5-style model sized by CLI args.

    Args:
        args (argparse.Namespace): Parsed CLI args; uses hidden_size,
            num_encoder_layers, num_decoder_layers, num_attention_heads,
            and gradient_checkpointing.
        tokenizer: Tokenizer used to size the vocabulary and set special
            token ids.

    Returns:
        transformers.PreTrainedModel: Freshly initialized encoder-decoder
        model with tied embeddings.
    """
    config = T5Config(
        vocab_size=len(tokenizer),
        d_model=args.hidden_size,
        d_kv=args.hidden_size // args.num_attention_heads,
        d_ff=4 * args.hidden_size,
        num_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        num_heads=args.num_attention_heads,
        decoder_start_token_id=tokenizer.pad_token_id,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    model = build_model_from_scratch(ARCHITECTURE, config, gradient_checkpointing=args.gradient_checkpointing)

    total_params = sum(p.numel() for p in model.parameters())
    embedding_params = model.shared.weight.numel()  # T5 ties encoder/decoder/lm_head embeddings
    print_banner("MODEL (FROM SCRATCH, RANDOM INIT)")
    print(f"Architecture: {ARCHITECTURE} (T5-style), vocab_size={config.vocab_size}, "
          f"hidden_size={config.d_model}, encoder_layers={config.num_layers}, decoder_layers={config.num_decoder_layers}, "
          f"num_heads={config.num_heads}")
    print(f"Total parameters:      {total_params:,}")
    print(f"Embedding parameters:  {embedding_params:,} ({100 * embedding_params / total_params:.1f}% of total, tied across encoder/decoder/lm_head)")
    print(f"Non-embedding params:  {total_params - embedding_params:,}")
    print()
    return model


def decode_example(example, index, tokenizer):
    """Decode a corrupted encoder input and its reconstruction target for display.

    Args:
        example (dict): One dataset row with "input_ids" (corrupted encoder
            input) and "labels" (reconstruction target) fields.
        index (int): Row index; unused, present for print_formatted_examples'
            decode_fn signature.
        tokenizer: Tokenizer to decode ids with.

    Returns:
        str: Human-readable side-by-side of the encoder input and target.
    """
    encoder_text = tokenizer.decode(example["input_ids"])
    target_text = tokenizer.decode(example["labels"])
    return f"Encoder input (corrupted):\n{encoder_text}\n\nDecoder target (reconstruction):\n{target_text}"


def main():
    """Parse CLI args, run from-scratch span-corruption pretraining, and write results.

    Loads the tokenizer and data (with span corruption applied), builds a
    randomly initialized model, optionally exits early for
    --debug_first_batch, otherwise trains with Trainer, evaluates
    reconstruction perplexity, saves the model, and writes a
    run_result.json.
    """
    args = build_arg_parser().parse_args()
    set_all_seeds(args.seed)
    print_config(args, "Span-corruption pretraining (encoder-decoder, from scratch)")

    # verify_dataset()  # commented out: avoids loading the dataset twice (a streaming peek here, then a full load in load_and_prepare_data()) to cut memory overhead
    tokenizer = load_tokenizer(TOKENIZER_NAME)

    train_dataset, eval_dataset = load_and_prepare_data(args, tokenizer)
    model = build_model(args, tokenizer)

    if args.debug_first_batch:
        print_formatted_examples(train_dataset, tokenizer, num_examples=2, decode_fn=decode_example)
        print("--debug_first_batch set: exiting without training.")
        return

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        bf16=(args.mixed_precision == "bf16"),
        fp16=(args.mixed_precision == "fp16"),
        gradient_checkpointing=args.gradient_checkpointing,
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        seed=args.seed,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    print_banner("TRAINING")
    start = time.time()
    trainer.train()
    train_runtime_seconds = time.time() - start

    print_banner("FINAL EVALUATION")
    eval_metrics = trainer.evaluate()
    eval_loss = eval_metrics["eval_loss"]
    perplexity = math.exp(eval_loss) if eval_loss < 20 else float("inf")
    print(f"Eval (reconstruction) loss: {eval_loss:.4f}  |  Perplexity: {perplexity:.2f}")

    save_model(model, tokenizer, args.output_dir, strategy="full")

    total_params = sum(p.numel() for p in model.parameters())
    write_run_result(
        output_dir=args.output_dir,
        stage="pretraining",
        task="span-corruption",
        modality="text",
        architecture=ARCHITECTURE,
        model_name=f"from-scratch-t5-style-{total_params}params",
        dataset_name=DATASET_NAME,
        hyperparameters=vars(args),
        metrics={"eval_loss": eval_loss, "perplexity": perplexity, "total_parameters": total_params},
        num_train_samples=len(train_dataset),
        num_eval_samples=len(eval_dataset),
        train_runtime_seconds=train_runtime_seconds,
    )
    print("Done.")


if __name__ == "__main__":
    main()
