"""Named Entity Recognition SFT -- decoder-only, standard.

Dataset: MorryShah/complex_ner. Fields verified live: id, language, text,
entities (entities already a list of {"text": ..., "type": ...} dicts) --
this is simpler than assumed by the archived version of this script
(old/ner_standard.py), which expected CoNLL-style `tokens`/`ner_tags`
fields that don't actually exist in this dataset, so it silently built every
training example from an empty token list (empty text, empty entities). See
supervised-finetuning/README.md.

Format: ### Instruction: ... ### Input: ... ### Response: {entities_json}
Loss: entity JSON tokens only.

Usage:
    python ner_standard.py --debug_first_batch --max_samples 20
"""

import argparse
import json
import time

import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments

from common.data_selection import select_samples
from common.logging_utils import print_banner, print_config, print_formatted_examples
from common.model_loading import load_causal_lm, load_tokenizer
from common.model_saving import save_model
from common.peft_setup import apply_lora, build_lora_config
from common.quantization import build_quantization_config
from common.run_results import write_run_result
from common.seeding import set_all_seeds

DATASET_NAME = "MorryShah/complex_ner"
ARCHITECTURE = "decoder-only"
INSTRUCTION = "Extract all named entities from the text and provide them in JSON format with entity text and label."
MAX_ENTITIES = 20


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for this training script.

    Returns:
        argparse.ArgumentParser: Parser covering model/quantization, LoRA,
            training hyperparameters, data sampling, output/checkpointing,
            and seeding/debug flags.
    """
    p = argparse.ArgumentParser(description="SFT a decoder-only model for NER (standard).")

    p.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B-Base", help="Base checkpoint. Default: Qwen/Qwen3-1.7B-Base.")
    p.add_argument("--quantization", type=str, default="no", choices=["no", "4bit", "8bit"], help="Quantization for loading. Default: no.")

    p.add_argument("--lora", action="store_true", default=False, help="Enable LoRA finetuning.")
    p.add_argument("--lora_r", type=int, default=16, help="LoRA rank. Default: 16.")
    p.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha. Default: 32.")
    p.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout. Default: 0.05.")
    p.add_argument("--lora_target_modules", type=str, default="q_proj,k_proj,v_proj,o_proj", help="Comma-separated LoRA target modules.")

    p.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"], help="Mixed precision. Default: bf16.")
    p.add_argument("--batch_size", type=int, default=2, help="Per-device training batch size. Default: 2.")
    p.add_argument("--eval_batch_size", type=int, default=4, help="Per-device eval batch size. Default: 4.")
    p.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps. Default: 4.")
    p.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate. Default: 2e-4.")
    p.add_argument("--epochs", type=int, default=3, help="Training epochs. Default: 3.")
    p.add_argument("--max_length", type=int, default=1536, help="Max sequence length (NER source docs are long). Default: 1536.")
    p.add_argument("--warmup_steps", type=int, default=100, help="LR warmup steps. Default: 100.")
    p.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay. Default: 0.01.")
    p.add_argument("--gradient_checkpointing", action="store_true", default=True, help="Enable gradient checkpointing. Default: on.")

    p.add_argument("--max_samples", type=int, default=-1, help="Number of training rows to use. -1 (default) = full split.")
    p.add_argument("--sample_selection", type=str, default="random", choices=["random", "first", "last"], help="Which rows to pick when --max_samples is positive. Default: random.")
    p.add_argument("--max_eval_samples", type=int, default=100, help="Eval rows to use (generation is slow). Default: 100.")

    p.add_argument("--output_dir", type=str, default="./output/supervised-finetuning/text/ner/decoder-only/standard", help="Output directory.")
    p.add_argument("--logging_steps", type=int, default=10, help="Logging frequency. Default: 10.")
    p.add_argument("--eval_steps", type=int, default=200, help="Evaluation frequency. Default: 200.")
    p.add_argument("--save_steps", type=int, default=200, help="Checkpoint save frequency. Default: 200.")
    p.add_argument("--save_strategy", type=str, default="adapter_only", choices=["full", "adapter_only", "merged", "adapter_and_merged", "base_reference"], help="Model saving strategy. Default: adapter_only when --lora, otherwise 'full'.")

    p.add_argument("--seed", type=int, default=42, help="Random seed. Default: 42.")
    p.add_argument("--debug_first_batch", action="store_true", default=False, help="Print formatted examples and exit without training.")

    return p


def verify_dataset() -> None:
    """Peek the dataset via streaming and assert the expected fields are present.

    Not called by default in ``main`` (see the comment there) since it would
    otherwise trigger a second load of the same dataset.
    """
    print_banner("VERIFYING DATASET")
    peek = load_dataset(DATASET_NAME, split="train", streaming=True)
    example = next(iter(peek))
    for field in ("text", "entities"):
        assert field in example, f"Expected a {field!r} field in {DATASET_NAME}, got fields: {list(example.keys())}"
    print(f"Dataset: {DATASET_NAME}")
    print(f"Fields: {list(example.keys())}")
    print(f"Sample: {len(example['entities'])} entities, first: {example['entities'][0] if example['entities'] else None}")
    print()


class NERDataset(Dataset):
    """Tokenized NER dataset with entity-JSON-only supervision.

    Formats each row as ``### Instruction: ... ### Input: ... ### Response:
    {entities_json}`` and masks the prompt out of the loss, so loss is
    computed on the entity-JSON tokens only.

    Attributes:
        rows (list): Raw dataset rows with "text" and "entities" (list of
            {"text": ..., "type": ...} dicts) fields.
        tokenizer: Tokenizer used to encode prompt and full text.
        max_length (int): Max sequence length used when tokenizing the full example.
    """

    def __init__(self, rows, tokenizer, max_length: int):
        """Initialize the dataset.

        Args:
            rows (list): Raw dataset rows with "text" and "entities" fields.
            tokenizer: Tokenizer used to encode prompt and full text.
            max_length (int): Max sequence length used when tokenizing the full example.
        """
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        """Return the number of examples in the dataset.

        Returns:
            int: Number of rows.
        """
        return len(self.rows)

    def __getitem__(self, idx):
        """Build a tokenized, loss-masked training example with entity-JSON-only supervision.

        Args:
            idx (int): Index of the row to fetch.

        Returns:
            dict: ``input_ids``, ``attention_mask``, and ``labels`` (prompt
                tokens masked with -100; only the entity-JSON kept).
        """
        row = self.rows[idx]
        text = row["text"][:3000]
        entities = [{"entity": e["text"], "label": e["type"]} for e in row["entities"][:MAX_ENTITIES]]
        entities_json = json.dumps(entities, ensure_ascii=False)

        prompt = f"### Instruction:\n{INSTRUCTION}\n\n### Input:\n{text}\n\n### Response:\n"
        full_text = prompt + entities_json

        prompt_ids = self.tokenizer(prompt, add_special_tokens=True, truncation=False)["input_ids"]
        full = self.tokenizer(full_text, max_length=self.max_length, truncation=True, add_special_tokens=True)

        input_ids = full["input_ids"]
        labels = input_ids.copy()
        prompt_len = min(len(prompt_ids), len(input_ids))
        labels[:prompt_len] = [-100] * prompt_len

        return {"input_ids": input_ids, "attention_mask": full["attention_mask"], "labels": labels}


def load_and_prepare_data(args, tokenizer):
    """Load the train/test splits and apply sample selection.

    Args:
        args (argparse.Namespace): Parsed CLI args; uses ``max_samples``,
            ``sample_selection``, ``max_eval_samples``, ``max_length``, and ``seed``.
        tokenizer: Tokenizer passed through to the constructed datasets.

    Returns:
        tuple: ``(train_dataset, eval_dataset, eval_rows)`` where the first two
            are :class:`NERDataset` instances and ``eval_rows`` is the raw
            (untokenized) list of eval rows for later generation-based evaluation.
    """
    print_banner("LOADING DATASET")
    train_raw = load_dataset(DATASET_NAME, split="train")
    eval_raw = load_dataset(DATASET_NAME, split="test")

    train_raw = select_samples(train_raw, args.max_samples, args.sample_selection, args.seed)
    eval_raw = select_samples(eval_raw, args.max_eval_samples, "first", args.seed)

    print(f"Train samples: {len(train_raw)}")
    print(f"Eval samples: {len(eval_raw)}")
    print()

    train_rows = list(train_raw)
    eval_rows = list(eval_raw)
    return NERDataset(train_rows, tokenizer, args.max_length), NERDataset(eval_rows, tokenizer, args.max_length), eval_rows


def decode_example(example, index, tokenizer):
    """Render a tokenized example back to text for ``--debug_first_batch`` inspection.

    Args:
        example (dict): Tokenized example with ``input_ids`` and ``labels``.
        index (int): Position of this example in the batch being printed (unused
            in the body but kept for a uniform ``decode_fn`` signature).
        tokenizer: Tokenizer used to decode ``input_ids`` and the unmasked labels.

    Returns:
        str: Human-readable dump of the full formatted text and the
            entity-JSON portion the loss is computed on.
    """
    input_ids = example["input_ids"]
    labels = example["labels"]
    full_text = tokenizer.decode(input_ids, skip_special_tokens=False)
    label_tokens = [tid for tid, lab in zip(input_ids, labels) if lab != -100]
    label_text = tokenizer.decode(label_tokens, skip_special_tokens=True)
    return f"Full text:\n{full_text[:600]}...\n\nEntity JSON (loss computed on): {label_text[:400]}"


def evaluate_model(model, tokenizer, eval_rows, args):
    """Generate entity-JSON completions and score entity-level precision/recall/F1.

    Args:
        model: Trained causal LM used for generation.
        tokenizer: Tokenizer used to build prompts and decode generations.
        eval_rows (list): Raw eval rows with "text" and "entities" fields.
        args (argparse.Namespace): Parsed CLI args; uses ``max_length``.

    Returns:
        dict: F1/precision/recall over predicted vs. true (entity, label) sets.
    """
    print_banner("EVALUATION")
    model.eval()
    num_correct = num_pred = num_true = 0

    with torch.no_grad():
        for i, row in enumerate(eval_rows):
            if i % 20 == 0:
                print(f"  progress: {i}/{len(eval_rows)}")
            text = row["text"][:3000]
            prompt = f"### Instruction:\n{INSTRUCTION}\n\n### Input:\n{text}\n\n### Response:\n"
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=args.max_length).to(model.device)
            output = model.generate(**inputs, max_new_tokens=512, do_sample=False, pad_token_id=tokenizer.pad_token_id)
            generated = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

            try:
                pred_entities = json.loads(generated)
                if not isinstance(pred_entities, list):
                    pred_entities = []
            except json.JSONDecodeError:
                pred_entities = []

            pred_set = {(e.get("entity", ""), e.get("label", "")) for e in pred_entities if isinstance(e, dict)}
            true_set = {(e["text"], e["type"]) for e in row["entities"][:MAX_ENTITIES]}

            num_correct += len(pred_set & true_set)
            num_pred += len(pred_set)
            num_true += len(true_set)

    precision = num_correct / num_pred if num_pred else 0.0
    recall = num_correct / num_true if num_true else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    print(f"Entity-level F1: {f1:.4f}  Precision: {precision:.4f}  Recall: {recall:.4f}")
    return {"f1": f1, "precision": precision, "recall": recall}


def main():
    """Run the end-to-end NER standard pipeline: load, train, evaluate, save, record."""
    args = build_arg_parser().parse_args()
    set_all_seeds(args.seed)
    print_config(args, "NER SFT -- decoder-only, standard")

    # verify_dataset()  # commented out: avoids loading the dataset twice (a streaming peek here, then a full load in load_and_prepare_data()) to cut memory overhead
    tokenizer = load_tokenizer(args.model)

    train_dataset, eval_dataset, eval_rows = load_and_prepare_data(args, tokenizer)

    quant_config = build_quantization_config(args.quantization, args.mixed_precision)
    torch_dtype = torch.bfloat16 if args.mixed_precision == "bf16" else torch.float16
    model = load_causal_lm(args.model, quant_config, torch_dtype, args.gradient_checkpointing)

    if args.lora:
        lora_config = build_lora_config(args.lora_r, args.lora_alpha, args.lora_dropout, args.lora_target_modules)
        model = apply_lora(model, lora_config, prepare_for_kbit=(quant_config is not None))

    if args.debug_first_batch:
        print_formatted_examples(train_dataset, tokenizer, num_examples=2, decode_fn=decode_example)
        print("--debug_first_batch set: exiting without training.")
        return

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
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
        ddp_find_unused_parameters=(True if args.lora else None),
    )

    trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=data_collator)

    print_banner("TRAINING")
    start = time.time()
    trainer.train()
    train_runtime_seconds = time.time() - start

    metrics = evaluate_model(model, tokenizer, eval_rows, args)

    save_strategy = args.save_strategy if args.lora else "full"
    save_model(model, tokenizer, args.output_dir, strategy=save_strategy, base_model_name=args.model, is_lora=args.lora)

    write_run_result(
        output_dir=args.output_dir,
        stage="supervised-finetuning",
        task="ner",
        modality="text",
        architecture=ARCHITECTURE,
        variant="standard",
        cot_enabled=False,
        model_name=args.model,
        dataset_name=DATASET_NAME,
        save_strategy=save_strategy,
        hyperparameters=vars(args),
        metrics=metrics,
        num_train_samples=len(train_dataset),
        num_eval_samples=len(eval_dataset),
        train_runtime_seconds=train_runtime_seconds,
    )
    print("Done.")


if __name__ == "__main__":
    main()
