"""Load an adapter trained WITHOUT quantization (bf16 base + LoRA) and
evaluate it with the base re-loaded in 4-bit -- the "quantize an
unquantized-trained model for inference" arm of this experiment.

This is deliberately NOT a training script -- it exists purely to produce
the third data point `compare.py` needs, alongside two already-real
supervised-finetuning/text/mcq/decoder-only/mcq_standard.py runs:
  - `--quantization no` (bf16 training, bf16 eval) -- the reference.
  - `--quantization 4bit` (QLoRA: 4-bit training, 4-bit eval) -- "quantized
    training."
This script takes the FIRST run's saved adapter and re-attaches it to a
FRESH 4-bit-quantized base model load, then runs the identical eval
protocol -- "quantized inference of an unquantized-trained model." The
LoRA adapter weights themselves are never quantized in any of these three
arms (bitsandbytes' 4-bit quantization applies to the frozen base
`nn.Linear` weights only) -- what differs is only whether the BASE weights
the adapter was trained against were quantized during training or not.

Eval protocol (dataset, prompt format, generation, and accuracy/F1 scoring)
is duplicated from mcq_standard.py rather than imported, per this project's
"duplicate task-specific logic across standalone scripts" convention --
see root README's "Model Selection Philosophy" note on when common/ vs.
inline is appropriate. This is intentionally the exact same protocol so the
three accuracy numbers are directly comparable.

Usage:
    python quantize_and_eval.py --debug_first_batch
    python quantize_and_eval.py --adapter_dir ../../output/experiments/quantized-training-vs-quantized-inference/train-bf16
"""

import argparse

import torch
from datasets import load_dataset
from peft import PeftModel
from sklearn.metrics import accuracy_score, f1_score

from common.data_selection import select_samples
from common.logging_utils import print_banner, print_config
from common.model_loading import load_causal_lm, load_tokenizer
from common.quantization import build_quantization_config
from common.run_results import write_run_result
from common.seeding import set_all_seeds

DATASET_NAME = "araag2/MedMCQA"
DATASET_CONFIG = "processed"
ARCHITECTURE = "decoder-only"
INSTRUCTION = "Answer the following medical multiple choice question by selecting the correct option (A, B, C, or D)."


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate a bf16-trained LoRA adapter with the base model re-loaded in 4-bit.")
    p.add_argument("--base_model", type=str, default="Qwen/Qwen3-1.7B-Base", help="Base checkpoint the adapter was trained against. Default: Qwen/Qwen3-1.7B-Base (must match --adapter_dir's training run).")
    p.add_argument("--adapter_dir", type=str, default="./output/experiments/quantized-training-vs-quantized-inference/train-bf16", help="Adapter trained WITHOUT quantization (mcq_standard.py --lora --quantization no output). Default: this experiment's own bf16-trained run.")
    p.add_argument("--max_length", type=int, default=1024, help="Max prompt length. Default: 1024.")
    p.add_argument("--max_eval_samples", type=int, default=40, help="Eval rows -- match the training runs' --max_eval_samples for a fair comparison. Default: 40.")
    p.add_argument("--seed", type=int, default=42, help="Random seed -- match the training runs' --seed so the SAME eval rows are used across all three arms. Default: 42.")
    p.add_argument("--output_dir", type=str, default="./output/experiments/quantized-training-vs-quantized-inference/quantized-inference-of-bf16-trained", help="Where to write run_result.json.")
    p.add_argument("--debug_first_batch", action="store_true", default=False, help="Load the model and print 2 formatted predictions, then exit without full evaluation.")
    return p


def format_question(row) -> str:
    return (
        f"{row['Question']}\n\n"
        f"A. {row['Option_A']}\n"
        f"B. {row['Option_B']}\n"
        f"C. {row['Option_C']}\n"
        f"D. {row['Option_D']}"
    )


def build_prompt(row) -> str:
    return f"### Instruction:\n{INSTRUCTION}\n\n### Input:\n{format_question(row)}\n\n### Response:\n"


def load_eval_rows(max_eval_samples: int, seed: int):
    print_banner("LOADING DATASET")
    # Same split choice as mcq_standard.py / grpo.py: "test" has Label=None for every row.
    eval_raw = load_dataset(DATASET_NAME, DATASET_CONFIG, split="dev")
    eval_raw = select_samples(eval_raw, max_eval_samples, "first", seed)
    eval_rows = list(eval_raw)
    print(f"Eval samples: {len(eval_rows)}")
    return eval_rows


def evaluate(model, tokenizer, eval_rows, max_length: int):
    print_banner("EVALUATION (4-bit-quantized base + bf16-trained adapter)")
    model.eval()
    predictions, ground_truths = [], []
    with torch.no_grad():
        for i, row in enumerate(eval_rows):
            if i % 20 == 0:
                print(f"  progress: {i}/{len(eval_rows)}")
            prompt = build_prompt(row)
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length).to(model.device)
            output = model.generate(**inputs, max_new_tokens=5, do_sample=False, pad_token_id=tokenizer.pad_token_id)
            generated = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip().upper()
            pred = next((c for c in generated[:10] if c in "ABCD"), "A")
            predictions.append(pred)
            ground_truths.append(row["Label"])

    accuracy = accuracy_score(ground_truths, predictions)
    f1_macro = f1_score(ground_truths, predictions, average="macro")
    print(f"Accuracy: {accuracy:.4f}  F1(macro): {f1_macro:.4f}")
    return {"accuracy": accuracy, "f1_macro": f1_macro}


def main():
    args = build_arg_parser().parse_args()
    set_all_seeds(args.seed)
    print_config(args, "Quantized inference of an unquantized-trained (bf16) adapter")

    eval_rows = load_eval_rows(args.max_eval_samples, args.seed)

    print_banner("LOADING 4-BIT-QUANTIZED BASE + BF16-TRAINED ADAPTER")
    tokenizer = load_tokenizer(args.base_model)
    quant_config = build_quantization_config("4bit", "bf16")
    base_model = load_causal_lm(args.base_model, quant_config, torch.bfloat16, gradient_checkpointing=False)
    model = PeftModel.from_pretrained(base_model, args.adapter_dir)

    if args.debug_first_batch:
        for row in eval_rows[:2]:
            prompt = build_prompt(row)
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                output = model.generate(**inputs, max_new_tokens=5, do_sample=False, pad_token_id=tokenizer.pad_token_id)
            generated = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            print(f"Label={row['Label']!r}  Generated={generated!r}")
        print("--debug_first_batch set: exiting without full evaluation.")
        return

    metrics = evaluate(model, tokenizer, eval_rows, args.max_length)

    write_run_result(
        output_dir=args.output_dir,
        stage="experiments",
        task="quantized_inference_of_unquantized_trained_adapter",
        modality="text",
        architecture=ARCHITECTURE,
        model_name=args.base_model,
        dataset_name=f"{DATASET_NAME} ({DATASET_CONFIG})",
        hyperparameters=vars(args),
        metrics=metrics,
        num_train_samples=0,
        num_eval_samples=len(eval_rows),
        train_runtime_seconds=0.0,
    )
    print("Done.")


if __name__ == "__main__":
    main()
