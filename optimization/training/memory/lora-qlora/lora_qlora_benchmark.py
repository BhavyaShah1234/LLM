"""LoRA / QLoRA -- isolated benchmark script (training, memory).

Compares three ways of training the same model on a real optimizer step
(forward + backward + `optimizer.step()`, not just forward+backward like
the other optimization/ benchmarks in this wave -- optimizer state is
exactly what makes this comparison interesting, see below):

  - `full`: every parameter trainable, plain AdamW.
  - `lora`: base weights frozen (bf16), small LoRA adapter matrices
    trainable, AdamW only tracks the adapter's (tiny) parameter count.
  - `qlora`: base weights frozen AND loaded in 4-bit (bitsandbytes NF4),
    LoRA adapter trainable in bf16 on top -- this project's `--quantization
    4bit --lora` combination used throughout supervised-finetuning/ and
    rlhf/.

**Why this needs a real optimizer step, unlike the other benchmarks in
this wave**: AdamW keeps two extra fp32 buffers per trainable parameter
(first and second moment estimates) on top of the parameter and its
gradient -- roughly 12-16 bytes/trainable-param total depending on dtype
mix. For `full` fine-tuning a 1.7B model, that alone is on the order of
20GB+, regardless of how the forward/backward pass itself is optimized
(flash attention, mixed precision, gradient checkpointing all reduce
activation/compute cost, not optimizer-state cost). LoRA/QLoRA's real
lever is shrinking the trainable parameter COUNT (from ~1.7B to a few
million), which shrinks optimizer state proportionally -- a fundamentally
different axis than every other technique benchmarked in this project's
optimization/ folder so far.

Expect (and this script is built to survive, not just measure) the `full`
config to OOM on this project's 8GB target hardware -- that OOM, caught
and reported rather than crashing the script, IS the finding this
benchmark exists to produce concretely, not a bug to work around.

Synthetic random-token batches are used instead of a real dataset -- same
rationale as the other optimization/ compute benchmarks in this wave.

Usage:
    python lora_qlora_benchmark.py --debug_first_batch
    python lora_qlora_benchmark.py --batch_size 1 --seq_len 256 --num_steps 5
"""

import argparse
import time

import torch

from common.logging_utils import print_banner, print_config
from common.peft_setup import apply_lora, build_lora_config
from common.quantization import build_quantization_config
from common.run_results import write_run_result
from common.seeding import set_all_seeds

ARCHITECTURE = "decoder-only"
CONFIGS = ["full", "lora", "qlora"]


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for this benchmark.

    Returns:
        argparse.ArgumentParser: Parser with all benchmark options (model,
        synthetic-batch shape, step counts, learning rate, seed, output
        dir, debug flag) registered.
    """
    p = argparse.ArgumentParser(description="Benchmark full fine-tuning vs. LoRA vs. QLoRA on a real optimizer step.")
    p.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B-Base", help="Model to benchmark (local path or HF Hub id). Default: this project's usual base-model default -- large enough that full fine-tuning's optimizer-state cost is the point.")
    p.add_argument("--batch_size", type=int, default=1, help="Synthetic batch size. Default: 1.")
    p.add_argument("--seq_len", type=int, default=256, help="Synthetic sequence length. Default: 256.")
    p.add_argument("--num_steps", type=int, default=5, help="Timed optimizer steps per configuration (after warmup). Default: 5.")
    p.add_argument("--num_warmup", type=int, default=1, help="Untimed warmup steps. Default: 1.")
    p.add_argument("--learning_rate", type=float, default=1e-5, help="AdamW learning rate. Default: 1e-5.")
    p.add_argument("--seed", type=int, default=42, help="Random seed. Default: 42.")
    p.add_argument("--output_dir", type=str, default="./output/optimization/lora_qlora_benchmark", help="Where to write run_result.json.")
    p.add_argument("--debug_first_batch", action="store_true", default=False, help="Build one synthetic batch and print trainable-param counts for each configuration, then exit without timing.")
    return p


def build_synthetic_batch(vocab_size: int, batch_size: int, seq_len: int, device: str):
    """Build a random-token CLM batch used to drive all three benchmark configurations.

    Args:
        vocab_size (int): Vocabulary size to sample token ids from.
        batch_size (int): Number of sequences in the batch.
        seq_len (int): Length of each sequence.
        device (str): Torch device to place the tensors on.

    Returns:
        dict: `{"input_ids", "attention_mask", "labels"}` tensors, with
        `labels` a clone of `input_ids` (standard CLM setup).
    """
    input_ids = torch.randint(low=0, high=vocab_size, size=(batch_size, seq_len), device=device)
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def load_model(model_name: str, config_name: str, device: str):
    """Load a causal LM configured as `full`, `lora`, or `qlora`.

    Args:
        model_name (str): Local path or HF Hub id of the base checkpoint to load.
        config_name (str): One of `CONFIGS` (`"full"`, `"lora"`, `"qlora"`) selecting the training configuration.
        device (str): Torch device to move the model to (unused for `"qlora"`, which uses `device_map="auto"`).

    Returns:
        transformers.PreTrainedModel: The loaded model (optionally wrapped
        with a LoRA adapter) in `.train()` mode.
    """
    from transformers import AutoModelForCausalLM

    if config_name == "qlora":
        quantization_config = build_quantization_config("4bit", "bf16")
        model = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=quantization_config, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
        )
        model = apply_lora(model, build_lora_config(), prepare_for_kbit=True, print_trainable=False)
    elif config_name == "lora":
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16, trust_remote_code=True).to(device)
        model = apply_lora(model, build_lora_config(), prepare_for_kbit=False, print_trainable=False)
    else:  # full
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16, trust_remote_code=True).to(device)
    model.train()
    return model


def count_params(model):
    """Count trainable vs. total parameters of a model.

    Args:
        model (torch.nn.Module): Model to inspect.

    Returns:
        tuple[int, int]: `(trainable_params, total_params)`.
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def run_benchmark(model, batch, num_steps: int, num_warmup: int, learning_rate: float, device: str):
    """Time a real optimizer step (forward + backward + `optimizer.step()`) for one configuration.

    Args:
        model (torch.nn.Module): Model to benchmark, already in train mode on `device`.
        batch (dict): Batch from `build_synthetic_batch`, already on `device`.
        num_steps (int): Timed optimizer steps to run after warmup.
        num_warmup (int): Untimed warmup steps run before timing starts.
        learning_rate (float): AdamW learning rate for the trainable parameters.
        device (str): Torch device the model and batch live on.

    Returns:
        tuple[float, float, float]: `(avg_step_seconds, peak_memory_mb, final_loss)`
        -- mean wall-clock time per timed step, peak CUDA memory allocated
        during the timed steps in MB, and the loss from the last timed step.
    """
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate)

    for _ in range(num_warmup):
        optimizer.zero_grad(set_to_none=True)
        outputs = model(**batch)
        outputs.loss.backward()
        optimizer.step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(device)

    start = time.time()
    final_loss = None
    for _ in range(num_steps):
        optimizer.zero_grad(set_to_none=True)
        outputs = model(**batch)
        final_loss = outputs.loss.item()
        outputs.loss.backward()
        optimizer.step()
    torch.cuda.synchronize()
    elapsed = time.time() - start

    peak_memory_mb = torch.cuda.max_memory_allocated(device) / 1e6
    avg_step_seconds = elapsed / num_steps
    return avg_step_seconds, peak_memory_mb, final_loss


def main():
    """Run the full/LoRA/QLoRA benchmark and write a `run_result.json`.

    Parses CLI args, builds a synthetic CLM batch, benchmarks each
    configuration in `CONFIGS` (or just prints trainable-param counts if
    `--debug_first_batch` is set), catching and recording per-configuration
    OOMs as an expected finding rather than crashing, prints a summary, and
    records the comparison via `common.run_results.write_run_result`.
    """
    args = build_arg_parser().parse_args()
    set_all_seeds(args.seed)
    print_config(args, "LoRA / QLoRA benchmark (full fine-tune vs. LoRA vs. QLoRA)")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    from transformers import AutoConfig

    vocab_size = AutoConfig.from_pretrained(args.model, trust_remote_code=True).vocab_size
    batch = build_synthetic_batch(vocab_size, args.batch_size, args.seq_len, device)
    print(f"Synthetic batch: batch_size={args.batch_size}, seq_len={args.seq_len}, vocab_size={vocab_size}")

    if args.debug_first_batch:
        for config_name in CONFIGS:
            print_banner(f"DEBUG: config={config_name}")
            model = load_model(args.model, config_name, device)
            trainable, total = count_params(model)
            print(f"trainable_params={trainable:,}  total_params={total:,}  ({100 * trainable / total:.3f}% trainable)")
            del model
            torch.cuda.empty_cache()
        print("--debug_first_batch set: exiting without the full benchmark.")
        return

    results = {}
    for config_name in CONFIGS:
        print_banner(f"BENCHMARKING config={config_name}")
        model = None
        try:
            model = load_model(args.model, config_name, device)
            trainable, total = count_params(model)
            print(f"trainable_params={trainable:,}  total_params={total:,}  ({100 * trainable / total:.3f}% trainable)")
            avg_step_seconds, peak_memory_mb, final_loss = run_benchmark(model, batch, args.num_steps, args.num_warmup, args.learning_rate, device)
            results[config_name] = {
                "trainable_params": trainable, "total_params": total,
                "avg_step_seconds": avg_step_seconds, "peak_memory_mb": peak_memory_mb,
                "final_loss": final_loss, "oom": False,
            }
            print(f"avg_step_seconds={avg_step_seconds:.4f}  peak_memory_mb={peak_memory_mb:.1f}  final_loss={final_loss:.4f}")
        except torch.OutOfMemoryError as e:
            print(f"OOM: {config_name} does not fit on this hardware with a real optimizer step -- this is a real, expected finding, not a bug. ({e})")
            results[config_name] = {"trainable_params": None, "total_params": None, "avg_step_seconds": None, "peak_memory_mb": None, "final_loss": None, "oom": True}
        finally:
            # Critical: an OOM leaves `model` (and its optimizer's partially-allocated state)
            # referenced by this local variable. Without explicitly deleting it here,
            # torch.cuda.empty_cache() only releases already-freed cached blocks, not this
            # still-live allocation -- confirmed via a live cascading failure where the
            # "full" config's OOM left enough memory pinned that "lora" (which fits on its
            # own) OOM'd too, and "qlora" then failed for a third, unrelated reason.
            del model
            torch.cuda.empty_cache()

    print_banner("SUMMARY")
    for config_name, r in results.items():
        if r["oom"]:
            print(f"{config_name}: OOM (did not fit)")
        else:
            print(f"{config_name}: trainable_params={r['trainable_params']:,}  peak_memory_mb={r['peak_memory_mb']:.1f}  avg_step_seconds={r['avg_step_seconds']:.4f}")

    metrics = {}
    for config_name, r in results.items():
        for k, v in r.items():
            metrics[f"{config_name}_{k}"] = v

    write_run_result(
        output_dir=args.output_dir,
        stage="optimization",
        task="lora_qlora_benchmark",
        modality="text",
        architecture=ARCHITECTURE,
        model_name=args.model,
        dataset_name="synthetic random tokens (no dataset -- compute/memory benchmark, see module docstring)",
        hyperparameters=vars(args),
        metrics=metrics,
        num_train_samples=0,
        num_eval_samples=0,
        train_runtime_seconds=sum((r["avg_step_seconds"] or 0) * args.num_steps for r in results.values()),
    )
    print("Done.")


if __name__ == "__main__":
    main()
