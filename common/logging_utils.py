"""Shared console-output conventions: config banners and formatted-example dumps.

Every script in this project prints the same shape of startup banner and, when
--debug_first_batch is passed, the same shape of "here's what a training
example actually looks like" dump. Keeping that formatting here (not the
decision of *what* to print, which is task-specific) is what lets a user
skim any script's output and immediately recognize the structure.
"""

import argparse
import datetime
import os
from typing import Callable, List, Optional, Tuple


def print_banner(title: str, width: int = 80) -> None:
    """Print a title surrounded by `=`-rule lines.

    Args:
        title (str): Text to print between the two rules.
        width (int): Width in characters of each `=` rule.
    """
    print("=" * width)
    print(title)
    print("=" * width)


def print_config(args: argparse.Namespace, task_description: str) -> None:
    """Print a standardized startup banner listing every parsed CLI argument.

    Args:
        args (argparse.Namespace): Parsed CLI arguments to dump, sorted by
            name.
        task_description (str): One-line description of what the script is
            about to run, printed above the argument list.
    """
    print_banner("CONFIGURATION")
    print(f"Task: {task_description}")
    for key, value in sorted(vars(args).items()):
        print(f"  {key}: {value}")
    print()


def log_generation_samples(
    output_dir: str,
    stage_label: str,
    samples: List[Tuple[str, str]],
    filename: str = "sample_generations.txt",
) -> str:
    """Append a labeled block of qualitative generation samples to a log file.

    Used by pretraining/ scripts to record what the model produces before
    and after each training stage, so progress across an overnight chain of
    resumed runs can be inspected as plain text afterward.

    Args:
        output_dir (str): Directory to write/append the log file into
            (created if missing).
        stage_label (str): Short label for this block (e.g. "before
            training", "after epochs 0-10").
        samples (List[Tuple[str, str]]): `(prompt, output)` pairs to log.
        filename (str): Log file name within `output_dir`.

    Returns:
        str: The path to the appended log file.
    """
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    with open(path, "a") as f:
        f.write(f"\n{'=' * 80}\n{stage_label}  ({timestamp})\n{'=' * 80}\n")
        for prompt, output in samples:
            f.write(f"\n--- Prompt/input ---\n{prompt}\n--- Output ---\n{output}\n")
    print(f"[generation_log] appended to {path}")
    return path


def print_formatted_examples(
    dataset,
    tokenizer,
    num_examples: int = 2,
    decode_fn: Optional[Callable] = None,
) -> None:
    """Print the first `num_examples` formatted training examples.

    `decode_fn(example, index, tokenizer) -> str` lets a script (e.g. VQA,
    which has an image field that can't just be tokenizer.decode()'d) supply
    its own rendering; the default assumes `example["input_ids"]` /
    `example["labels"]` are already tokenized with -100 masking applied, and
    prints the decoded input alongside which spans are masked vs. trained on.

    Args:
        dataset: A `datasets.Dataset`-like object supporting `__len__` and
            integer indexing.
        tokenizer: Tokenizer used to decode `input_ids`/`labels` when
            `decode_fn` is not supplied.
        num_examples (int): Maximum number of examples to print.
        decode_fn (Optional[Callable]): Optional `(example, index, tokenizer)
            -> str` override for rendering a single example.
    """
    print_banner("FORMATTED EXAMPLES")
    n = min(num_examples, len(dataset))
    for i in range(n):
        example = dataset[i]
        print(f"\n--- Example {i + 1} ---")
        if decode_fn is not None:
            print(decode_fn(example, i, tokenizer))
            continue
        input_ids = example["input_ids"]
        labels = example.get("labels")
        print("Full text:")
        print(tokenizer.decode(input_ids, skip_special_tokens=False))
        if labels is not None:
            trained_ids = [tid for tid, lab in zip(input_ids, labels) if lab != -100]
            print("\nTokens loss is actually computed on:")
            print(tokenizer.decode(trained_ids, skip_special_tokens=False) if trained_ids else "(none)")
    print()
