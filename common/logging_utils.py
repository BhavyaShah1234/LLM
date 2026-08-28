"""Shared console-output conventions: config banners and formatted-example dumps.

Every script in this project prints the same shape of startup banner and, when
--debug_first_batch is passed, the same shape of "here's what a training
example actually looks like" dump. Keeping that formatting here (not the
decision of *what* to print, which is task-specific) is what lets a user
skim any script's output and immediately recognize the structure.
"""

import argparse
from typing import Callable, Optional


def print_banner(title: str, width: int = 80) -> None:
    print("=" * width)
    print(title)
    print("=" * width)


def print_config(args: argparse.Namespace, task_description: str) -> None:
    print_banner("CONFIGURATION")
    print(f"Task: {task_description}")
    for key, value in sorted(vars(args).items()):
        print(f"  {key}: {value}")
    print()


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
