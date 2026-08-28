"""Shared --max_samples / --sample_selection truncation logic.

Every script in this project exposes the same two flags with the same
semantics (see root README.md, "Shared CLI flag conventions"): `--max_samples`
defaults to -1 (use the whole dataset), and `--sample_selection` picks *which*
samples get used when it's a positive integer. That behavior is identical
everywhere, so it lives here rather than being reimplemented per script.
"""

from typing import Union


def select_samples(
    dataset,
    max_samples: int,
    sample_selection: str,
    seed: int,
    is_streaming: bool = False,
):
    """max_samples == -1 means "use everything" (dataset returned unchanged).
    sample_selection: 'random' | 'first' | 'last'. A max_samples larger than
    the dataset is clamped to the dataset's actual size rather than raising
    an IndexError -- found necessary in practice: some HF dataset mirrors
    have far fewer rows than expected for a given split (e.g. a "validation"
    split with 60 rows), and scripts' --max_eval_samples defaults are sized
    for typical splits, not every possible dataset's actual split sizes.
    """
    if max_samples == -1:
        return dataset

    if is_streaming:
        if sample_selection == "last":
            raise ValueError(
                "--sample_selection last is not supported on a streamed dataset "
                "without materializing the full stream first; use 'first' or "
                "'random', or disable streaming for this run."
            )
        if sample_selection == "random":
            dataset = dataset.shuffle(seed=seed, buffer_size=max(10_000, max_samples * 10))
        elif sample_selection != "first":
            raise ValueError(f"Unknown sample_selection {sample_selection!r}; expected 'random', 'first', or 'last'")
        return dataset.take(max_samples)

    max_samples = min(max_samples, len(dataset))
    if sample_selection == "random":
        dataset = dataset.shuffle(seed=seed)
        return dataset.select(range(max_samples))
    if sample_selection == "first":
        return dataset.select(range(max_samples))
    if sample_selection == "last":
        n = len(dataset)
        return dataset.select(range(max(0, n - max_samples), n))
    raise ValueError(f"Unknown sample_selection {sample_selection!r}; expected 'random', 'first', or 'last'")
