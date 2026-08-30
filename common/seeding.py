"""Reproducibility: seed every RNG a training script touches."""

import random

import numpy as np
import torch


def set_all_seeds(seed: int) -> None:
    """Seed every RNG a training script touches (Python, NumPy, PyTorch CPU/CUDA).

    Args:
        seed (int): Seed value to apply to all RNGs.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
