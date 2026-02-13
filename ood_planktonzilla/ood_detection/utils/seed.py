"""Reproducibility utilities for setting random seeds."""

import random
import numpy as np
import torch


def set_seed(seed_value=42):
    """
    Set random seeds for Python, NumPy, and PyTorch to ensure reproducibility.
    
    Args:
        seed_value (int): The seed value to use. Default is 42.
    """
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
