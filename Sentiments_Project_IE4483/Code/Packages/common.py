"""Shared paths, seeding and device selection."""
import os
import random
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]          # sentiment_project/
DATA_DIR = PROJECT.parent                              # original assignment folder (read-only)
RESULTS = PROJECT / "results"
FIGURES = PROJECT / "figures"
MODELS = PROJECT / "models"
for d in (RESULTS, FIGURES, MODELS):
    d.mkdir(exist_ok=True)

SEED = 42
SEEDS = [42, 43, 44]
N_FOLDS = 5
HOLDOUT_FOLD = 0  # fold used as the common validation set for all models


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_device():
    """Prefer Apple MPS, then CUDA, then CPU."""
    import torch
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
