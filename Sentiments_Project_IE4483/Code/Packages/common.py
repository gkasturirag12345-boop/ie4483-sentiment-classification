"""Shared paths, seeding and device selection."""
import os
import random
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]             # repository root
# train.json / test.json are course data and are not in the repository: put them in data/ or point IE4483_DATA at them.
DATA_DIR = Path(os.environ.get("IE4483_DATA", REPO / "data")).expanduser()
RESULTS = Path(os.environ.get("IE4483_RESULTS", REPO / "results")).expanduser()
FIGURES = REPO / "Sentiments_Project_IE4483" / "figures"
MODELS = REPO / "models"                               # git-ignored


def ensure_dirs() -> None:
    """Create the output folders. Called by the code that writes, never at import time."""
    for d in (RESULTS, FIGURES, MODELS):
        d.mkdir(parents=True, exist_ok=True)

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
