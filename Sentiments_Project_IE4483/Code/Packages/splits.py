"""Leakage-safe stratified, grouped K-fold splits.

Reviews are grouped by their normalised text, so duplicate reviews always fall in the
same fold and can never appear in both a training fold and its validation fold.
"""
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from common import HOLDOUT_FOLD, N_FOLDS, SEED
from preprocessing import group_key


def get_folds(df, n_splits: int = N_FOLDS, seed: int = SEED):
    groups = df["reviews"].map(group_key).values
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = list(sgkf.split(np.zeros(len(df)), df["sentiments"].values, groups))
    for tr, va in folds:  # explicit leakage check
        assert not set(groups[tr]) & set(groups[va]), "duplicate text crosses train/val boundary"
    return folds


def holdout(df):
    """The common ~80/20 hold-out split (fold HOLDOUT_FOLD) used to compare all models."""
    return get_folds(df)[HOLDOUT_FOLD]


if __name__ == "__main__":
    from data_loader import load_train
    df = load_train()
    for i, (tr, va) in enumerate(get_folds(df)):
        print(f"fold {i}: train {len(tr)} val {len(va)} val_pos_rate {df.sentiments.values[va].mean():.3f}")
