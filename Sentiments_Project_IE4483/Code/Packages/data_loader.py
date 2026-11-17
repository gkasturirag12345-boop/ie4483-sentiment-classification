"""Load and validate train.json / test.json, and write a data profile.

Run directly to produce results/data_profile.json:  python src/data_loader.py
"""
import json

import pandas as pd

from common import DATA_DIR, RESULTS, ensure_dirs
from preprocessing import group_key


def _load(name: str, labelled: bool) -> pd.DataFrame:
    path = DATA_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: the course data is not in the repository; "
                                f"put {name} in {DATA_DIR} or set IE4483_DATA")
    # The files are JSON arrays of objects (verified by inspection), not CSV.
    df = pd.read_json(path, orient="records")
    if "reviews" not in df.columns:
        raise ValueError(f"{name}: missing 'reviews' column, found {list(df.columns)}")
    if labelled:
        if "sentiments" not in df.columns:
            raise ValueError(f"{name}: missing 'sentiments' column")
        bad = ~df["sentiments"].isin([0, 1])
        if bad.any():
            raise ValueError(f"{name}: {bad.sum()} labels outside {{0,1}}")
        df["sentiments"] = df["sentiments"].astype(int)
    df["reviews"] = df["reviews"].astype("string")
    df.insert(0, "row_id", range(len(df)))  # original row order, used to rebuild submission
    return df


def load_train() -> pd.DataFrame:
    df = _load("train.json", labelled=True)
    empty = df["reviews"].isna() | (df["reviews"].str.strip() == "")
    if empty.any():
        # Empty reviews carry no sentiment evidence; they are dropped from training (none at time of writing).
        print(f"[data_loader] dropping {empty.sum()} empty training reviews")
        df = df[~empty].reset_index(drop=True)
    return df


def load_test() -> pd.DataFrame:
    df = _load("test.json", labelled=False)
    # Empty test reviews are kept (every test row needs a prediction); fill with "" so models still run.
    df["reviews"] = df["reviews"].fillna("")
    if "sentiments" in df.columns:
        print("[data_loader] WARNING: test.json contains labels; they are ignored for training/selection")
    return df


def profile() -> dict:
    tr, te = load_train(), load_test()
    out = {}
    for name, df in (("train", tr), ("test", te)):
        words = df["reviews"].str.split().str.len()
        out[name] = {
            "n_rows": int(len(df)),
            "columns": [c for c in df.columns if c != "row_id"],
            "missing_or_empty_reviews": int((df["reviews"].str.strip() == "").sum()),
            "exact_duplicate_reviews": int(df["reviews"].duplicated().sum()),
            "normalised_duplicate_reviews": int(df["reviews"].map(group_key).duplicated().sum()),
            "words_mean": round(float(words.mean()), 1),
            "words_median": int(words.median()),
            "words_p95": int(words.quantile(0.95)),
            "words_max": int(words.max()),
            "chars_mean": round(float(df["reviews"].str.len().mean()), 1),
        }
    out["train"]["label_counts"] = {str(k): int(v) for k, v in tr["sentiments"].value_counts().sort_index().items()}
    out["train"]["positive_rate"] = round(float(tr["sentiments"].mean()), 4)
    g = tr.groupby(tr["reviews"].map(group_key))["sentiments"].nunique()
    out["train"]["duplicate_groups_with_conflicting_labels"] = int((g > 1).sum())
    tr_keys = set(tr["reviews"].map(group_key))
    out["test"]["reviews_also_in_train_normalised"] = int(te["reviews"].map(group_key).isin(tr_keys).sum())
    out["test"]["has_labels"] = "sentiments" in te.columns
    ensure_dirs()
    (RESULTS / "data_profile.json").write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    print(json.dumps(profile(), indent=2))
    tr = load_train()
    for lab in (0, 1):
        print(f"\n--- example label {lab} ---")
        for t in tr[tr.sentiments == lab]["reviews"].head(2):
            print(" *", t[:200])
