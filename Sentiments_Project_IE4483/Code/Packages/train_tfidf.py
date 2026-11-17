"""Model A: TF-IDF features + linear classifier, tuned by 5-fold stratified grouped CV.

Outputs
  results/tfidf_cv_grid.csv          mean/std of every grid configuration over 5 folds
  results/val_preds_tfidf_lr.csv      hold-out (fold 0) predictions of the best LR config
  results/best_tfidf_config.json      selected configuration
Usage:  python src/train_tfidf.py
"""
import itertools
import json
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from common import HOLDOUT_FOLD, RESULTS, SEED, set_seed, ensure_dirs
from data_loader import load_train
from evaluate import log_run, metrics
from preprocessing import TOKEN_PATTERN, normalise
from splits import get_folds

FEATURE_GRID = [dict(ngram=ng, char=ch, min_df=md)
                for ng, ch, md in itertools.product([(1, 1), (1, 2)], [False, True], [1, 2])]
CLF_GRID = ([("lr", dict(C=C, class_weight=cw)) for C in (0.5, 1, 2, 4, 8) for cw in (None, "balanced")]
            + [("svm", dict(C=C, class_weight=cw)) for C in (0.1, 0.5, 1) for cw in (None, "balanced")])


def build_vectorizers(fc: dict):
    """Word TF-IDF (+ optional char_wb 2-5 TF-IDF); fitted on training data only by the caller."""
    vecs = [TfidfVectorizer(preprocessor=normalise, token_pattern=TOKEN_PATTERN, ngram_range=fc["ngram"],
                            min_df=fc["min_df"], sublinear_tf=True)]
    if fc["char"]:
        vecs.append(TfidfVectorizer(preprocessor=normalise, analyzer="char_wb", ngram_range=(2, 5),
                                    min_df=fc["min_df"], sublinear_tf=True, max_features=100_000))
    return vecs


def featurise(fc, train_text, *other_texts):
    vecs = build_vectorizers(fc)
    Xtr = sp.hstack([v.fit_transform(train_text) for v in vecs]).tocsr()
    others = [sp.hstack([v.transform(t) for v in vecs]).tocsr() for t in other_texts]
    return vecs, Xtr, others


def make_clf(kind: str, p: dict):
    if kind == "lr":
        return LogisticRegression(C=p["C"], class_weight=p["class_weight"], solver="liblinear", max_iter=2000,
                                  random_state=SEED)
    return LinearSVC(C=p["C"], class_weight=p["class_weight"], random_state=SEED)


def main():
    ensure_dirs()
    set_seed(SEED)
    df = load_train()
    X_text, y = df["reviews"].values, df["sentiments"].values
    folds = get_folds(df)

    scores = {}  # (feature idx, clf idx) -> list of fold metrics
    t0 = time.time()
    for fi, fc in enumerate(FEATURE_GRID):
        for k, (tr, va) in enumerate(folds):
            _, Xtr, (Xva,) = featurise(fc, X_text[tr], X_text[va])
            for ci, (kind, p) in enumerate(CLF_GRID):
                clf = make_clf(kind, p).fit(Xtr, y[tr])
                pred = clf.predict(Xva)
                score = clf.predict_proba(Xva)[:, 1] if kind == "lr" else clf.decision_function(Xva)
                scores.setdefault((fi, ci), []).append(metrics(y[va], pred, score))
        print(f"feature config {fi + 1}/{len(FEATURE_GRID)} done ({time.time() - t0:.0f}s)")

    rows = []
    for (fi, ci), ms in scores.items():
        kind, p = CLF_GRID[ci]
        fc = FEATURE_GRID[fi]
        row = {"clf": kind, "C": p["C"], "class_weight": p["class_weight"] or "none",
               "ngram": f"{fc['ngram'][0]}-{fc['ngram'][1]}", "char_wb_2_5": fc["char"], "min_df": fc["min_df"]}
        for key in ("macro_f1", "accuracy", "f1_neg", "f1_pos", "recall_neg", "roc_auc"):
            v = np.array([m[key] for m in ms])
            row[f"{key}_mean"], row[f"{key}_std"] = round(v.mean(), 4), round(v.std(ddof=1), 4)
        row["_fi"], row["_ci"] = fi, ci
        rows.append(row)
    grid = pd.DataFrame(rows).sort_values("macro_f1_mean", ascending=False)
    grid.drop(columns=["_fi", "_ci"]).to_csv(RESULTS / "tfidf_cv_grid.csv", index=False)
    print(grid.drop(columns=["_fi", "_ci"]).head(10).to_string(index=False))

    # Best probability-capable config (LR) is Model A; best SVM is logged for comparison.
    for kind in ("lr", "svm"):
        best = grid[grid.clf == kind].iloc[0]
        fc, (k2, p) = FEATURE_GRID[int(best._fi)], CLF_GRID[int(best._ci)]
        cfg = {"clf": kind, **p, "ngram": list(fc["ngram"]), "char_wb_2_5": fc["char"], "min_df": fc["min_df"]}
        # Re-fit on the common hold-out fold for cross-model comparison and timing.
        tr, va = folds[HOLDOUT_FOLD]
        t1 = time.time()
        vecs, Xtr, (Xva,) = featurise(fc, X_text[tr], X_text[va])
        clf = make_clf(kind, p).fit(Xtr, y[tr])
        train_s = time.time() - t1
        t2 = time.time()
        Xva2 = sp.hstack([v.transform(X_text[va]) for v in vecs]).tocsr()
        score = clf.predict_proba(Xva2)[:, 1] if kind == "lr" else clf.decision_function(Xva2)
        pred = clf.predict(Xva2)
        infer_s = time.time() - t2
        m = metrics(y[va], pred, score)
        name = f"tfidf_{kind}"
        log_run(name, cfg, m, {"split": "holdout_fold0", "seed": SEED, "train_seconds": round(train_s, 2),
                               "infer_seconds_val": round(infer_s, 3), "n_features": Xtr.shape[1],
                               "cv5_macro_f1_mean": best.macro_f1_mean, "cv5_macro_f1_std": best.macro_f1_std})
        if kind == "lr":
            pd.DataFrame({"row_id": df["row_id"].values[va], "y_true": y[va],
                          "y_pred": pred, "p_pos": score}).to_csv(RESULTS / "val_preds_tfidf_lr.csv", index=False)
            (RESULTS / "best_tfidf_config.json").write_text(json.dumps(cfg, indent=2))
        print(f"\n{name} best cfg {cfg}\n  CV5 macro-F1 {best.macro_f1_mean:.4f} ± {best.macro_f1_std:.4f}"
              f" | hold-out macro-F1 {m['macro_f1']:.4f} acc {m['accuracy']:.4f} AUC {m['roc_auc']:.4f}"
              f" | {Xtr.shape[1]} features, train {train_s:.1f}s")

    # Majority-class baseline on the same hold-out fold (sanity reference).
    tr, va = folds[HOLDOUT_FOLD]
    log_run("majority_baseline", {"predict": 1}, metrics(y[va], np.ones(len(va), int), None),
            {"split": "holdout_fold0", "seed": SEED})


if __name__ == "__main__":
    main()
