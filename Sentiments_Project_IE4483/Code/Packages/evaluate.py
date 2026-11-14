"""Metrics and run logging shared by all training scripts."""
import csv
import json
import time

import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support, roc_auc_score)

from common import RESULTS


def metrics(y_true, y_pred, p_pos=None) -> dict:
    """Accuracy, per-class P/R/F1, macro-F1, confusion matrix and (if probabilities) ROC-AUC."""
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1], zero_division=0)
    m = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "precision_neg": p[0], "recall_neg": r[0], "f1_neg": f[0],
        "precision_pos": p[1], "recall_pos": r[1], "f1_pos": f[1],
        "roc_auc": roc_auc_score(y_true, p_pos) if p_pos is not None else np.nan,
    }
    m = {k: float(v) for k, v in m.items()}
    m["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    return m


def log_run(model: str, config: dict, m: dict, extra: dict | None = None, file: str = "training_log.csv"):
    """Append one run (config + metrics) to results/<file>."""
    row = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "model": model,
           "config": json.dumps(config, sort_keys=True)}
    row.update({k: (round(v, 5) if isinstance(v, float) else v) for k, v in m.items() if k != "confusion_matrix"})
    row["confusion_matrix"] = json.dumps(m.get("confusion_matrix"))
    row.update(extra or {})
    path = RESULTS / file
    keys = list(row)
    if path.exists():
        with open(path) as f:
            old_keys = next(csv.reader(f))
        keys = old_keys + [k for k in keys if k not in old_keys]
        rows = list(csv.DictReader(open(path)))
    else:
        rows = []
    rows.append(row)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
