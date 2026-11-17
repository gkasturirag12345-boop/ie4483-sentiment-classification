"""Data loading, leakage-safe splits, tokenisation, metrics and the attention BiLSTM used for the reported results."""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from conftest import CODE, ROOT, synthetic_rows, write_data


def test_import_creates_no_folders(tmp_path):
    out = tmp_path / "never_created"
    env = {**os.environ, "IE4483_RESULTS": str(out), "IE4483_DATA": str(tmp_path)}
    subprocess.run([sys.executable, "-c", "import common, evaluate, splits, data_loader"],
                   cwd=CODE / "Packages", env=env, check=True)
    assert not out.exists()


def test_loader_validates_and_explains_missing_data():
    import data_loader
    write_data(synthetic_rows())
    df = data_loader.load_train()
    assert list(df.columns) == ["row_id", "reviews", "sentiments"] and set(df["sentiments"]) == {0, 1}
    write_data([{"reviews": "ok", "sentiments": 2}])
    with pytest.raises(ValueError, match="labels outside"):
        data_loader.load_train()
    with pytest.raises(FileNotFoundError, match="not in the repository"):
        data_loader.load_test()


def test_near_duplicates_never_cross_train_validation():
    import data_loader
    import splits
    from preprocessing import group_key
    write_data(synthetic_rows())
    df = data_loader.load_train()
    keys = df["reviews"].map(group_key).values
    folds = splits.get_folds(df, n_splits=3, seed=0)
    for tr, va in folds:
        assert not set(keys[tr]) & set(keys[va])
        assert set(df["sentiments"].values[va]) == {0, 1}               # stratified: both classes in each fold
    assert np.array_equal(splits.get_folds(df, 3, 0)[0][1], folds[0][1])  # deterministic


def test_tokeniser_keeps_sentiment_cues():
    from preprocessing import group_key, tokenize
    assert tokenize("I DON'T like it!!") == ["i", "don't", "like", "it", "!", "!"]
    assert tokenize("Not&nbsp;good?") [-1] == "?" and "not" in tokenize("Not good")
    assert group_key("Great   PRODUCT!") == group_key("great product")


def test_metrics_on_known_predictions():
    from evaluate import metrics
    m = metrics([1, 1, 0, 0], [1, 0, 0, 0], [0.9, 0.4, 0.2, 0.1])
    assert m["accuracy"] == 0.75 and m["recall_neg"] == 1.0 and m["recall_pos"] == 0.5
    assert m["confusion_matrix"] == [[2, 0], [1, 1]] and m["roc_auc"] == 1.0


def _notebook_model():
    """BiLSTMAttn exactly as defined in the notebook that produced the reported results."""
    nb = json.loads((CODE / "IE4483_all_models.ipynb").read_text())
    src = next("".join(c["source"]) for c in nb["cells"]
               if c["cell_type"] == "code" and "class BiLSTMAttn" in "".join(c["source"]))
    import torch
    import torch.nn as nn
    from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
    ns = {"torch": torch, "nn": nn, "pack_padded_sequence": pack_padded_sequence,
          "pad_packed_sequence": pad_packed_sequence}
    exec(src, ns)
    return ns["BiLSTMAttn"]


def test_attention_bilstm_shapes_and_masking():
    torch = pytest.importorskip("torch")
    torch.manual_seed(0)
    model = _notebook_model()(vocab_size=50, emb_dim=16, hidden=8, dropout=0.0).eval()
    x = torch.randint(1, 50, (3, 10))
    lens = torch.tensor([10, 4, 1])
    x[1, 4:] = 0
    x[2, 1:] = 0
    logits, alpha = model(x, lens, return_attn=True)
    assert logits.shape == (3, 2) and alpha.shape == (3, 10)
    assert torch.allclose(alpha.sum(1), torch.ones(3))
    assert alpha[1, 4:].abs().max() < 1e-6 and alpha[2, 1:].abs().max() < 1e-6    # no attention on padding
    x2 = x.clone()
    x2[1, 4:] = 7                                                     # changing padding must not change the output
    assert torch.allclose(model(x2, lens), logits, atol=1e-6)


def test_committed_results_contain_no_review_text():
    for f in (ROOT / "results").glob("*.csv"):
        header = f.read_text().splitlines()[0].lower().split(",")
        assert "reviews" not in header and "text" not in header, f.name


def test_readme_table_matches_results():
    """Every number in the README table equals the committed result files (rounded to 4 places)."""
    import csv
    import re
    readme = (ROOT / "README.md").read_text().split("<!-- results-table -->")[1]
    comp = {r["model"]: r for r in csv.DictReader(open(ROOT / "results" / "model_comparison.csv"))}
    svm = next(r for r in csv.DictReader(open(ROOT / "results" / "training_log.csv"))
               if r["model"] == "tfidf_svm" and r["split"] == "holdout_fold0")
    src = {"TF-IDF + LR (default)": comp["TF-IDF + LR (default, Part 1)"],
           "TF-IDF + LR (tuned)": comp["TF-IDF + LR (tuned)"],
           "TF-IDF + linear SVM (tuned)": svm,
           "BiLSTM-attention, scratch (3 seeds)": comp["BiLSTM-attn A: scratch (3 seeds, mean ± std)"],
           "BiLSTM-attention, GloVe (3 seeds)": comp["BiLSTM-attn B: GloVe (3 seeds, mean ± std)"],
           "BiLSTM-attention, GloVe, 3-seed ensemble": comp["BiLSTM-attn B: GloVe (3-seed ensemble)"]}

    def fmt(v):
        return v if "±" in v else f"{float(v):.4f}"
    rows = [r for r in re.findall(r"^\| (.+?) \| (.+?) \| (.+?) \| (.+?) \|$", readme, re.M) if r[0] != "Model"]
    assert {r[0] for r in rows} == set(src)
    for name, acc, f1, rec in rows:
        s = src[name]
        assert (acc, f1, rec) == (fmt(s["accuracy"]), fmt(s["macro_f1"]), fmt(s["recall_neg"])), name
