"""Tests run without the course data: a small synthetic train.json stands in. Outputs go to a temp folder."""
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "Sentiments_Project_IE4483" / "Code"
_TMP = Path(tempfile.mkdtemp(prefix="ie4483-tests-"))
os.environ["IE4483_DATA"] = str(_TMP / "data")          # set before `common` is imported anywhere
os.environ["IE4483_RESULTS"] = str(_TMP / "results")
sys.path.insert(0, str(CODE / "Packages"))

POS = ["great product, works well", "love it!", "excellent quality and fast delivery", "very happy with this",
       "would buy again", "fantastic value", "it's perfect", "really good fit"]
NEG = ["broke after a day", "don't buy this", "terrible quality", "not worth the money", "never arrived",
       "very disappointed", "it doesn't work"]


def write_data(rows, name="train.json"):
    d = Path(os.environ["IE4483_DATA"])
    d.mkdir(parents=True, exist_ok=True)
    (d / name).write_text(json.dumps(rows))


def synthetic_rows(n_copies=3):
    """Positive and negative reviews, each repeated with case/whitespace changes (near-duplicates)."""
    rows = []
    for i in range(n_copies):
        for t in POS:
            rows.append({"reviews": t.upper() if i == 1 else t + "  " * i, "sentiments": 1})
        for t in NEG:
            rows.append({"reviews": t.upper() if i == 1 else t + "  " * i, "sentiments": 0})
    return rows
