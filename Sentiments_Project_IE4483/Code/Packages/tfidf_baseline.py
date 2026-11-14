"""Part 1 baseline: default TfidfVectorizer (exactly as in tfidf_features_example) + logistic regression.

test.json has no labels, so accuracy is measured on the common grouped hold-out fold of train.json
(plus 5-fold CV); the final model is refit on all of train.json to predict test.json.

Outputs
  results/submission_tfidf_baseline.csv   test predictions (reviews, predictions) in original order
  results/training_log.csv                 appended row "tfidf_baseline_default"
Usage:  python src/tfidf_baseline.py
"""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

from common import RESULTS, SEED, set_seed
from data_loader import load_test, load_train
from evaluate import log_run, metrics
from splits import get_folds, holdout


def fit_predict(train_text, y_train, other_text):
    """Fit default TF-IDF on training text only, train LR, return (vectorizer, clf, pred, p_pos) for other_text."""
    vectorizer = TfidfVectorizer().fit(train_text)
    clf = LogisticRegression(max_iter=1000, random_state=SEED).fit(vectorizer.transform(train_text), y_train)
    X_other = vectorizer.transform(other_text)
    return vectorizer, clf, clf.predict(X_other), clf.predict_proba(X_other)[:, 1]


def main():
    set_seed(SEED)
    df, test_df = load_train(), load_test()
    X_text, y = df["reviews"].values, df["sentiments"].values

    # Hold-out (fold 0, ~80/20, stratified and duplicate-grouped).
    tr, va = holdout(df)
    vec, _, pred, p_pos = fit_predict(X_text[tr], y[tr], X_text[va])
    m = metrics(y[va], pred, p_pos)
    print(f"Hold-out: train {len(tr)}  val {len(va)}  features {len(vec.vocabulary_)}")
    print(f"  accuracy {m['accuracy']:.4f}  macro-F1 {m['macro_f1']:.4f}  ROC-AUC {m['roc_auc']:.4f}")
    print(f"  majority-class baseline accuracy {np.mean(y[va] == 1):.4f}")
    print(f"  confusion matrix [[TN, FP], [FN, TP]] = {m['confusion_matrix']}")
    print(classification_report(y[va], pred, target_names=["negative", "positive"], digits=4))

    # 5-fold CV for a more stable estimate.
    accs, f1s = [], []
    for k, (ftr, fva) in enumerate(get_folds(df)):
        _, _, fpred, fp = fit_predict(X_text[ftr], y[ftr], X_text[fva])
        fm = metrics(y[fva], fpred, fp)
        accs.append(fm["accuracy"])
        f1s.append(fm["macro_f1"])
        print(f"  fold {k}: accuracy {fm['accuracy']:.4f}  macro-F1 {fm['macro_f1']:.4f}")
    print(f"5-fold CV accuracy {np.mean(accs):.4f} ± {np.std(accs, ddof=1):.4f}"
          f" | macro-F1 {np.mean(f1s):.4f} ± {np.std(f1s, ddof=1):.4f}")

    # Final model on all of train.json; test transformed with the same (train-fitted) vectorizer.
    vec, _, test_pred, _ = fit_predict(X_text, y, test_df["reviews"].values)
    print(f"\nFull train: X_train ({len(X_text)}, {len(vec.vocabulary_)}) | X_test ({len(test_df)}, {len(vec.vocabulary_)})")
    out = RESULTS / "submission_tfidf_baseline.csv"
    pd.DataFrame({"reviews": test_df["reviews"].values, "predictions": test_pred}).to_csv(out, index=False)
    print(f"Test predictions: {int((test_pred == 1).sum())} positive, {int((test_pred == 0).sum())} negative -> {out}")

    log_run("tfidf_baseline_default", {"vectorizer": "TfidfVectorizer()", "clf": "lr", "C": 1.0}, m,
            {"split": "holdout_fold0", "seed": SEED, "n_features": len(vec.vocabulary_),
             "cv5_accuracy_mean": round(float(np.mean(accs)), 4), "cv5_accuracy_std": round(float(np.std(accs, ddof=1)), 4),
             "cv5_macro_f1_mean": round(float(np.mean(f1s)), 4), "cv5_macro_f1_std": round(float(np.std(f1s, ddof=1)), 4)})


if __name__ == "__main__":
    main()
