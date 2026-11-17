"""Model B1: word-embedding + BiLSTM classifier trained from scratch (PyTorch).

Architecture: Embedding(V, 128) -> BiLSTM(hidden H per direction) -> [masked max-pool ; masked mean-pool] (4H)
              -> Dropout(0.3) -> Linear(4H, 2).  Loss: class-weighted CrossEntropy.  Optimiser: Adam.
Selection: small sweep (lr x hidden) on the hold-out fold with seed 42, then 3 seeds for the best config.
Usage:  python src/train_bilstm.py
"""
import collections
import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from common import RESULTS, SEED, SEEDS, get_device, set_seed, ensure_dirs
from data_loader import load_train
from evaluate import log_run, metrics
from preprocessing import tokenize
from splits import holdout

MAX_LEN, EMB, BATCH, MAX_EPOCHS, PATIENCE, MIN_FREQ, DROPOUT = 256, 128, 64, 15, 3, 2, 0.3
SWEEP = [dict(lr=lr, hidden=h) for lr in (1e-3, 2e-3) for h in (64, 128)]


def build_vocab(texts, min_freq=MIN_FREQ) -> dict:
    """Vocabulary from TRAINING texts only. 0 = <pad>, 1 = <unk>."""
    c = collections.Counter(t for s in texts for t in tokenize(s))
    vocab = {"<pad>": 0, "<unk>": 1}
    for w, n in c.most_common():
        if n >= min_freq:
            vocab[w] = len(vocab)
    return vocab


def encode(texts, vocab):
    ids = [[vocab.get(t, 1) for t in tokenize(s)][:MAX_LEN] or [1] for s in texts]  # empty text -> one <unk>
    lens = torch.tensor([len(x) for x in ids])
    out = torch.zeros(len(ids), MAX_LEN, dtype=torch.long)
    for i, x in enumerate(ids):
        out[i, :len(x)] = torch.tensor(x)
    return out, lens


class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, emb=EMB, hidden=128, dropout=DROPOUT):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb, padding_idx=0)
        self.lstm = nn.LSTM(emb, hidden, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(4 * hidden, 2)

    def forward(self, x, lens):
        e = self.drop(self.emb(x))
        packed = pack_padded_sequence(e, lens.cpu(), batch_first=True, enforce_sorted=False)
        h, _ = pad_packed_sequence(self.lstm(packed)[0], batch_first=True)    # (B, T, 2H)
        mask = (torch.arange(h.size(1), device=h.device)[None, :] < lens.to(h.device)[:, None]).unsqueeze(-1)
        mx = h.masked_fill(~mask, -1e9).max(1).values
        mean = (h * mask).sum(1) / lens.to(h.device).unsqueeze(1)
        return self.head(self.drop(torch.cat([mx, mean], 1)))                  # logits (B, 2)


def predict_proba(model, X, L, device):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), 256):
            out.append(torch.softmax(model(X[i:i + 256].to(device), L[i:i + 256]), 1)[:, 1].cpu())
    return torch.cat(out).numpy()


def train(tr_text, tr_y, va_text=None, va_y=None, lr=1e-3, hidden=128, seed=SEED, epochs=MAX_EPOCHS, verbose=True):
    """Train one model. With validation data: early stopping on val macro-F1, best checkpoint restored.
    Without validation data: train for a fixed number of `epochs` (used for the final refit)."""
    set_seed(seed)
    device = get_device()
    vocab = build_vocab(tr_text)
    Xtr, Ltr = encode(tr_text, vocab)
    ytr = torch.tensor(tr_y)
    model = BiLSTMClassifier(len(vocab), hidden=hidden).to(device)
    w = torch.tensor(len(tr_y) / (2 * np.bincount(tr_y, minlength=2)), dtype=torch.float32)  # 'balanced' weights
    loss_fn = nn.CrossEntropyLoss(weight=w.to(device))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    if va_text is not None:
        Xva, Lva = encode(va_text, vocab)
    g = torch.Generator().manual_seed(seed)
    best, best_state, best_ep, bad, history = -1, None, 0, 0, []
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(Xtr), generator=g)
        tot = 0.0
        for i in range(0, len(perm), BATCH):
            idx = perm[i:i + BATCH]
            opt.zero_grad()
            loss = loss_fn(model(Xtr[idx].to(device), Ltr[idx]), ytr[idx].to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(idx)
        rec = {"epoch": ep, "train_loss": tot / len(Xtr)}
        if va_text is not None:
            p = predict_proba(model, Xva, Lva, device)
            m = metrics(va_y, (p >= 0.5).astype(int), p)
            rec.update(val_macro_f1=m["macro_f1"], val_acc=m["accuracy"])
            if m["macro_f1"] > best:
                best, best_ep, bad = m["macro_f1"], ep, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
        history.append(rec)
        if verbose:
            print("   ", {k: round(v, 4) if isinstance(v, float) else v for k, v in rec.items()})
        if va_text is not None and bad >= PATIENCE:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, vocab, best_ep, history


def main():
    ensure_dirs()
    df = load_train()
    tr, va = holdout(df)
    X, y = df["reviews"].values, df["sentiments"].values
    device = get_device()
    print("device:", device)

    def run(cfg, seed):
        t0 = time.time()
        model, vocab, best_ep, hist = train(X[tr], y[tr], X[va], y[va], seed=seed, **cfg)
        train_s = time.time() - t0
        Xva, Lva = encode(X[va], vocab)
        t1 = time.time()
        p = predict_proba(model, Xva, Lva, device)
        infer_s = time.time() - t1
        m = metrics(y[va], (p >= 0.5).astype(int), p)
        n_params = sum(q.numel() for q in model.parameters())
        full = {**cfg, "emb": EMB, "max_len": MAX_LEN, "batch": BATCH, "dropout": DROPOUT, "min_freq": MIN_FREQ,
                "vocab_size": len(vocab), "optimizer": "Adam", "loss": "weighted CE"}
        log_run("bilstm", full, m, {"split": "holdout_fold0", "seed": seed, "best_epoch": best_ep,
                                    "train_seconds": round(train_s, 1), "infer_seconds_val": round(infer_s, 3),
                                    "n_params": n_params})
        log_run("bilstm", full, m, {"seed": seed, "best_epoch": best_ep, "history": json.dumps(hist)},
                file="bilstm_runs.csv")
        print(f"  cfg {cfg} seed {seed}: macro-F1 {m['macro_f1']:.4f} acc {m['accuracy']:.4f} "
              f"AUC {m['roc_auc']:.4f} best_epoch {best_ep} ({train_s:.0f}s)")
        return m, p, best_ep

    sweep = []
    for cfg in SWEEP:
        print("sweep", cfg)
        m, _, ep = run(cfg, SEED)
        sweep.append((m["macro_f1"], cfg))
    best_cfg = max(sweep, key=lambda t: t[0])[1]
    print("best BiLSTM config:", best_cfg)

    res = {}
    for seed in SEEDS:
        if seed == SEED:  # already trained in the sweep; retrain to capture predictions (deterministic seed)
            pass
        res[seed] = run(best_cfg, seed)
    f1s = [res[s][0]["macro_f1"] for s in SEEDS]
    eps = [res[s][2] for s in SEEDS]
    summary = {"config": best_cfg, "seeds": SEEDS, "macro_f1_mean": float(np.mean(f1s)),
               "macro_f1_std": float(np.std(f1s, ddof=1)), "best_epochs": eps,
               "final_epochs": int(round(np.median(eps)))}
    (RESULTS / "best_bilstm_config.json").write_text(json.dumps(summary, indent=2))
    m, p, _ = res[SEED]
    pd.DataFrame({"row_id": df["row_id"].values[va], "y_true": y[va],
                  "y_pred": (p >= 0.5).astype(int), "p_pos": p}).to_csv(RESULTS / "val_preds_bilstm.csv", index=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
