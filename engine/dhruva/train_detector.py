#!/usr/bin/env python3
"""Train the landmark detector (contribution 3) — TCN-BiLSTM, ETLNet-style.

    python dhruva/train_detector.py --data data/detector --epochs 60

Three classes: none / bump / hump. Two reasons this is worth training rather
than thresholding:

  * RECALL drives the system pass rate. Measured (RESOURCES.md 7d): 80% recall
    gives 92% of blackouts under ISRO's 10%; 60% recall gives 62%. Recall is
    worth more here than precision, so the loss is weighted accordingly.
  * The TYPED alphabet (bump vs hump) cannot come from a threshold, and the
    types are what let the matcher disambiguate when drift is large.

Severe class imbalance is expected (~1% positive), so class weights are derived
from the data and recall is reported per class, never plain accuracy.
"""
from __future__ import annotations
import argparse, glob, json, os, sys, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CLASSES = ["none", "bump", "hump"]


class TCNBlock(nn.Module):
    def __init__(self, cin, cout, k=5, d=1):
        super().__init__()
        pad = (k - 1) * d
        self.net = nn.Sequential(
            nn.Conv1d(cin, cout, k, padding=pad, dilation=d), nn.BatchNorm1d(cout),
            nn.ReLU(inplace=True), nn.Dropout(0.15),
            nn.Conv1d(cout, cout, k, padding=pad, dilation=d), nn.BatchNorm1d(cout),
            nn.ReLU(inplace=True))
        self.trim = pad * 2
        self.down = nn.Conv1d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x):
        o = self.net(x)
        if self.trim: o = o[:, :, :-self.trim]
        return torch.relu(o + self.down(x))


class ETLNetish(nn.Module):
    """Two TCN layers into a BiLSTM — the architecture ETLNet reports 99.3% F1 with."""

    def __init__(self, in_ch=4, n_cls=3, base=48, lstm=64):
        super().__init__()
        self.t1 = TCNBlock(in_ch, base, d=1)
        self.t2 = TCNBlock(base, base, d=2)
        self.lstm = nn.LSTM(base, lstm, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(lstm * 2, 64),
                                  nn.ReLU(inplace=True), nn.Linear(64, n_cls))

    def forward(self, x):                      # (B, C, T)
        h = self.t2(self.t1(x)).transpose(1, 2)
        o, _ = self.lstm(h)
        return self.head(o.mean(dim=1))


def featurise(X):
    """(N,T,3) raw accel -> (N,4,T) yaw-invariant channels."""
    mag = np.linalg.norm(X, axis=2)
    mag = mag - np.median(mag, axis=1, keepdims=True)
    d1 = np.diff(mag, axis=1, prepend=mag[:, :1])
    vert = X[:, :, 2] - np.median(X[:, :, 2], axis=1, keepdims=True)
    horiz = np.linalg.norm(X[:, :, :2], axis=2)
    horiz = horiz - np.median(horiz, axis=1, keepdims=True)
    return np.stack([mag, d1, vert, horiz], axis=1).astype(np.float32)


def load(data_dir):
    Xs, ys = [], []
    for f in sorted(glob.glob(os.path.join(data_dir, "*.npz"))):
        z = np.load(f, allow_pickle=True)
        ty = z["types"]
        lab = np.array([CLASSES.index(t) if t in CLASSES else 0 for t in ty])
        Xs.append(z["X"]); ys.append(lab)
        print(f"  {os.path.basename(f):44s} {len(z['X']):6d} windows, "
              f"{int((lab>0).sum()):4d} positive")
    if not Xs:
        sys.exit(f"no .npz in {data_dir} -- run scripts/label_ride.py first")
    return np.concatenate(Xs), np.concatenate(ys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/detector")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-frac", type=float, default=0.25)
    ap.add_argument("--out", default="checkpoints/detector")
    a = ap.parse_args()

    print("loading labelled windows")
    X, y = load(a.data)
    F = featurise(X)
    print(f"\ntotal {len(F)} windows | positives {int((y>0).sum())} "
          f"({(y>0).mean()*100:.2f}%) | classes {np.bincount(y, minlength=3)}")
    if (y > 0).sum() < 40:
        print("!! fewer than 40 positive examples -- expect the model to memorise.")
        print("   More laps is the cheapest fix: each lap adds ~14 positives.")

    # split by TIME (last chunk held out) so windows don't leak across the split
    n = len(F); cut = int(n * (1 - a.val_frac))
    Xtr, ytr, Xva, yva = F[:cut], y[:cut], F[cut:], y[cut:]

    # Inverse-frequency weights, but SQRT-damped and floored. Raw inverse
    # frequency drove the 'none' weight to ~0.002, so false positives cost
    # nothing and precision collapsed to 0.11. Sqrt damping keeps the recall
    # bias without making the negative class free.
    cnt = np.bincount(ytr, minlength=3).astype(float)
    w = np.sqrt(cnt.sum() / np.maximum(cnt, 1))
    w = w / w.max()
    w = np.maximum(w, 0.15)                            # floor: negatives still cost
    w = w * np.array([1.0, 2.0, 2.0])                  # mild bias toward RECALL
    print(f"class weights (recall-biased): {np.round(w,2)}")

    dev = ("mps" if torch.backends.mps.is_available()
           else "cuda" if torch.cuda.is_available() else "cpu")
    tr = DataLoader(TensorDataset(torch.tensor(Xtr), torch.tensor(ytr)),
                    batch_size=a.batch, shuffle=True, drop_last=len(Xtr) > a.batch)
    va = DataLoader(TensorDataset(torch.tensor(Xva), torch.tensor(yva)), batch_size=256)

    net = ETLNetish(in_ch=F.shape[1]).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32).to(dev))
    os.makedirs(a.out, exist_ok=True)
    best, hist = -1.0, []

    for ep in range(1, a.epochs + 1):
        net.train(); tl = k = 0; t0 = time.time()
        for xb, yb in tr:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad(); l = lossf(net(xb), yb); l.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
            tl += l.item() * len(xb); k += len(xb)
        net.eval(); P = []; Y = []
        with torch.no_grad():
            for xb, yb in va:
                P.append(net(xb.to(dev)).argmax(1).cpu().numpy()); Y.append(yb.numpy())
        P, Y = np.concatenate(P), np.concatenate(Y)
        det_rec = ((P > 0) & (Y > 0)).sum() / max((Y > 0).sum(), 1)
        det_pre = ((P > 0) & (Y > 0)).sum() / max((P > 0).sum(), 1)
        f1 = 2 * det_pre * det_rec / max(det_pre + det_rec, 1e-9)
        hist.append(dict(epoch=ep, loss=tl / max(k, 1), recall=float(det_rec),
                         precision=float(det_pre), f1=float(f1)))
        flag = ""
        if det_rec > best:
            best = det_rec
            torch.save({"model": net.state_dict(), "classes": CLASSES}, f"{a.out}/detector_best.pt")
            flag = "  <- best recall"
        print(f"ep {ep:3d}/{a.epochs} loss {tl/max(k,1):.4f} | recall {det_rec:.3f} "
              f"precision {det_pre:.3f} F1 {f1:.3f} ({time.time()-t0:.0f}s){flag}")

    json.dump(hist, open(f"{a.out}/history.json", "w"), indent=1)
    print(f"\nbest detection recall {best:.3f} -> {a.out}/detector_best.pt")
    print("Recall is the number that matters: 80% -> 92% system pass, 60% -> 62%.")


if __name__ == "__main__":
    main()
