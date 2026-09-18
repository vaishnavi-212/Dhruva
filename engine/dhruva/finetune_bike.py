#!/usr/bin/env python3
"""Fine-tune the A2 distance model on our own two-wheeler data.

Measured: the IO-VNBD-trained model transfers terribly to an Indian motorcycle
-- 3x speed overestimate, correlation ~0, distance error 155-310%. IO-VNBD's
mean speed is 13.7 m/s; the bike averaged 5.0 m/s, so the model has essentially
never seen this regime and falls back on its training prior.

This is contribution 6 (cross-region generalisation) measured for real, and the
fix is the obvious one: fine-tune on 16 minutes of our own labelled riding.

    python dhruva/finetune_bike.py --holdout 2
"""
from __future__ import annotations
import argparse, glob, os, sys, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dhruva.model import ResNet1D
from dhruva.infer_speed import load_imu, features, TRAIN_HZ

R = 6378137.0


def build(folder, window):
    t, acc, gyro = load_imu(folder)
    F = features(acc, gyro)
    L = pd.read_csv(os.path.join(folder, "Location.csv"))
    lt = L["seconds_elapsed"].to_numpy(float)
    lat, lon = L["latitude"].to_numpy(float), L["longitude"].to_numpy(float)
    x = (np.deg2rad(lon) - np.deg2rad(lon[0])) * np.cos(np.deg2rad(lat[0])) * R
    y = (np.deg2rad(lat) - np.deg2rad(lat[0])) * R
    # clean impossible GPS steps before they become distance LABELS (gpsclean)
    from dhruva.gpsclean import clean_for_truth
    _xy, _lt = clean_for_truth(np.column_stack([x, y]), lt)
    x, y, lt = _xy[:, 0], _xy[:, 1], _lt
    seg = np.r_[0, np.cumsum(np.hypot(np.diff(x), np.diff(y)))]
    s = np.interp(t, lt, seg)
    X, Y = [], []
    for i in range(0, len(F) - window, 2):
        d = s[i + window - 1] - s[i]
        if np.isfinite(d) and d >= 0:
            X.append(F[i:i + window]); Y.append(d)
    return np.stack(X).transpose(0, 2, 1).astype(np.float32), np.array(Y, np.float32)[:, None]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rides", default="rides/bike")
    ap.add_argument("--ckpt", default="checkpoints/speed_best.pt")
    ap.add_argument("--out", default="checkpoints/speed_bike.pt")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--holdout", type=int, default=2)
    a = ap.parse_args()

    c = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    W = int(c["window"]); mean, std = np.asarray(c["mean"]), np.asarray(c["std"])
    runs = sorted(glob.glob(os.path.join(a.rides, "*/")))
    print(f"{len(runs)} runs | window {W} ({W/TRAIN_HZ:.0f}s)")

    data = []
    for r in runs:
        X, Y = build(r, W)
        data.append((os.path.basename(r[:-1]), X, Y))
        print(f"  {os.path.basename(r[:-1]):32s} {len(X):5d} windows  "
              f"target {Y.mean():6.1f} +- {Y.std():5.1f} m")

    va = data[-a.holdout:]; tr = data[:-a.holdout]
    Xtr = np.concatenate([d[1] for d in tr]); Ytr = np.concatenate([d[2] for d in tr])
    Xva = np.concatenate([d[1] for d in va]); Yva = np.concatenate([d[2] for d in va])
    print(f"\ntrain {len(Xtr)} | val {len(Xva)} (held out: {[d[0] for d in va]})")

    Xtr = (Xtr - mean) / std; Xva = (Xva - mean) / std
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    net = ResNet1D(in_ch=6, out_dim=1).to(dev)
    net.load_state_dict(c["model"])

    base = float(np.sqrt(np.mean((Yva - Ytr.mean()) ** 2)))
    with torch.no_grad():
        net.eval()
        p0 = net(torch.tensor(Xva, dtype=torch.float32).to(dev)).cpu().numpy()
    print(f"\nbefore fine-tuning : val RMSE {np.sqrt(np.mean((p0-Yva)**2)):7.1f} m")
    print(f"predict-the-mean   : val RMSE {base:7.1f} m")

    dl = DataLoader(TensorDataset(torch.tensor(Xtr), torch.tensor(Ytr)),
                    batch_size=64, shuffle=True, drop_last=True)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    lossf = nn.SmoothL1Loss(); best = 1e9; hist = []
    for ep in range(1, a.epochs + 1):
        net.train()
        for xb, yb in dl:
            opt.zero_grad(); l = lossf(net(xb.to(dev)), yb.to(dev)); l.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
        net.eval()
        with torch.no_grad():
            p = net(torch.tensor(Xva, dtype=torch.float32).to(dev)).cpu().numpy()
        rmse = float(np.sqrt(np.mean((p - Yva) ** 2)))
        hist.append(rmse)
        if rmse < best:
            best = rmse
            torch.save({"model": net.state_dict(), "window": W, "mean": mean,
                        "std": std, "val_rmse": rmse}, a.out)
        if ep % 10 == 0 or ep == 1:
            print(f"  ep {ep:3d}  val RMSE {rmse:7.1f} m" + ("  <- best" if rmse == best else ""))
    print(f"\nafter fine-tuning  : val RMSE {best:7.1f} m   "
          f"({100*(1-best/base):.0f}% better than predict-the-mean)")
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
