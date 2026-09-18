#!/usr/bin/env python3
"""A2 — train the speed-regression network on IO-VNBD.

    python dhruva/train_speed.py --epochs 60

Held-out file for validation, so the score is on data the model never saw.
Writes checkpoints/speed_best.pt + a training curve.
"""
from __future__ import annotations
import argparse, glob, json, os, sys, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dhruva.dataset_iovnbd import WindowSet
from dhruva.model import ResNet1D


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--raw6", action="store_true", help="use old yaw-dependent features")
    ap.add_argument("--only", default="", help="json list of filenames to restrict to")
    ap.add_argument("--window", type=int, default=300)
    ap.add_argument("--step", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--holdout", default="S-S2.csv,S-S3a.csv,S-Vta1a.csv,S-Vw2.csv,S-Y1.csv",
                    help="comma-separated validation files")
    ap.add_argument("--out", default="checkpoints")
    a = ap.parse_args()

    files = sorted(f for f in glob.glob(os.path.join(a.data, "S-*.csv")))
    if a.only:
        keep = set(json.load(open(a.only)))
        files = [f for f in files if os.path.basename(f) in keep]
        print(f"restricted to {len(files)} files from {a.only}")
    if not files:
        sys.exit("no S-*.csv in data/ -- run scripts/fetch_iovnbd.py first")
    hold = {h.strip() for h in a.holdout.split(",") if h.strip()}
    val_files = [f for f in files if os.path.basename(f) in hold]
    train_files = [f for f in files if f not in val_files]
    if not val_files:
        val_files, train_files = files[-3:], files[:-3]
    print(f"train: {len(train_files)} files")
    print(f"val  : {[os.path.basename(f) for f in val_files]}\n")

    print("building train windows")
    tr = WindowSet(train_files, a.window, a.step, verbose=(len(train_files) <= 10), invariant=not a.raw6)
    print("building val windows")
    va = WindowSet(val_files, a.window, a.step, stats=tr.stats, verbose=True, invariant=not a.raw6)
    print(f"\ntrain {len(tr)} windows | val {len(va)} windows | "
          f"target distance mean {tr.y.mean():.1f} std {tr.y.std():.1f} m")

    dev = ("mps" if torch.backends.mps.is_available()
           else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {dev}\n")

    dl_tr = DataLoader(tr, batch_size=a.batch, shuffle=True, drop_last=True)
    dl_va = DataLoader(va, batch_size=256)

    net = ResNet1D(in_ch=6, out_dim=1).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=a.lr, total_steps=a.epochs * max(len(dl_tr), 1))
    lossf = nn.SmoothL1Loss()

    os.makedirs(a.out, exist_ok=True)
    hist, best = [], 1e9
    # baseline: always predict the training mean
    base = float(np.sqrt(np.mean((va.y - tr.y.mean()) ** 2)))
    print(f"baseline RMSE (predict-the-mean): {base:.2f} m\n")

    for ep in range(1, a.epochs + 1):
        net.train(); tl = n = 0
        t0 = time.time()
        for x, y in dl_tr:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            loss = lossf(net(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step(); sched.step()
            tl += loss.item() * len(x); n += len(x)
        net.eval(); se = m = 0
        with torch.no_grad():
            for x, y in dl_va:
                p = net(x.to(dev)).cpu()
                se += float(((p - y) ** 2).sum()); m += len(y)
        rmse = (se / m) ** 0.5
        hist.append({"epoch": ep, "train_loss": tl / n, "val_rmse": rmse})
        flag = ""
        if rmse < best:
            best = rmse
            torch.save({"model": net.state_dict(), "window": a.window,
                        "mean": tr.mean, "std": tr.std, "val_rmse": rmse},
                       os.path.join(a.out, "speed_best.pt"))
            flag = "  <- best"
        print(f"ep {ep:3d}/{a.epochs}  train {tl/n:.4f}  val RMSE {rmse:.2f} m"
              f"  ({time.time()-t0:.0f}s){flag}")

    json.dump({"history": hist, "baseline_rmse": base, "best_rmse": best},
              open(os.path.join(a.out, "speed_history.json"), "w"), indent=1)
    print(f"\nbest val RMSE {best:.2f} m  vs baseline {base:.2f} m"
          f"  ({100*(1-best/base):.1f}% better)")
    print(f"-> {a.out}/speed_best.pt")


if __name__ == "__main__":
    main()
