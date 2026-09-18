#!/usr/bin/env python3
"""Train and CACHE one leave-one-run-out speed model per run.

Every honest evaluation needs a model that never saw the run being scored, and
retraining inside each evaluation script wastes ~40 minutes each time. Train
once, save to checkpoints/loro/, and let every evaluation load them.

Fixed epoch budget, no early stopping -- the held-out run must not influence the
model even through the choice of stopping point (RESOURCES.md 7u).
"""
from __future__ import annotations
import glob, os, sys
import numpy as np, torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dhruva.model import ResNet1D
from dhruva.finetune_bike import build

EPOCHS = 40
OUT = "checkpoints/loro"
SEED = 0          # without this, every run trains different models (RESOURCES.md 8i)

def main():
    runs = sorted(glob.glob("rides/bike/*/")) + sorted(glob.glob("rides/aug31/*/"))
    base = torch.load("checkpoints/speed_best.pt", map_location="cpu", weights_only=False)
    W = int(base["window"]); mean, std = base["mean"], base["std"]
    os.makedirs(OUT, exist_ok=True)
    data = [build(r, W) for r in runs]
    torch.manual_seed(SEED); np.random.seed(SEED)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    for i, r in enumerate(runs):
        torch.manual_seed(SEED + i)          # deterministic, and different per fold
        name = os.path.basename(r[:-1])
        path = os.path.join(OUT, name + ".pt")
        if os.path.exists(path):
            print(f"[{i+1}/{len(runs)}] {name}  cached"); continue
        Xtr = np.concatenate([data[j][0] for j in range(len(runs)) if j != i])
        Ytr = np.concatenate([data[j][1] for j in range(len(runs)) if j != i])
        Xtr = (Xtr - mean) / std
        net = ResNet1D(in_ch=6, out_dim=1).to(dev)
        net.load_state_dict(base["model"])
        dl = DataLoader(TensorDataset(torch.tensor(Xtr), torch.tensor(Ytr)),
                        batch_size=64, shuffle=True, drop_last=True)
        opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-4)
        lf = nn.SmoothL1Loss(); net.train()
        for _ in range(EPOCHS):
            for xb, yb in dl:
                opt.zero_grad(); l = lf(net(xb.to(dev)), yb.to(dev)); l.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
        torch.save({"model": net.cpu().state_dict(), "window": W,
                    "mean": mean, "std": std, "held_out": name}, path)
        print(f"[{i+1}/{len(runs)}] {name}  saved")

if __name__ == "__main__":
    main()
