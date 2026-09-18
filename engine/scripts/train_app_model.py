#!/usr/bin/env python3
"""Train the ONE speed model that ships inside the phone app.

Same recipe as `train_loro_models.py` -- fine-tuned from `speed_best.pt`, fixed 40 epochs,
no early stopping, seeded -- but on ALL 11 benchmark rides. Nothing recorded after 31 Aug
(the 11 Sept demo rides, every future ride) is used, so those stay honest test data.

--frame decides the input features:
  offline   infer_speed.features: centred windows, looks ahead. Laptop only.
  replica   the same transform with trailing windows -- what the phone computes (ImuFeatures.kt).
            Chosen for the app on 12 Sept (RESOURCES 8ao): 5.0% median, 10/11 phone-style.
  gravity   levelled by the gravity sensor, trailing.

Every accuracy number we quote still comes from leave-one-run-out models. This model is the
product, not the evidence.

    python scripts/train_app_model.py --frame replica    # -> checkpoints/app/speed_app_replica.pt
"""
from __future__ import annotations
import argparse, glob, os, sys, time
import numpy as np, torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dhruva.model import ResNet1D

EPOCHS = 40
SEED = 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", default="offline", choices=["offline", "replica", "gravity"])
    a = ap.parse_args()
    out = "checkpoints/app/speed_app.pt" if a.frame == "offline" else f"checkpoints/app/speed_app_{a.frame}.pt"

    runs = [r.rstrip("/") for r in sorted(glob.glob("rides/bike/*/")) + sorted(glob.glob("rides/aug31/*/"))]
    assert len(runs) == 11, f"expected the 11 benchmark rides, found {len(runs)}"
    base = torch.load("checkpoints/speed_best.pt", map_location="cpu", weights_only=False)
    W = int(base["window"]); mean, std = base["mean"], base["std"]
    if a.frame == "offline":
        from dhruva.finetune_bike import build
        data = [build(r + "/", W) for r in runs]
    else:
        import eval_causal_features as EC
        from train_eval_phone_features import frame_features, windows
        data = []
        for r in runs:
            t, acc, gyro, grav = EC.load(r)
            data.append(windows(r, frame_features(a.frame, acc, gyro, grav), t))
    X = (np.concatenate([d[0] for d in data]) - mean) / std
    Y = np.concatenate([d[1] for d in data])
    torch.manual_seed(SEED); np.random.seed(SEED)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    net = ResNet1D(in_ch=6, out_dim=1).to(dev)
    net.load_state_dict(base["model"])
    dl = DataLoader(TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(Y)), batch_size=64, shuffle=True, drop_last=True)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-4)
    lf = nn.SmoothL1Loss(); net.train()
    t0 = time.time()
    for ep in range(EPOCHS):
        tot = 0.0
        for xb, yb in dl:
            opt.zero_grad(); l = lf(net(xb.to(dev)), yb.to(dev)); l.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); tot += float(l)
        print(f"epoch {ep+1:2d}/{EPOCHS}  loss {tot/len(dl):.3f}  ({time.time()-t0:.0f} s)", flush=True)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    torch.save({"model": net.cpu().state_dict(), "window": W, "mean": mean, "std": std, "frame": a.frame,
                "trained_on": [os.path.basename(r) for r in runs], "epochs": EPOCHS, "seed": SEED}, out)
    print(f"saved {out}: {len(X)} windows from {len(runs)} rides, frame {a.frame}")


if __name__ == "__main__":
    main()
