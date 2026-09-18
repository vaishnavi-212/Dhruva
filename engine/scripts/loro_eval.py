#!/usr/bin/env python3
"""Honest leave-one-run-out evaluation of requirement 8.

`checkpoints/speed_bike.pt` cannot be used to claim a benchmark number: it was
fine-tuned on runs 1-4 and its epoch was SELECTED by validation RMSE on runs 5-6,
so every run is either training data or selection data.

Here: for each run, fine-tune on the other five for a FIXED epoch budget (no
early stopping, so the held-out run never influences the model), then evaluate.
"""
from __future__ import annotations
import glob, os, sys
import numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dhruva.model import ResNet1D
from dhruva.infer_speed import load_imu, features, TRAIN_HZ
from dhruva.finetune_bike import build
from dhruva.osm_road import OsmNetwork, road_from_osm, to_xy
from dhruva.mapmatch import RoadPolyline
from harness import metrics
from dhruva.gpsclean import clean_for_truth
from dhruva import landmarks, landmark_update
from evaluate_system import _extend, map_landmark_arcs

EPOCHS = 40
SEED = 0
USE_CACHE = os.environ.get('CACHE', '1') == '1'   # cached models = reproducible
USE_LANDMARKS = os.environ.get('LANDMARKS', '1') == '1'
CONST_SPEED = float(os.environ.get('CONST', '0'))   # >0 replaces the model with a constant

def main():
    M = pd.read_csv("data/landmarks/campus_landmarks.csv")
    lat0, lon0 = M.lat.iloc[0], M.lon.iloc[0]
    net_osm = OsmNetwork(os.environ.get("OSM", "data/osm/wide_ways.json"), lat0, lon0)
    base = torch.load("checkpoints/speed_best.pt", map_location="cpu", weights_only=False)
    W = int(base["window"]); mean, std = base["mean"], base["std"]
    runs = sorted(glob.glob("rides/bike/*/")) + sorted(glob.glob("rides/aug31/*/"))
    data = [build(r, W) for r in runs]
    dev = "mps" if torch.backends.mps.is_available() else "cpu"

    print(f"LORO, {EPOCHS} fixed epochs, no early stopping | landmarks {'ON' if USE_LANDMARKS else 'OFF'}\n")
    print(f"{'held-out run':30s} {'dist':>6} {'drift m':>8} {'drift %':>8} {'pass':>5}")
    print("-" * 64)
    out = []
    for i, r in enumerate(runs):
        cached = os.path.join("checkpoints/loro", os.path.basename(r[:-1]) + ".pt")
        if USE_CACHE and os.path.exists(cached):
            # Reproducible path: load the model trained once by train_loro_models.py.
            # Training here instead makes the headline number depend on the shuffle
            # order -- measured 1.7% vs 7.2% median across two runs (RESOURCES.md 8i).
            co = torch.load(cached, map_location="cpu", weights_only=False)
            net = ResNet1D(in_ch=6, out_dim=1).to(dev)
            net.load_state_dict(co["model"])
            net.eval()
        else:
            torch.manual_seed(SEED + i)
            np.random.seed(SEED + i)
            Xtr = np.concatenate([data[j][0] for j in range(len(runs)) if j != i])
            Ytr = np.concatenate([data[j][1] for j in range(len(runs)) if j != i])
            Xtr = (Xtr - mean) / std
            net = ResNet1D(in_ch=6, out_dim=1).to(dev)
            net.load_state_dict(base["model"])
            dl = DataLoader(TensorDataset(torch.tensor(Xtr), torch.tensor(Ytr)),
                            batch_size=64, shuffle=True, drop_last=True)
            opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-4)
            lf = nn.SmoothL1Loss()
            net.train()
            for _ in range(EPOCHS):
                for xb, yb in dl:
                    opt.zero_grad(); l = lf(net(xb.to(dev)), yb.to(dev)); l.backward()
                    nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
            net.eval()


        L = pd.read_csv(os.path.join(r, "Location.csv"))
        lt = L["seconds_elapsed"].to_numpy(float)
        gps = to_xy(L["latitude"].to_numpy(float), L["longitude"].to_numpy(float), lat0, lon0)
        gps, lt = clean_for_truth(gps, lt)      # GPS spikes are not ground truth
        osm, _ = road_from_osm(net_osm, gps)
        road = RoadPolyline(_extend(osm, 400.0), 2.0)
        t, acc, gyro = load_imu(r); F = features(acc, gyro)
        idx = np.arange(0, len(F) - W, 1)
        X = ((np.stack([F[k:k+W] for k in idx]).transpose(0, 2, 1)) - mean) / std
        p = []
        with torch.no_grad():
            for k in range(0, len(X), 256):
                p.append(net(torch.tensor(X[k:k+256], dtype=torch.float32).to(dev)).cpu().numpy())
        v = np.concatenate(p).ravel() / (W / TRAIN_HZ)
        v = np.interp(t, t[idx + W // 2], v)
        if CONST_SPEED > 0:
            v = np.full(len(t), CONST_SPEED)   # ablation: is the learned model earning its place?
        dt = float(np.median(np.diff(t)))
        s0 = road.project(gps[0])
        raw = np.cumsum(np.clip(v, 0, None)) * dt
        arc = s0 + raw
        if USE_LANDMARKS:
            A = pd.read_csv(os.path.join(r, "Accelerometer.csv"))
            ta = A["seconds_elapsed"].to_numpy(float)
            ev = landmarks.detect(A[["x", "y", "z"]].to_numpy(float), ta,
                                  1.0 / np.median(np.diff(ta)))
            ev = landmark_update.merge_double_fires(ev, t, raw)
            ms = map_landmark_arcs(road, M.lat, M.lon, lat0, lon0)
            arc, _st = landmark_update.correct(arc, t, ev, ms)
        pos, _ = road.at(arc)
        truth = np.column_stack([np.interp(t, lt, gps[:, 0]), np.interp(t, lt, gps[:, 1])])
        s = metrics.score(pos, truth, t); out.append(s)
        print(f"{os.path.basename(r[:-1])[:30]:30s} {s.distance_m:6.0f} {s.final_drift_m:8.1f} "
              f"{s.drift_pct:8.1f} {'YES' if s.passes_isro else 'NO':>5}")
    d = np.array([x.drift_pct for x in out])
    print("-" * 64)
    print(f"median {np.median(d):.1f}%  mean {d.mean():.1f}%  worst {d.max():.1f}%  "
          f"pass {100*np.mean([x.passes_isro for x in out]):.0f}%   (ISRO limit 10%)")

if __name__ == "__main__":
    main()
