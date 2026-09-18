#!/usr/bin/env python3
"""Retrain the leave-one-run-out speed models in a frame the PHONE can compute, then score
them the way the phone will run them (look-back features, speed from the window that just ended).

Why (results/causal_features.csv): the offline models were fine-tuned in a frame built from
CENTRED 8 s averages of gravity-free acceleration. A phone can only look back, and the trailing
version of that frame correlates just 0.43 with the centred one: 6.7% -> 8.0% median, 9 -> 6 of 11.
Imitating the offline frame is the wrong fix; training in the phone's own frame is the right one.

Two candidate frames, identical recipe to train_loro_models.py (from speed_best.pt, 40 fixed
epochs, AdamW 3e-4, batch 64, SmoothL1, seeded per fold):
  replica   the training transform with TRAILING windows (what the phone can reproduce exactly)
  gravity   levelled by the phone's own gravity sensor, trailing 8 s (physical; what IO-VNBD
            pre-training used)

Models -> checkpoints/loro_phone_<frame>/   Results -> results/phone_features_retrained.csv

    python scripts/train_eval_phone_features.py
"""
from __future__ import annotations
import glob, os, sys, time
import numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_causal_features as EC
import dhruva.finetune_bike as FB
from dhruva.model import ResNet1D
from dhruva.osm_road import OsmNetwork, road_from_osm, to_xy
from dhruva.mapmatch import RoadPolyline
from dhruva.gpsclean import clean_for_truth
from evaluate_system import _extend

EPOCHS, SEED, W = 40, 0, 300


def frame_features(kind, acc, gyro, grav):
    if kind == "replica":
        sm = EC.trailing(acc, EC.N8); down = sm / np.maximum(np.linalg.norm(sm, axis=1, keepdims=True), 1e-6)
        return EC.feats(EC.basis(EC.trailing(acc + down * EC.G, EC.N8)), gyro, acc)
    if kind == "gravity":
        return EC.feats(EC.basis(EC.trailing(grav, EC.N8)), gyro, acc)
    raise ValueError(kind)


def windows(folder, F, t):
    """finetune_bike.build, with the features passed in."""
    L = pd.read_csv(os.path.join(folder, "Location.csv")); lt = L["seconds_elapsed"].to_numpy(float)
    lat, lon = L["latitude"].to_numpy(float), L["longitude"].to_numpy(float)
    x = (np.deg2rad(lon) - np.deg2rad(lon[0])) * np.cos(np.deg2rad(lat[0])) * FB.R
    y = (np.deg2rad(lat) - np.deg2rad(lat[0])) * FB.R
    _xy, _lt = clean_for_truth(np.column_stack([x, y]), lt)
    seg = np.r_[0, np.cumsum(np.hypot(np.diff(_xy[:, 0]), np.diff(_xy[:, 1])))]
    s = np.interp(t, _lt, seg)
    X, Y = [], []
    for i in range(0, len(F) - W, 2):
        d = s[i + W - 1] - s[i]
        if np.isfinite(d) and d >= 0:
            X.append(F[i:i + W]); Y.append(d)
    return np.stack(X).transpose(0, 2, 1).astype(np.float32), np.array(Y, np.float32)[:, None]


def main():
    base = torch.load("checkpoints/speed_best.pt", map_location="cpu", weights_only=False)
    mean, std = base["mean"], base["std"]
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    bench = [r.rstrip("/") for r in sorted(glob.glob("rides/bike/*/")) + sorted(glob.glob("rides/aug31/*/"))]
    evening = [("rides/app_DhruvaRun_2026-09-11_19-37-43", 55.0, 172.0), ("rides/app_DhruvaRun_2026-09-11_19-41-28", 48.0, 137.0)]
    raw = {f: EC.load(f) for f in bench + [e[0] for e in evening]}
    S = pd.read_csv("data/landmarks/campus_landmarks.csv"); lat0, lon0 = float(S.lat.iloc[0]), float(S.lon.iloc[0])
    net_osm = OsmNetwork("data/osm/wide_ways_v2.json", lat0, lon0)
    rows = []
    for kind in ("replica", "gravity"):
        out_dir = f"checkpoints/loro_phone_{kind}"; os.makedirs(out_dir, exist_ok=True)
        F = {f: frame_features(kind, *raw[f][1:]) for f in raw}
        data = {f: windows(f, F[f], raw[f][0]) for f in bench}
        nets = {}
        for i, held in enumerate(bench):
            name = os.path.basename(held); path = os.path.join(out_dir, name + ".pt")
            if not os.path.exists(path):
                t0 = time.time(); torch.manual_seed(SEED + i); np.random.seed(SEED + i)
                Xtr = (np.concatenate([data[f][0] for f in bench if f != held]) - mean) / std
                Ytr = np.concatenate([data[f][1] for f in bench if f != held])
                net = ResNet1D(in_ch=6, out_dim=1).to(dev); net.load_state_dict(base["model"])
                dl = DataLoader(TensorDataset(torch.tensor(Xtr, dtype=torch.float32), torch.tensor(Ytr)), batch_size=64, shuffle=True, drop_last=True)
                opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-4); lf = nn.SmoothL1Loss(); net.train()
                for _ in range(EPOCHS):
                    for xb, yb in dl:
                        opt.zero_grad(); l = lf(net(xb.to(dev)), yb.to(dev)); l.backward()
                        nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
                torch.save({"model": net.cpu().state_dict(), "window": W, "mean": mean, "std": std, "held_out": name, "frame": kind}, path)
                print(f"[{kind}] [{i+1}/11] {name} trained in {time.time()-t0:.0f} s", flush=True)
            ck = torch.load(path, map_location="cpu", weights_only=False)
            n = ResNet1D(in_ch=6, out_dim=1); n.load_state_dict(ck["model"]); n.eval(); nets[name] = (n, np.asarray(mean), np.asarray(std))
        jobs = [(f, 60.0, None, [os.path.basename(f)]) for f in bench] + [(f, c, e, list(nets)) for f, c, e in evening]
        for folder, cut, fin, use in jobs:
            t = raw[folder][0]
            L = pd.read_csv(os.path.join(folder, "Location.csv")); lt = L.seconds_elapsed.to_numpy(float)
            gps = to_xy(L.latitude.to_numpy(float), L.longitude.to_numpy(float), lat0, lon0); gps, lt = clean_for_truth(gps, lt)
            road = RoadPolyline(_extend(np.asarray(road_from_osm(net_osm, gps)[0], float), 3000.0), 2.0)
            keep = (t >= lt[0]) & (t <= lt[-1])
            truth = np.column_stack([np.interp(t, lt, gps[:, 0]), np.interp(t, lt, gps[:, 1])])
            cum = np.r_[0, np.cumsum(np.linalg.norm(np.diff(truth, axis=0), axis=1))]
            i0 = int(np.searchsorted(t, cut)); i1 = int(np.searchsorted(t, fin)) + 1 if fin else int(np.flatnonzero(keep)[-1]) + 1
            d = [EC.score(t, EC.speeds(F[folder], t, *nets[m], causal=True), truth, cum, road, i0, i1) for m in use]
            rows.append(dict(frame=kind, ride=os.path.basename(folder), kind="evening" if fin else "bench",
                             drift=float(np.median(d)), passes=int(np.sum(np.array(d) < 10)), n=len(d)))
            print(f"[{kind}] {os.path.basename(folder)[:34]:34s} {rows[-1]['drift']:5.1f}%  ({rows[-1]['passes']}/{len(d)})", flush=True)
    D = pd.DataFrame(rows); D.to_csv("results/phone_features_retrained.csv", index=False)
    print("\nreference, offline models (results/causal_features.csv): offline 6.7% 9/11 | phone as-is 8.0% 6/11 | 19-37 9.7% | 19-41 6.8%")
    for kind in D.frame.unique():
        b = D[(D.frame == kind) & (D.kind == "bench")]; e = D[(D.frame == kind) & (D.kind == "evening")].reset_index()
        print(f"retrained in {kind:8s} frame, phone-style: bench {b.drift.median():5.1f}% {int((b.drift < 10).sum())}/11 | "
              f"19-37 {e.drift[0]:5.1f}% {e.passes[0]}/11 | 19-41 {e.drift[1]:5.1f}% {e.passes[1]}/11")
    print("saved results/phone_features_retrained.csv")


if __name__ == "__main__":
    main()
