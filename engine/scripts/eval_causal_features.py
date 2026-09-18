#!/usr/bin/env python3
"""Which LOOK-BACK-ONLY feature pipeline should the phone run?

The offline pipeline (`infer_speed.features`) levels the phone with CENTRED 8 s averages and
assigns each window's speed to the window's centre -- both look up to 15 s into the future.
A phone cannot. This compares, on recorded rides, the offline answer with three causal ones:

  V1 replica   the training transform exactly, with trailing instead of centred windows.
               Note what the training transform is: our recordings are gravity-FREE, so
               "down" is rebuilt from the 8 s mean of linear acceleration, not true gravity.
  V2 gravity   the physically correct level frame from the phone's gravity sensor (trailing 8 s)
  V3 edge      dhruva/edge.py: raw = linear + gravity, exponential 20 s gravity, lin = raw - EMA

Causal speed = the model on the 30 s window ENDING now, held until the next inference.
Scenario = GPS cut mid-ride, road-bound on OpenStreetMap, scored to the end:
  * 11 benchmark rides, cut at 60 s, each with its own leave-one-run-out model
  * 11 Sept evening demo rides at their real cuts, all 11 models (median)

    python scripts/eval_causal_features.py
"""
from __future__ import annotations
import glob, os, sys
import numpy as np, pandas as pd, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dhruva.model import ResNet1D
from dhruva.infer_speed import features as offline_features, TRAIN_HZ
from dhruva.osm_road import OsmNetwork, road_from_osm, to_xy
from dhruva.mapmatch import RoadPolyline
from dhruva.gpsclean import clean_for_truth
from evaluate_system import _extend

G = 9.80665
HZ = TRAIN_HZ
W = 300
N8 = max(int(8.0 * HZ) | 1, 3)          # 81 samples, as offline
STRIDE = 5                               # inference every 0.5 s


def load(folder):
    """load_imu, plus the gravity sensor on the same 10 Hz grid."""
    a = pd.read_csv(os.path.join(folder, "Accelerometer.csv"))
    g = pd.read_csv(os.path.join(folder, "Gyroscope.csv"))
    r = pd.read_csv(os.path.join(folder, "Gravity.csv"))
    ta, tg, tr = (d["seconds_elapsed"].to_numpy(float) for d in (a, g, r))
    t0, t1 = max(ta[0], tg[0]), min(ta[-1], tg[-1])
    grid = np.arange(t0, t1, 1.0 / HZ)
    rs = lambda t, v: np.column_stack([np.interp(grid, t, v[:, i]) for i in range(3)])
    acc = rs(ta, a[["x", "y", "z"]].to_numpy(float)); gyro = rs(tg, g[["x", "y", "z"]].to_numpy(float))
    grav = rs(tr, r[["x", "y", "z"]].to_numpy(float))
    return grid - grid[0], acc, gyro, grav


def trailing(x, n):
    return pd.DataFrame(x).rolling(n, min_periods=1).mean().to_numpy()


def basis(down):
    down = down / np.maximum(np.linalg.norm(down, axis=1, keepdims=True), 1e-9)
    ref = np.tile(np.array([1.0, 0.0, 0.0]), (len(down), 1))
    ref[np.abs(np.einsum("ij,ij->i", ref, down)) > 0.9] = np.array([0.0, 1.0, 0.0])
    east = np.cross(ref, down); east /= np.maximum(np.linalg.norm(east, axis=1, keepdims=True), 1e-9)
    north = np.cross(down, east)
    return np.stack([east, north, down], axis=1)


def feats(R, gyro, acc):
    return np.concatenate([np.einsum("nij,nj->ni", R, gyro), np.einsum("nij,nj->ni", R, acc)], axis=1).astype(np.float32)


def variants(acc, gyro, grav):
    out = {"offline (centred)": offline_features(acc, gyro)}
    sm = trailing(acc, N8); down = sm / np.maximum(np.linalg.norm(sm, axis=1, keepdims=True), 1e-6)
    out["V1 replica, trailing"] = feats(basis(trailing(acc + down * G, N8)), gyro, acc)
    out["V2 gravity sensor"] = feats(basis(trailing(grav, N8)), gyro, acc)
    raw = acc + grav; alpha = float(np.exp(-1.0 / (20.0 * HZ))); ema = np.array([0.0, 0.0, G]); E = np.empty_like(raw)
    for i in range(len(raw)):
        ema = alpha * ema + (1 - alpha) * raw[i]; E[i] = ema
    out["V3 edge engine"] = feats(basis(E), gyro, raw - E)
    return out


def speeds(F, t, net, mean, std, causal):
    idx = np.arange(0, len(F) - W, STRIDE)
    X = ((np.stack([F[i:i + W] for i in idx]).transpose(0, 2, 1) - mean) / std).astype(np.float32)
    with torch.no_grad():
        m = np.concatenate([net(torch.from_numpy(X[k:k + 512])).numpy().ravel() for k in range(0, len(X), 512)])
    v = np.clip(m / (W / HZ), 0, None)
    if not causal:                                   # offline: speed at the window centre, interpolated
        return np.interp(t, t[idx + W // 2], v)
    tend = t[idx + W - 1]                            # causal: latest window that has ENDED, held
    k = np.searchsorted(tend, t, side="right") - 1
    return np.where(k >= 0, v[np.clip(k, 0, None)], v[0])


def score(t, v, truth, cum, road, i0, i1):
    dt = float(np.median(np.diff(t)))
    p = road.at(road.project(truth[i0]) + np.cumsum(v[i0:i1]) * dt)[0][-1]
    return 100 * float(np.linalg.norm(p - truth[i1 - 1])) / float(cum[i1 - 1] - cum[i0])


def main():
    S = pd.read_csv("data/landmarks/campus_landmarks.csv"); lat0, lon0 = float(S.lat.iloc[0]), float(S.lon.iloc[0])
    net_osm = OsmNetwork("data/osm/wide_ways_v2.json", lat0, lon0)
    models = {os.path.basename(c)[:-3]: torch.load(c, map_location="cpu", weights_only=False) for c in sorted(glob.glob("checkpoints/loro/*.pt"))}
    nets = {}
    for k, ck in models.items():
        n = ResNet1D(in_ch=6, out_dim=1); n.load_state_dict(ck["model"]); n.eval(); nets[k] = (n, np.asarray(ck["mean"]), np.asarray(ck["std"]))
    bench = sorted(glob.glob("rides/bike/*/")) + sorted(glob.glob("rides/aug31/*/"))
    evening = [("rides/app_DhruvaRun_2026-09-11_19-37-43", 55.0, 172.0), ("rides/app_DhruvaRun_2026-09-11_19-41-28", 48.0, 137.0)]
    jobs = [(r.rstrip("/"), 60.0, None, [os.path.basename(r.rstrip("/"))]) for r in bench] + [(r, c, f, list(models)) for r, c, f in evening]
    rows = []
    for folder, cut, fin, use in jobs:
        t, acc, gyro, grav = load(folder)
        V = variants(acc, gyro, grav)
        L = pd.read_csv(os.path.join(folder, "Location.csv")); lt = L.seconds_elapsed.to_numpy(float)
        gps = to_xy(L.latitude.to_numpy(float), L.longitude.to_numpy(float), lat0, lon0); gps, lt = clean_for_truth(gps, lt)
        road = RoadPolyline(_extend(np.asarray(road_from_osm(net_osm, gps)[0], float), 3000.0), 2.0)
        keep = (t >= lt[0]) & (t <= lt[-1])
        truth = np.column_stack([np.interp(t, lt, gps[:, 0]), np.interp(t, lt, gps[:, 1])])
        cum = np.r_[0, np.cumsum(np.linalg.norm(np.diff(truth, axis=0), axis=1))]
        i0 = int(np.searchsorted(t, cut)); i1 = int(np.searchsorted(t, fin)) + 1 if fin else int(np.flatnonzero(keep)[-1]) + 1
        corr = {name: float(np.mean([np.corrcoef(F[:, c], V["offline (centred)"][:, c])[0, 1] for c in range(6)])) for name, F in V.items()}
        # the last two rows split the causal cost: levelling that looks back vs speed that arrives late
        combos = [("offline (centred)", "offline (centred)", False),
                  ("V1 replica, trailing", "V1 replica, trailing", True),
                  ("V2 gravity sensor", "V2 gravity sensor", True),
                  ("V3 edge engine", "V3 edge engine", True),
                  ("split: offline feats, late speed", "offline (centred)", True),
                  ("split: V1 feats, centred speed", "V1 replica, trailing", False)]
        for label, key, causal in combos:
            d = [score(t, speeds(V[key], t, *nets[m], causal=causal), truth, cum, road, i0, i1) for m in use]
            rows.append(dict(ride=os.path.basename(folder), kind="evening" if fin else "bench", variant=label,
                             drift=float(np.median(d)), passes=int(np.sum(np.array(d) < 10)), n=len(d), corr=corr[key]))
        print(f"{os.path.basename(folder)[:34]:34s} " + "  ".join(f"{r['drift']:5.1f}%" for r in rows[-len(combos):]), flush=True)
    D = pd.DataFrame(rows)
    D.to_csv("results/causal_features.csv", index=False)
    print("\nvariant                           bench median (cut 60 s)  pass  | 19-37 cut      19-41 cut      (11-model median, passes) | feature corr")
    for name in D.variant.unique():
        b = D[(D.variant == name) & (D.kind == "bench")]; e = D[(D.variant == name) & (D.kind == "evening")].reset_index()
        print(f"{name:34s} {b.drift.median():6.1f}%             {int((b.drift < 10).sum()):2d}/11 | "
              f"{e.drift[0]:5.1f}% {e.passes[0]:2d}/11   {e.drift[1]:5.1f}% {e.passes[1]:2d}/11                      | {D.loc[D.variant == name, 'corr'].mean():.3f}")
    print("saved results/causal_features.csv")


if __name__ == "__main__":
    main()
