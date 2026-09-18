#!/usr/bin/env python3
"""End-to-end evaluation of the full Dhruva pipeline on the 30-Aug bike laps.

Reproduces the number STATUS.md reports for ISRO requirement 8 (drift < 10% of
distance travelled). Written during the 31-Aug audit because the original figure
came from an uncommitted ad-hoc script and could not be re-derived.

Pipeline, GNSS-denied for the WHOLE lap after the first fix:

    OSM road (public map)  ->  RoadPolyline
    IMU  ->  A2 speed model  ->  distance travelled   (no GPS)
    constrained_track: propagate ALONG the road, heading from road geometry
    accelerometer -> landmark detector -> OnlineMatcher vs surveyed map
    matched landmark  ->  along-track position reset
    score vs GPS truth

    python scripts/evaluate_system.py --ckpt checkpoints/speed_bike.pt
"""
from __future__ import annotations
import argparse, glob, os, sys
import numpy as np, pandas as pd, torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dhruva.osm_road import OsmNetwork, road_from_osm, to_xy
from dhruva.mapmatch import RoadPolyline
from dhruva.matcher import OnlineMatcher, MatchConfig, GAP
from dhruva.model import ResNet1D
from dhruva.infer_speed import load_imu, features, TRAIN_HZ
from dhruva import landmarks
from harness import metrics


def speed_from_imu(folder, ckpt_obj, device="cpu"):
    """A2 model -> per-sample speed (m/s). Uses IMU only."""
    W = int(ckpt_obj["window"]); mean, std = ckpt_obj["mean"], ckpt_obj["std"]
    net = ResNet1D(in_ch=6, out_dim=1).to(device)
    net.load_state_dict(ckpt_obj["model"]); net.eval()
    t, acc, gyro = load_imu(folder)
    F = features(acc, gyro)
    idx = np.arange(0, len(F) - W, 1)
    X = np.stack([F[i:i + W] for i in idx]).transpose(0, 2, 1)
    X = (X - mean) / std
    out = []
    with torch.no_grad():
        for i in range(0, len(X), 256):
            out.append(net(torch.tensor(X[i:i+256], dtype=torch.float32).to(device)).cpu().numpy())
    dist_per_window = np.concatenate(out).ravel()        # metres per W samples
    v = dist_per_window / (W / TRAIN_HZ)                 # -> m/s
    # centre each window's speed, then extend to full length
    tv = t[idx + W // 2]
    return t, np.interp(t, tv, v)


def _extend(xy, pad_m=400.0):
    """Continue the road straight past both ends.

    `RoadPolyline.at()` CLIPS arc length to the road, and `road_from_osm` builds
    the road from the run's own GPS — so the road can be SHORTER than the path
    actually ridden, especially when the route revisits itself. Any overshoot is
    then silently pinned to the road's end, which is where the ride finished, and
    the run reports a near-perfect drift it did not earn.

    Measured (RESOURCES.md 8x): three of eleven runs were pinned this way and
    reported 0.0%, 0.3% and 0.1%. Their true errors are far larger.

    The direction for each end must come from the last NON-DEGENERATE segment.
    Snapped OSM polylines contain repeated points (6 of 509 on the ghat route,
    including the final segment), and normalising a zero-length segment gives a
    meaningless direction — which silently disabled the tail extension entirely.
    """
    xy = np.asarray(xy, float)
    seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    good = np.flatnonzero(seg > 1e-6)
    if len(good) == 0:
        return xy
    i0 = good[0]                       # first real segment, for the head
    i1 = good[-1]                      # last real segment, for the tail
    d0 = xy[i0] - xy[i0 + 1]; d0 /= np.linalg.norm(d0)
    d1 = xy[i1 + 1] - xy[i1]; d1 /= np.linalg.norm(d1)
    n = max(int(pad_m / 2.0), 2)
    head = xy[0] + np.outer(np.arange(n, 0, -1) * 2.0, d0)
    tail = xy[-1] + np.outer(np.arange(1, n + 1) * 2.0, d1)
    return np.vstack([head, xy, tail])


def map_landmark_arcs(road, lat, lon, lat0, lon0):
    xy = to_xy(np.asarray(lat, float), np.asarray(lon, float), lat0, lon0)
    return np.sort(np.array([road.project(p) for p in xy]))


def evaluate_run(folder, net_ckpt, net_obj, lm, lat0, lon0, use_landmarks=True, device="cpu"):
    L = pd.read_csv(os.path.join(folder, "Location.csv"))
    lt = L["seconds_elapsed"].to_numpy(float)
    gps = to_xy(L["latitude"].to_numpy(float), L["longitude"].to_numpy(float), lat0, lon0)
    osm_xy, _ = road_from_osm(OsmNetwork.CACHE, gps)
    osm_xy = _extend(osm_xy, 400.0)     # see _extend: clamping must not flatter us
    road = RoadPolyline(osm_xy, 2.0)

    t, v = speed_from_imu(folder, net_obj, device)
    dt = float(np.median(np.diff(t)))
    v = np.clip(v, 0.0, None)

    s0 = road.project(gps[0])
    arc = s0 + np.cumsum(v) * dt

    if use_landmarks:
        A = pd.read_csv(os.path.join(folder, "Accelerometer.csv"))
        ta = A["seconds_elapsed"].to_numpy(float)
        hz = 1.0 / np.median(np.diff(ta))
        ev = landmarks.detect(A[["x", "y", "z"]].to_numpy(float), ta, hz)
        map_s = map_landmark_arcs(road, lm.lat, lm.lon, lat0, lon0)
        om = OnlineMatcher(map_s, cfg=MatchConfig())
        for e in ev:
            i = int(np.searchsorted(t, e.t))
            if i >= len(arc):
                break
            j = om.match(arc[i])          # arc already carries every past reset
            if j != GAP:
                arc[i:] += (map_s[j] - arc[i])     # along-track reset

    pos, _ = road.at(arc)
    truth = np.column_stack([np.interp(t, lt, gps[:, 0]), np.interp(t, lt, gps[:, 1])])
    return metrics.score(pos, truth, t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/speed_bike.pt")
    ap.add_argument("--rides", default="rides/bike")
    ap.add_argument("--no-landmarks", action="store_true")
    a = ap.parse_args()

    M = pd.read_csv("data/landmarks/campus_landmarks.csv")
    lat0, lon0 = M.lat.iloc[0], M.lon.iloc[0]
    OsmNetwork.CACHE = OsmNetwork("data/osm/campus_ways.json", lat0, lon0)
    obj = torch.load(a.ckpt, map_location="cpu", weights_only=False)

    runs = sorted(glob.glob(os.path.join(a.rides, "*/")))
    trained_on = [os.path.basename(r[:-1]) for r in runs[:-2]]   # finetune_bike holdout=2
    print(f"checkpoint {a.ckpt}   landmarks {'OFF' if a.no_landmarks else 'ON'}\n")
    print(f"{'run':30s} {'dist':>7} {'drift m':>8} {'drift %':>8} {'pass':>5}  role")
    print("-" * 76)
    res = []
    for r in runs:
        nm = os.path.basename(r[:-1])
        s = evaluate_run(r, a.ckpt, obj, M, lat0, lon0, not a.no_landmarks)
        res.append(s)
        role = "TRAIN (leak)" if nm in trained_on else "held out"
        print(f"{nm[:30]:30s} {s.distance_m:7.0f} {s.final_drift_m:8.1f} "
              f"{s.drift_pct:8.1f} {'YES' if s.passes_isro else 'NO':>5}  {role}")
    d = np.array([x.drift_pct for x in res])
    print("-" * 76)
    print(f"median {np.median(d):.1f}%   mean {d.mean():.1f}%   "
          f"pass {100*np.mean([x.passes_isro for x in res]):.0f}%   (ISRO limit 10%)")
    ho = np.array([x.drift_pct for x, r in zip(res, runs)
                   if os.path.basename(r[:-1]) not in trained_on])
    print(f"held-out only: {np.round(ho,1)}  median {np.median(ho):.1f}%")


if __name__ == "__main__":
    main()
