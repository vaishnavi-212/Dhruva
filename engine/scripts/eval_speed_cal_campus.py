#!/usr/bin/env python3
"""Does calibrating the AI speed against GPS before the cut help on our own campus rides?

The same rule as the IO-VNBD result (scripts/eval_iovnbd.py, "AI cal + road"), settings unchanged:
scale = GPS distance / AI distance over the 120 s before the cut, clipped to 0.8-1.25, not applied if
the AI distance is under 50 m. One phone-honest change: only speed from 30 s windows that have fully
ENDED before the cut counts (the phone cannot see ahead), so the first 30 s of a ride give nothing.

Models: checkpoints/loro_phone_replica/ (the phone-frame models, 5.0% headline). A benchmark ride is
scored only with the model that never saw it; the two 11 Sept evening rides with all 11 (median).

  A  the existing cuts: benchmark at 60 s to the end of the ride, evening at the rider's own cut
  B  more cuts: every 15 s from 45 s while at least 300 m of the ride remains, blackout to the end

    python scripts/eval_speed_cal_campus.py      ->  results/speed_cal_campus.csv
"""
from __future__ import annotations
import glob, os, sys
import numpy as np, pandas as pd, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_causal_features as EC
from train_eval_phone_features import frame_features
from dhruva.model import ResNet1D
from dhruva.osm_road import OsmNetwork, road_from_osm, to_xy
from dhruva.mapmatch import RoadPolyline
from dhruva.gpsclean import clean_for_truth
from evaluate_system import _extend

CAL_S, CAL_MIN, CAL_MAX, CAL_MIN_M = 120.0, 0.8, 1.25, 50.0     # identical to eval_iovnbd.py
W_S = EC.W / EC.HZ                                                # 30 s window


def drift(t, v, truth, cum, road, i0, i1):
    return EC.score(t, v, truth, cum, road, i0, i1)


def scale_before(t, v, cum, i0):
    """GPS / AI distance over the CAL_S before the cut, using only windows that have ended."""
    lo = max(t[i0] - CAL_S, W_S)                                  # first ended window at t = 30 s
    if lo >= t[i0]:
        return 1.0
    a, dt = int(np.searchsorted(t, lo)), float(np.median(np.diff(t)))
    ai, gps = float(np.sum(v[a:i0]) * dt), float(cum[i0] - cum[a])
    return float(np.clip(gps / ai, CAL_MIN, CAL_MAX)) if ai > CAL_MIN_M else 1.0


def main():
    base = torch.load("checkpoints/speed_best.pt", map_location="cpu", weights_only=False)
    mean, std = np.asarray(base["mean"]), np.asarray(base["std"])
    bench = [r.rstrip("/") for r in sorted(glob.glob("rides/bike/*/")) + sorted(glob.glob("rides/aug31/*/"))]
    evening = {"rides/app_DhruvaRun_2026-09-11_19-37-43": (55.0, 172.0), "rides/app_DhruvaRun_2026-09-11_19-41-28": (48.0, 137.0)}
    nets = {}
    for f in bench:
        ck = torch.load(f"checkpoints/loro_phone_replica/{os.path.basename(f)}.pt", map_location="cpu", weights_only=False)
        n = ResNet1D(in_ch=6, out_dim=1); n.load_state_dict(ck["model"]); n.eval(); nets[os.path.basename(f)] = n
    S = pd.read_csv("data/landmarks/campus_landmarks.csv"); lat0, lon0 = float(S.lat.iloc[0]), float(S.lon.iloc[0])
    net_osm = OsmNetwork("data/osm/wide_ways_v2.json", lat0, lon0)
    rows = []
    for folder in bench + list(evening):
        t, acc, gyro, grav = EC.load(folder)
        F = frame_features("replica", acc, gyro, grav)
        L = pd.read_csv(os.path.join(folder, "Location.csv")); lt = L.seconds_elapsed.to_numpy(float)
        gps = to_xy(L.latitude.to_numpy(float), L.longitude.to_numpy(float), lat0, lon0); gps, lt = clean_for_truth(gps, lt)
        road = RoadPolyline(_extend(np.asarray(road_from_osm(net_osm, gps)[0], float), 3000.0), 2.0)
        keep = (t >= lt[0]) & (t <= lt[-1])
        truth = np.column_stack([np.interp(t, lt, gps[:, 0]), np.interp(t, lt, gps[:, 1])])
        cum = np.r_[0, np.cumsum(np.linalg.norm(np.diff(truth, axis=0), axis=1))]
        name = os.path.basename(folder)
        use = list(nets) if folder in evening else [name]
        V = {m: EC.speeds(F, t, nets[m], mean, std, causal=True) for m in use}
        end = int(np.flatnonzero(keep)[-1]) + 1
        if folder in evening:
            c, e = evening[folder]; cuts_a, i_end = [c], int(np.searchsorted(t, e)) + 1
        else:
            cuts_a, i_end = [60.0], end
        cuts_b = [c for c in np.arange(45.0, t[i_end - 1], 15.0) if cum[i_end - 1] - cum[int(np.searchsorted(t, c))] >= 300.0]
        for set_name, cuts in (("A", cuts_a), ("B", cuts_b)):
            for cut in cuts:
                i0 = int(np.searchsorted(t, cut))
                raw = [drift(t, V[m], truth, cum, road, i0, i_end) for m in use]
                sc = [scale_before(t, V[m], cum, i0) for m in use]
                cal = [drift(t, V[m] * s, truth, cum, road, i0, i_end) for m, s in zip(use, sc)]
                rows.append(dict(set=set_name, ride=name, kind="evening" if folder in evening else "bench", cut_s=cut,
                                 gps_off_m=float(cum[i_end - 1] - cum[i0]), cal_used_s=float(min(CAL_S, max(cut - W_S, 0))),
                                 scale=float(np.median(sc)), raw=float(np.median(raw)), cal=float(np.median(cal)),
                                 raw_pass=int(np.sum(np.array(raw) < 10)), cal_pass=int(np.sum(np.array(cal) < 10)), n=len(use)))
        a = [r for r in rows if r["ride"] == name and r["set"] == "A"][0]
        print(f"{name[:36]:36s} A: raw {a['raw']:5.1f}%  cal {a['cal']:5.1f}%  scale {a['scale']:.2f}  (cal from {a['cal_used_s']:.0f} s)"
              f" | B: {sum(r['ride'] == name and r['set'] == 'B' for r in rows)} cuts", flush=True)
    D = pd.DataFrame(rows); D.to_csv("results/speed_cal_campus.csv", index=False)
    for s in ("A", "B"):
        for k in ("bench", "evening"):
            X = D[(D.set == s) & (D.kind == k)]
            if len(X):
                print(f"set {s} {k:7s} ({len(X):3d} blackouts): raw median {X.raw.median():5.1f}% ({(X.raw < 10).sum()}/{len(X)} under 10%)"
                      f"  ->  cal {X.cal.median():5.1f}% ({(X.cal < 10).sum()}/{len(X)})  | cal better in {(X.cal < X.raw).mean() * 100:.0f}%")
    B = D[D.set == "B"]
    for lo, hi in ((0, 45), (45, 90), (90, 121)):
        X = B[(B.cal_used_s >= lo) & (B.cal_used_s < hi)]
        if len(X):
            print(f"set B, {lo:3d}-{hi:3d} s of GPS to calibrate on: {len(X):3d} blackouts, raw {X.raw.median():5.1f}% -> cal {X.cal.median():5.1f}%")
    print("saved results/speed_cal_campus.csv")


if __name__ == "__main__":
    main()
