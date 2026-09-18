#!/usr/bin/env python3
"""Re-derive ISRO requirements 4 and 5 reproducibly.

The 31-Aug audit found both requirements' numbers came from an uncommitted
script (RESOURCES.md 7u) and marked them PROVISIONAL. This is the replacement.

Scenario per run: GNSS for the first 20% of the ride, a hard blackout for the
middle 60%, GNSS returns for the last 20%. That exercises BOTH transitions the
PS asks about, in one pass.

  requirement 4  -- GNSS+INS fusion: drift through the blackout, and throughput
  requirement 5  -- seamless handling: the jump at blackout onset and at
                    reacquisition, filter-optimal vs user-facing

    python scripts/eval_fusion.py
"""
from __future__ import annotations
import glob, os, sys, time
import numpy as np, pandas as pd, torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dhruva.osm_road import OsmNetwork, road_from_osm, to_xy
from dhruva.mapmatch import RoadPolyline
from dhruva.gpsclean import clean_for_truth
from dhruva.fusion import GnssInsFusion, SeamlessOutput
from dhruva import align
from evaluate_system import speed_from_imu, _extend
from harness import metrics

USE_ROAD_POS = os.environ.get("ROADPOS", "1") == "1"

RUNS = sorted(glob.glob("rides/bike/*/")) + sorted(glob.glob("rides/aug31/*/"))


def prepare(d, net, lat0, lon0, ckpt):
    L = pd.read_csv(os.path.join(d, "Location.csv"))
    lt = L["seconds_elapsed"].to_numpy(float)
    gps = to_xy(L["latitude"].to_numpy(float), L["longitude"].to_numpy(float), lat0, lon0)
    gps, lt = clean_for_truth(gps, lt)

    t, v = speed_from_imu(d, ckpt)
    A = pd.read_csv(os.path.join(d, "Accelerometer.csv"))
    G = pd.read_csv(os.path.join(d, "Gyroscope.csv"))
    Gr = pd.read_csv(os.path.join(d, "Gravity.csv"))
    g = np.arange(0, min(A["seconds_elapsed"].iloc[-1], G["seconds_elapsed"].iloc[-1]), 0.05)
    rs = lambda df: np.column_stack([np.interp(g, df["seconds_elapsed"], df[c]) for c in ("x", "y", "z")])
    R = align.estimate_tilt(rs(A) + rs(Gr), 20.0)
    lev_w = align.apply(R, rs(G))
    lev_a = align.apply(R, rs(A))
    wz = np.interp(t, g, lev_w[:, 2])
    ax = np.interp(t, g, lev_a[:, 0])
    ay = np.interp(t, g, lev_a[:, 1])

    osm, _ = road_from_osm(net, gps)
    road = RoadPolyline(_extend(osm, 400.0), 2.0)
    truth = np.column_stack([np.interp(t, lt, gps[:, 0]), np.interp(t, lt, gps[:, 1])])
    return t, np.clip(v, 0, None), wz, ax, ay, truth, road


def run_one(t, v, wz, ax, ay, truth, road, use_seamless=True):
    n = len(t)
    i0, i1 = int(0.20 * n), int(0.80 * n)          # blackout window
    dt = float(np.median(np.diff(t)))

    f = GnssInsFusion()
    h0 = np.arctan2(truth[5, 1] - truth[0, 1], truth[5, 0] - truth[0, 0])
    f.initialise(truth[0], v[0], h0)
    so = SeamlessOutput()

    est, rep = np.empty((n, 2)), np.empty((n, 2))
    jump_onset = jump_reacq = 0.0
    t0 = time.perf_counter()
    for i in range(n):
        blackout = i0 <= i < i1
        gnss = None if blackout else truth[i]
        # forward acceleration: level-frame accel projected on current heading
        psi = f.s.heading
        a_fwd = float(ax[i] * np.cos(psi) + ay[i] * np.sin(psi))
        prev = f.s.pos.copy()
        arc = road.project(f.s.pos)
        rp, rh = road.at(np.array([arc]))
        f.step(dt, float(wz[i]), a_fwd, gnss_pos=gnss,
               model_speed=float(v[i]), road_heading=float(rh[0]),
               road_pos=(None if USE_ROAD_POS else None) or (rp[0] if USE_ROAD_POS else None))
        est[i] = f.s.pos
        if i == i0:
            jump_onset = float(np.linalg.norm(f.s.pos - prev))
        if i == i1:
            jump_reacq = float(np.linalg.norm(f.s.pos - prev))
        rep[i] = so.update(f.s.pos, dt, f.s.mode) if use_seamless else f.s.pos
    hz = n / max(time.perf_counter() - t0, 1e-9)

    sl = slice(i0, i1)
    res = metrics.score(est[sl], truth[sl], t[sl])
    # largest single-epoch move of the USER-FACING dot during reacquisition
    k = slice(i1, min(i1 + 200, n))
    rep_jump = float(np.max(np.linalg.norm(np.diff(rep[k], axis=0), axis=1))) if k.stop > k.start + 1 else 0.0
    est_jump = float(np.max(np.linalg.norm(np.diff(est[k], axis=0), axis=1))) if k.stop > k.start + 1 else 0.0
    return res, jump_onset, jump_reacq, est_jump, rep_jump, hz


def main():
    M = pd.read_csv("data/landmarks/campus_landmarks.csv")
    lat0, lon0 = M.lat.iloc[0], M.lon.iloc[0]
    net = OsmNetwork("data/osm/wide_ways.json", lat0, lon0)
    loro_dir = "checkpoints/loro"
    use_loro = os.path.isdir(loro_dir) and len(os.listdir(loro_dir)) >= len(RUNS)
    fallback = torch.load("checkpoints/speed_bike.pt", map_location="cpu", weights_only=False)
    print("speed model: " + ("cached leave-one-run-out (the scored run never trained it)"
                             if use_loro else "checkpoints/speed_bike.pt  [NOT held out]") + "\n")

    print("REQUIREMENT 4 — GNSS+INS fusion through a 60%-of-route blackout")
    print(f"{'run':30s} {'blackout m':>11} {'drift m':>8} {'drift %':>8} {'pass':>5} {'Hz':>9}")
    print("-" * 78)
    D, HZ, ON, EJ, RJ = [], [], [], [], []
    for d in RUNS:
        try:
            name = os.path.basename(d[:-1])
            lp = os.path.join(loro_dir, name + ".pt")
            ck = torch.load(lp, map_location="cpu", weights_only=False) if (use_loro and os.path.exists(lp)) else fallback
            p = prepare(d, net, lat0, lon0, ck)
            res, j0, j1, ej, rj, hz = run_one(*p)
        except Exception as e:
            print(f"{os.path.basename(d[:-1])[:30]:30s} FAILED: {e}")
            continue
        D.append(res.drift_pct); HZ.append(hz); ON.append(j0); EJ.append(ej); RJ.append(rj)
        print(f"{os.path.basename(d[:-1])[:30]:30s} {res.distance_m:11.0f} {res.final_drift_m:8.1f} "
              f"{res.drift_pct:8.1f} {'YES' if res.passes_isro else 'NO':>5} {hz:9.0f}")
    D = np.array(D)
    print("-" * 78)
    print(f"median {np.median(D):.1f}%   mean {D.mean():.1f}%   pass {100*np.mean(D<10):.0f}%   "
          f"throughput {np.median(HZ):,.0f} epochs/s")

    print("\nREQUIREMENT 5 — seamless transition, both directions")
    print(f"  GNSS -> dead reckoning : max jump {max(ON):.3f} m  (no branch, no re-init)")
    print(f"  dead reckoning -> GNSS : optimal filter jumps {np.median(EJ):.1f} m median, "
          f"{max(EJ):.1f} m worst")
    print(f"                           user-facing dot     {np.median(RJ):.2f} m median, "
          f"{max(RJ):.2f} m worst")
    print(f"  improvement            : {np.median(EJ)/max(np.median(RJ),1e-9):.0f}x smaller "
          f"largest single-epoch move")


if __name__ == "__main__":
    main()
