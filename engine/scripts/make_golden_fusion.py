#!/usr/bin/env python3
"""Golden files for the phone's fusion filter (FusionEngine.kt).

Runs dhruva/fusion.py's GnssInsFusion + SeamlessOutput through one benchmark ride under PHONE
conditions (eval_fusion_phone.py, variant "phone + local projection"): GNSS position and speed once a
second, AI speed only during the blackout (middle 60% of the ride), no forward-acceleration input, the
road measured at the nearest point within 40 m of the previous one. Every epoch's INPUTS and OUTPUTS
are written, so the Kotlin filter can be fed identical inputs and must give identical outputs.

    python scripts/make_golden_fusion.py --out <app>/app/src/test/resources/golden
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np, pandas as pd, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_fusion as EF
from eval_fusion_phone import local_project
from dhruva.osm_road import OsmNetwork
from dhruva.fusion import GnssInsFusion, SeamlessOutput

RIDE = "94CC_6RQ-2026-08-30_16-39-43"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    M = pd.read_csv("data/landmarks/campus_landmarks.csv"); lat0, lon0 = M.lat.iloc[0], M.lon.iloc[0]
    net = OsmNetwork("data/osm/wide_ways.json", lat0, lon0)
    d = next(r for r in EF.RUNS if RIDE in r)
    ck = torch.load(f"checkpoints/loro/{RIDE}.pt", map_location="cpu", weights_only=False)
    t, v, wz, ax, ay, truth, road = EF.prepare(d, net, lat0, lon0, ck)
    n = len(t); i0, i1 = int(0.20 * n), int(0.80 * n); dt = float(np.median(np.diff(t)))
    every = max(int(round(1.0 / dt)), 1)
    gspeed = np.r_[0, np.linalg.norm(np.diff(truth, axis=0), axis=1)] / dt

    f = GnssInsFusion()
    h0 = float(np.arctan2(truth[5, 1] - truth[0, 1], truth[5, 0] - truth[0, 0]))
    f.initialise(truth[0], v[0], h0)
    so = SeamlessOutput()
    arc_prev = road.project(truth[0])
    nan = float("nan")
    rows = []
    for i in range(n):
        blackout = i0 <= i < i1
        gnss = truth[i] if (not blackout and i % every == 0) else None
        gsp = float(gspeed[i]) if gnss is not None else None
        ms = float(v[i]) if blackout else None
        arc = local_project(road, f.s.pos, arc_prev); arc_prev = arc
        rp, rh = road.at(np.array([arc]))
        f.step(dt, float(wz[i]), 0.0, gnss_pos=gnss, gnss_speed=gsp, model_speed=ms,
               road_heading=float(rh[0]), road_pos=rp[0])
        rep = so.update(f.s.pos, dt, f.s.mode)
        x = f.s.x; P = f.s.P
        rows.append([dt, float(wz[i]), 0.0,
                     gnss[0] if gnss is not None else nan, gnss[1] if gnss is not None else nan,
                     gsp if gsp is not None else nan, ms if ms is not None else nan,
                     float(rh[0]), float(rp[0][0]), float(rp[0][1]),
                     x[0], x[1], x[2], x[3], x[4], P[0, 0], P[1, 1], P[2, 2], P[3, 3], P[4, 4],
                     {"INIT": 0, "GNSS_AIDED": 1, "DEAD_RECKONING": 2, "DR_LANDMARK": 3}[f.s.mode],
                     rep[0], rep[1], float(truth[i][0]), float(truth[i][1])])
    cols = ["dt", "gyro_z", "accel_fwd", "gnss_x", "gnss_y", "gnss_speed", "model_speed", "road_heading", "road_x", "road_y",
            "x", "y", "v", "psi", "bias", "p00", "p11", "p22", "p33", "p44", "mode", "dot_x", "dot_y", "truth_x", "truth_y"]
    with open(os.path.join(a.out, "fusion_golden.csv"), "w") as fh:
        fh.write(f"# ride {RIDE}; init x={float(truth[0][0])!r} y={float(truth[0][1])!r} v={float(v[0])!r} psi={float(h0)!r}; blackout epochs {i0}-{i1}\n")
        fh.write(",".join(cols) + "\n")
        for r in rows:
            fh.write(",".join(repr(float(c)) for c in r) + "\n")
    err = np.linalg.norm(np.array([r[10:12] for r in rows[i0:i1]]) - truth[i0:i1], axis=1)
    dist = float(np.linalg.norm(np.diff(truth[i0:i1], axis=0), axis=1).sum())
    print(f"golden fusion: {n} epochs at {1/dt:.0f} Hz, blackout {i0}-{i1} ({dist:.0f} m), "
          f"final drift {100*err[-1]/dist:.1f}% -> {a.out}/fusion_golden.csv")


if __name__ == "__main__":
    main()
