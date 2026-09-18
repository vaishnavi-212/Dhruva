#!/usr/bin/env python3
"""The fusion filter under PHONE conditions, before porting it to Kotlin.

eval_fusion.py (the reference) feeds the filter GNSS position at 10 Hz, the AI speed at every epoch,
and a forward acceleration that combines the level frame's ARBITRARY x-axis with the true heading.
A phone gets GNSS once a second, runs the AI model only during a blackout (J1, Part 2), and has no
clean forward axis. Same rides, same held-out models, same 60%-of-route blackout; four variants:

  reference      exactly eval_fusion.py
  a_fwd = 0      no acceleration input (constant-speed process model)
  + GNSS 1 Hz    position and speed from the fix only once a second, with the fix's own sigma
  + AI in DR     AI speed only during the blackout  (= what the phone app will do)

    python scripts/eval_fusion_phone.py
"""
from __future__ import annotations
import glob, os, sys
import numpy as np, pandas as pd, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dhruva.osm_road import OsmNetwork
from dhruva.fusion import GnssInsFusion
from harness import metrics
import eval_fusion as EF


def local_project(road, pos, arc_prev, win=40.0):
    """Arc of the nearest road point within +-win metres of the previous arc (never the far side of a loop)."""
    m = np.abs(road.s - arc_prev) <= win
    idx = np.flatnonzero(m)
    j = idx[int(np.argmin(np.linalg.norm(road.xy[idx] - pos, axis=1)))]
    return float(road.s[j])


def run(t, v, wz, ax, ay, truth, road, afwd=True, gnss_hz=None, ai_only_dr=False, local=False, road_dr_only=False):
    n = len(t); i0, i1 = int(0.20 * n), int(0.80 * n); dt = float(np.median(np.diff(t)))
    f = GnssInsFusion()
    h0 = np.arctan2(truth[5, 1] - truth[0, 1], truth[5, 0] - truth[0, 0])
    f.initialise(truth[0], v[0], h0)
    every = None if gnss_hz is None else max(int(round(1.0 / (gnss_hz * dt))), 1)
    gspeed = np.r_[0, np.linalg.norm(np.diff(truth, axis=0), axis=1)] / dt
    est = np.empty((n, 2))
    arc_prev = road.project(truth[0])
    for i in range(n):
        blackout = i0 <= i < i1
        gnss = gsp = None
        if not blackout and (every is None or i % every == 0):
            gnss = truth[i]; gsp = float(gspeed[i]) if every is not None else None
        psi = f.s.heading
        a_fwd = float(ax[i] * np.cos(psi) + ay[i] * np.sin(psi)) if afwd else 0.0
        arc = local_project(road, f.s.pos, arc_prev) if local else road.project(f.s.pos)
        arc_prev = arc
        rp, rh = road.at(np.array([arc]))
        ms = float(v[i]) if (not ai_only_dr or blackout) else None
        use_road = blackout or not road_dr_only
        f.step(dt, float(wz[i]), a_fwd, gnss_pos=gnss, gnss_speed=gsp, model_speed=ms,
               road_heading=float(rh[0]) if use_road else None, road_pos=rp[0] if use_road else None)
        est[i] = f.s.pos
    sl = slice(i0, i1)
    return metrics.score(est[sl], truth[sl], t[sl]).drift_pct


def main():
    M = pd.read_csv("data/landmarks/campus_landmarks.csv"); lat0, lon0 = M.lat.iloc[0], M.lon.iloc[0]
    net = OsmNetwork("data/osm/wide_ways.json", lat0, lon0)
    variants = [("reference", {}), ("a_fwd = 0", dict(afwd=False)),
                ("+ GNSS 1 Hz", dict(afwd=False, gnss_hz=1.0)),
                ("+ AI in DR only (phone)", dict(afwd=False, gnss_hz=1.0, ai_only_dr=True)),
                ("phone + local projection", dict(afwd=False, gnss_hz=1.0, ai_only_dr=True, local=True)),
                ("phone + local + road in DR only", dict(afwd=False, gnss_hz=1.0, ai_only_dr=True, local=True, road_dr_only=True)),
                ("reference + local + road in DR only", dict(local=True, road_dr_only=True))]
    rows = []
    for d in EF.RUNS:
        name = os.path.basename(d[:-1])
        ck = torch.load(os.path.join("checkpoints/loro", name + ".pt"), map_location="cpu", weights_only=False)
        p = EF.prepare(d, net, lat0, lon0, ck)
        r = {k: run(*p, **kw) for k, kw in variants}; r["ride"] = name; rows.append(r)
        print(f"{name[:30]:30s} " + "  ".join(f"{r[k]:5.1f}%" for k, _ in variants), flush=True)
    D = pd.DataFrame(rows)
    print()
    for k, _ in variants:
        print(f"{k:36s} median {D[k].median():5.1f}%   pass {int((D[k] < 10).sum())}/11")


if __name__ == "__main__":
    main()
