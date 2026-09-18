#!/usr/bin/env python3
"""Contribution ⑤ — is the reported confidence radius honest?

The filter reports a 1-sigma position radius. For a 2-D Gaussian the error
magnitude is Rayleigh-distributed, so a nominal p-coverage circle has radius
    k(p) = sigma * sqrt(-2 ln(1-p))
Coverage should match p at EVERY level, not just at 90%. A filter can hit 90%
by accident while being badly wrong about 50% and 99%.

Reports the full calibration curve plus ANEES (average normalised estimation
error squared). For a correctly calibrated 2-D filter ANEES = 1.0:
    ANEES < 1  -> pessimistic, the circle is too big
    ANEES > 1  -> OVERCONFIDENT, the circle lies

    python scripts/eval_calibration.py
"""
from __future__ import annotations
import glob, os, sys
import numpy as np, pandas as pd, torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dhruva.osm_road import OsmNetwork
from dhruva.fusion import GnssInsFusion, FusionConfig
from eval_fusion import prepare, RUNS

LEVELS = (0.50, 0.68, 0.90, 0.95, 0.99)


def k_for(p):
    return np.sqrt(-2.0 * np.log(1.0 - p))


def collect(p, cfg):
    """Run one blackout, return (error, sigma) per epoch inside it."""
    t, v, wz, ax, ay, truth, road = p
    n = len(t); i0, i1 = int(.2 * n), int(.8 * n)
    dt = float(np.median(np.diff(t)))
    f = GnssInsFusion(cfg)
    f.initialise(truth[0], v[0],
                 np.arctan2(truth[5, 1] - truth[0, 1], truth[5, 0] - truth[0, 0]))
    err, sig = [], []
    for i in range(n):
        blk = i0 <= i < i1
        psi = f.s.heading
        a = float(ax[i] * np.cos(psi) + ay[i] * np.sin(psi))
        arc = road.project(f.s.pos); rp, rh = road.at(np.array([arc]))
        f.step(dt, float(wz[i]), a, gnss_pos=(None if blk else truth[i]),
               model_speed=float(v[i]), road_heading=float(rh[0]), road_pos=rp[0])
        if blk:
            err.append(float(np.linalg.norm(f.s.pos - truth[i])))
            sig.append(f.s.pos_sigma)
    return np.array(err), np.array(sig)


def report(name, E, S):
    anees = float(np.mean((E / np.maximum(S, 1e-9)) ** 2) / 2.0)
    cov = {p: float(np.mean(E <= k_for(p) * S)) for p in LEVELS}
    print(f"{name:26s} " + " ".join(f"{100*cov[p]:5.0f}%" for p in LEVELS) +
          f"   ANEES {anees:6.2f}")
    return cov, anees


def main():
    M = pd.read_csv("data/landmarks/campus_landmarks.csv")
    lat0, lon0 = M.lat.iloc[0], M.lon.iloc[0]
    net = OsmNetwork("data/osm/wide_ways.json", lat0, lon0)
    P = []
    for d in RUNS:
        nm = os.path.basename(d[:-1])
        ck = torch.load(f"checkpoints/loro/{nm}.pt", map_location="cpu", weights_only=False)
        P.append(prepare(d, net, lat0, lon0, ck))

    hdr = " ".join(f"{int(100*p):4d}% " for p in LEVELS)
    print("Coverage at each nominal level — should match the header\n")
    print(f"{'configuration':26s} {hdr}   ANEES")
    print("-" * 74)

    base = FusionConfig()
    E = np.concatenate([collect(p, base)[0] for p in P])
    S = np.concatenate([collect(p, base)[1] for p in P])
    report("as built", E, S)

    print("\nSweeping the two process-noise terms that dominate during a blackout:")
    best = None
    for qa in (1.2, 2.0, 3.0, 4.5):
        for qy in (0.35, 0.6, 1.0, 1.6):
            cfg = FusionConfig(q_accel=qa, q_yawrate=qy)
            Es, Ss = [], []
            for p in P:
                e, s = collect(p, cfg); Es.append(e); Ss.append(s)
            E, S = np.concatenate(Es), np.concatenate(Ss)
            anees = float(np.mean((E / np.maximum(S, 1e-9)) ** 2) / 2.0)
            # score by how far the whole curve sits from nominal
            miss = np.mean([abs(np.mean(E <= k_for(p) * S) - p) for p in LEVELS])
            if best is None or miss < best[0]:
                best = (miss, qa, qy, E, S)
            print(f"  q_accel={qa:<4} q_yawrate={qy:<4} " +
                  " ".join(f"{100*np.mean(E<=k_for(p)*S):5.0f}%" for p in LEVELS) +
                  f"   ANEES {anees:6.2f}  miss {miss:.3f}")
    print(f"\nbest: q_accel={best[1]}, q_yawrate={best[2]}")
    print(f"{'configuration':26s} {hdr}   ANEES")
    print("-" * 74)
    report("calibrated", best[3], best[4])


if __name__ == "__main__":
    main()
