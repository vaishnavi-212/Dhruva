#!/usr/bin/env python3
"""ISRO requirement 3 — HMM map matching over a branching road graph.

The critical difference from every earlier evaluation: `road_from_osm` is handed
the run's own GPS to choose which ways were traversed. The HMM is given ONLY the
last known fix plus IMU-derived distance and turn per step, and must choose the
route itself — including at every junction.

    python scripts/eval_hmm.py
"""
from __future__ import annotations
import glob, os, sys, time
import numpy as np, pandas as pd, torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dhruva.osm_road import to_xy
from dhruva.roadgraph import RoadGraph
from dhruva.hmm import decode, HmmConfig
from dhruva.gpsclean import clean_for_truth
from dhruva import align
from evaluate_system import speed_from_imu

RUNS = sorted(glob.glob("rides/bike/*/")) + sorted(glob.glob("rides/aug31/*/"))


def observations(d, ck, step_m=10.0):
    """IMU only: metres travelled and heading change per step."""
    t, v = speed_from_imu(d, ck)
    dt = float(np.median(np.diff(t)))
    A = pd.read_csv(os.path.join(d, "Accelerometer.csv"))
    G = pd.read_csv(os.path.join(d, "Gyroscope.csv"))
    Gr = pd.read_csv(os.path.join(d, "Gravity.csv"))
    g = np.arange(0, min(A["seconds_elapsed"].iloc[-1], G["seconds_elapsed"].iloc[-1]), 0.05)
    rs = lambda df: np.column_stack([np.interp(g, df["seconds_elapsed"], df[c])
                                     for c in ("x", "y", "z")])
    R = align.estimate_tilt(rs(A) + rs(Gr), 20.0)
    wz = np.interp(t, g, align.apply(R, rs(G))[:, 2])
    dist = np.cumsum(np.clip(v, 0, None)) * dt
    head = np.concatenate([[0.0], np.cumsum(wz[:-1] * np.diff(t))])
    marks = np.arange(0, dist[-1], step_m)
    tm = np.interp(marks, dist, t)
    hm = np.interp(tm, t, head)
    return np.full(len(marks) - 1, step_m), np.diff(hm), tm


def main():
    M = pd.read_csv("data/landmarks/campus_landmarks.csv")
    lat0, lon0 = M.lat.iloc[0], M.lon.iloc[0]
    graph = RoadGraph("data/osm/wide_ways.json", lat0, lon0)
    print("road graph:", graph.stats(), "\n")
    print("HMM given ONLY the last fix + IMU. It chooses the route, including junctions.\n")
    print(f"{'run':30s} {'len m':>7} {'cross-track':>12} {'final err':>10} {'junctions':>10} {'s':>6}")
    print("-" * 84)
    CT, FE = [], []
    for d in RUNS:
        nm = os.path.basename(d[:-1])
        ck = torch.load(f"checkpoints/loro/{nm}.pt", map_location="cpu", weights_only=False)
        L = pd.read_csv(os.path.join(d, "Location.csv"))
        lt = L["seconds_elapsed"].to_numpy(float)
        gps = to_xy(L["latitude"].to_numpy(float), L["longitude"].to_numpy(float), lat0, lon0)
        gps, lt = clean_for_truth(gps, lt)
        dists, turns, tm = observations(d, ck)
        t0 = time.perf_counter()
        path, edges, junc = decode(graph, gps[0], dists, turns, HmmConfig())
        wall = time.perf_counter() - t0
        if path is None:
            print(f"{nm[:30]:30s}  no seed state"); continue
        truth = np.column_stack([np.interp(tm[:len(path)], lt, gps[:, 0]),
                                 np.interp(tm[:len(path)], lt, gps[:, 1])])
        n = min(len(path), len(truth))
        err = np.linalg.norm(path[:n] - truth[:n], axis=1)
        CT.append(np.median(err)); FE.append(err[-1])
        true_len = np.linalg.norm(np.diff(gps, axis=0), axis=1).sum()
        print(f"{nm[:30]:30s} {true_len:7.0f} {np.median(err):11.1f}m {err[-1]:9.1f}m "
              f"{junc:10d} {wall:6.1f}")
    print("-" * 84)
    print(f"median cross-track {np.median(CT):.1f} m   median final error {np.median(FE):.1f} m")


if __name__ == "__main__":
    main()
