#!/usr/bin/env python3
"""Golden series for the phone's route guard (RouteGuard.kt must raise OFF ROUTE at the same sample).

One recorded ride (94CC_6RQ 30 Aug 16:39, mounted phone), exactly as scripts/eval_route_guard.py
scores it: a WRONG destination (the rider leaves that route; Python alarms 21 m later) and the TRUE
destination (Python stays quiet). Per 10 Hz sample: time, distance since the cut (AI speed), gyro
heading since the cut, and Python's mismatch and alarm.

    python scripts/make_golden_route_guard.py   ->  data/golden_route_guard.json
"""
import json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_route_guard as RG
import eval_causal_features as EC
from train_eval_phone_features import frame_features
from dhruva.roadgraph import RoadGraph
from dhruva.mapmatch import RoadPolyline
from dhruva.osm_road import to_xy
from dhruva.gpsclean import clean_for_truth

RIDE = "rides/bike/94CC_6RQ-2026-08-30_16-39-43"
CASES = {"wrong": 9047445013, "true": 3846582275}


def series(route_xy, t, v, wz, i0, i1):
    """RG.evaluate's detector, returning every intermediate the phone must reproduce."""
    road = RoadPolyline(route_xy, 1.0)
    dt = float(np.median(np.diff(t)))
    arc = np.cumsum(v[i0:i1]) * dt
    psi_r = np.unwrap(road.heading); psi_r = psi_r - psi_r[0]
    psi_m = np.cumsum(wz[i0:i1]) * dt
    offs = np.linspace(-1.0, 1.0, 21)
    mismatch = np.zeros(len(arc))
    for k, s in enumerate(arc):
        tol = RG.TOL_M + RG.TOL_FRAC * s; win = 2.0 * tol + RG.WIN_EXTRA_M
        if s < win: continue
        kb = int(np.searchsorted(arc, s - win))
        dm = psi_m[k] - psi_m[kb]
        so = s + offs * tol
        dr = (np.interp(np.clip(so, 0, road.length), road.s, psi_r) - np.interp(np.clip(so - win, 0, road.length), road.s, psi_r))
        mismatch[k] = np.min(np.abs(RG.wrap(dm - dr)))
    k_alarm = RG.first_sustained(np.degrees(mismatch) > RG.ALARM_DEG, t[i0:i1], RG.ALARM_HOLD_S)
    return arc, psi_m, np.degrees(mismatch), k_alarm


def main():
    S = pd.read_csv("data/landmarks/campus_landmarks.csv"); lat0, lon0 = float(S.lat.iloc[0]), float(S.lon.iloc[0])
    G = RoadGraph("data/osm/wide_ways_v2.json", lat0, lon0)
    net, mean, std, _ = RG.model_for(RIDE)
    t, acc, gyro, grav = EC.load(RIDE)
    v = EC.speeds(frame_features("replica", acc, gyro, grav), t, net, mean, std, causal=True)
    L = pd.read_csv(os.path.join(RIDE, "Location.csv")); lt = L.seconds_elapsed.to_numpy(float)
    gps = to_xy(L.latitude.to_numpy(float), L.longitude.to_numpy(float), lat0, lon0); gps, lt = clean_for_truth(gps, lt)
    keep = (t >= lt[0]) & (t <= lt[-1])
    truth = np.column_stack([np.interp(t, lt, gps[:, 0]), np.interp(t, lt, gps[:, 1])])
    cum = np.r_[0, np.cumsum(np.linalg.norm(np.diff(truth, axis=0), axis=1))]
    i0 = int(np.searchsorted(t, RG.CUT_S)); i1 = int(np.flatnonzero(keep)[-1]) + 1
    gu = grav / np.maximum(np.linalg.norm(grav, axis=1, keepdims=True), 1e-9)
    wz = np.einsum("ij,ij->i", gyro, gu)
    j = max(int(np.searchsorted(cum, cum[i0] - 15.0)), 0); d = truth[i0] - truth[j]
    planner = RG.Planner(G, truth[i0], np.arctan2(d[1], d[0]))
    out = []
    for name, node in CASES.items():
        P = planner.route_to_node(node)
        arc, psi, mis, k = series(P, t, v, wz, i0, i1)
        k_dev, k_alarm, _, _ = RG.evaluate(P, t, truth, cum, v, wz, i0, i1)
        assert k == k_alarm, (k, k_alarm)
        out.append(dict(name=name, route_x=P[:, 0].tolist(), route_y=P[:, 1].tolist(), t=(t[i0:i1] - t[i0]).tolist(),
                        arc=arc.tolist(), psi=psi.tolist(), mismatch_deg=mis.tolist(),
                        alarm_index=(None if k is None else int(k)), alarm_arc=(None if k is None else float(arc[k]))))
        print(f"{name}: {len(arc)} samples, route {RoadPolyline(P, 1.0).length:.0f} m, alarm "
              + (f"at sample {k} ({arc[k]:.0f} m after the cut)" if k is not None else "none"))
    json.dump(dict(ride=os.path.basename(RIDE), cases=out), open("data/golden_route_guard.json", "w"))
    print("wrote data/golden_route_guard.json")


if __name__ == "__main__":
    main()
