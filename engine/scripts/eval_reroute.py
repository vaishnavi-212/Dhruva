#!/usr/bin/env python3
"""After a wrong turn WITHOUT GPS: which road did the rider take? Re-bind there.

eval_route_guard.py raises OFF ROUTE when the gyro's heading change cannot be matched to the planned
route. For every wrong-destination case it caught, this asks the next question:

  1. Candidate junctions: graph nodes on the planned route that the rider could have left it at,
     given the alarm and the along-track uncertainty.
  2. Candidate roads: at each, every branch the planned route did NOT take (and not the way in).
  3. For each candidate, build the path "planned route up to the junction, then that branch, then
     the straightest road onward". Compare the heading change the gyro measured between a point safely
     BEFORE the junction and the alarm with the heading change along that candidate path over the same
     estimated distance. Best match wins; the dot is re-bound to that path.

A choice is FORCED when only one candidate road existed -- the gyro decided nothing -- and is
reported separately from choices the gyro actually made.

Scored against the hidden GPS: was it the road the rider took, and how far off is the dot at the
alarm and 50 m / 100 m later -- re-bound vs left on the planned route.

    python scripts/eval_route_guard.py && python scripts/eval_reroute.py
"""
from __future__ import annotations
import os, sys
import numpy as np, pandas as pd
from scipy.spatial import cKDTree
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_route_guard as RG
import eval_causal_features as EC
from train_eval_phone_features import frame_features
from dhruva.roadgraph import RoadGraph
from dhruva.mapmatch import RoadPolyline
from dhruva.osm_road import to_xy
from dhruva.gpsclean import clean_for_truth

LEAD_M = 20.0            # start the gyro comparison this far before the earliest plausible junction position


def wrapd(x):
    return np.degrees(np.angle(np.exp(1j * np.radians(x))))


def oriented_edge(G, ei, fwd):
    _, xy, _ = G.edge_points(ei, 2.0)
    return xy if fwd else xy[::-1]


def straightest(G, node, ei, fwd, length=300.0):
    """Follow a branch from `node`, then keep taking the straightest road onward."""
    parts = [oriented_edge(G, ei, fwd)]
    total = float(np.linalg.norm(np.diff(parts[0], axis=0), axis=1).sum())
    u, v, _ = G.edges[ei]
    cur, prev_edge = (v if fwd else u), ei
    while total < length:
        last = parts[-1]
        if len(last) < 2:
            break
        h = np.degrees(np.arctan2(*(last[-1] - last[-2])[::-1]))
        best = None
        for e2, f2 in G.adj.get(cur, []):
            if e2 == prev_edge:
                continue
            seg = oriented_edge(G, e2, f2)
            if len(seg) < 2:
                continue
            h2 = np.degrees(np.arctan2(*(seg[min(3, len(seg) - 1)] - seg[0])[::-1]))
            turn = abs(wrapd(h2 - h))
            if best is None or turn < best[0]:
                best = (turn, e2, f2, seg)
        if best is None:
            break
        parts.append(best[3]); total += float(np.linalg.norm(np.diff(best[3], axis=0), axis=1).sum())
        u2, v2, _ = G.edges[best[1]]
        cur, prev_edge = (v2 if best[2] else u2), best[1]
    P = np.concatenate(parts)
    return P[np.r_[True, np.linalg.norm(np.diff(P, axis=0), axis=1) > 1e-6]]


def heading_at(road, s, span=10.0):
    """Heading (deg) of a RoadPolyline around arc s, from points span/2 either side."""
    a = np.clip(s - span / 2, 0, road.length); b = np.clip(s + span / 2, 0, road.length)
    pa, pb = road.at(np.array([a]))[0][0], road.at(np.array([b]))[0][0]
    return np.degrees(np.arctan2(*(pb - pa)[::-1]))


def unwrapped_change(road, s0, s1, step=5.0):
    """Total heading change along a path between arcs s0 and s1 (deg), following every bend."""
    ss = np.arange(s0, max(s1, s0 + step), step)
    h = np.array([heading_at(road, s) for s in ss])
    return float(np.sum(wrapd(np.diff(h)))) if len(h) > 1 else 0.0


def main():
    S = pd.read_csv("data/landmarks/campus_landmarks.csv"); lat0, lon0 = float(S.lat.iloc[0]), float(S.lon.iloc[0])
    G = RoadGraph("data/osm/wide_ways_v2.json", lat0, lon0)
    cases = pd.read_csv("results/route_guard.csv")
    cases = cases[(cases.case == "wrong") & (cases.outcome == "caught")]
    rows = []
    for ride in cases.ride.unique():
        folder = next(f for f in (f"rides/bike/{ride}", f"rides/aug31/{ride}", f"rides/{ride}") if os.path.isdir(f))
        net, mean, std, _ = RG.model_for(folder)
        t, acc, gyro, grav = EC.load(folder)
        v = EC.speeds(frame_features("replica", acc, gyro, grav), t, net, mean, std, causal=True)
        L = pd.read_csv(os.path.join(folder, "Location.csv")); lt = L.seconds_elapsed.to_numpy(float)
        gps = to_xy(L.latitude.to_numpy(float), L.longitude.to_numpy(float), lat0, lon0); gps, lt = clean_for_truth(gps, lt)
        keep = (t >= lt[0]) & (t <= lt[-1])
        truth = np.column_stack([np.interp(t, lt, gps[:, 0]), np.interp(t, lt, gps[:, 1])])
        cum = np.r_[0, np.cumsum(np.linalg.norm(np.diff(truth, axis=0), axis=1))]
        dt = float(np.median(np.diff(t))); i0 = int(np.searchsorted(t, RG.CUT_S)); i1 = int(np.flatnonzero(keep)[-1]) + 1
        gu = grav / np.maximum(np.linalg.norm(grav, axis=1, keepdims=True), 1e-9)
        wz = np.einsum("ij,ij->i", gyro, gu)
        j = max(int(np.searchsorted(cum, cum[i0] - 15.0)), 0); d = truth[i0] - truth[j]
        planner = RG.Planner(G, truth[i0], np.arctan2(d[1], d[0]))
        arc = np.cumsum(v[i0:i1]) * dt                                   # estimated distance since the cut (no GPS)
        psi_m = np.degrees(np.cumsum(wz[i0:i1]) * dt)                    # gyro heading since the cut (no GPS)
        seg_truth = truth[i0:i1]; s_true = cum[i0:i1] - cum[i0]
        for _, c in cases[cases.ride == ride].iterrows():
            node = int(c.dest_node)
            P = planner.route_to_node(node)
            if P is None:
                continue
            road = RoadPolyline(P, 1.0)
            k_alarm = min(int(np.searchsorted(s_true, c.alarm_m)), len(arc) - 1)
            s_alarm = arc[k_alarm]
            tol = RG.TOL_M + RG.TOL_FRAC * s_alarm
            steps, n = [], node
            while n != planner.first:
                n0, ei, fwd = planner.prev[n]; steps.append((n0, ei)); n = n0
            steps.reverse()
            # junctions the rider could have left at: up to the detector's window before the alarm
            lo, hi = s_alarm - (2 * tol + RG.WIN_EXTRA_M) - tol, s_alarm + tol
            cands = []
            for jn, next_edge in steps:
                s_j = road.project(G.node_xy[jn])
                if G.junction_degree(jn) < 3 or not (lo <= s_j <= hi):
                    continue
                s_start = max(s_j - tol - LEAD_M, 0.0)
                kb = int(np.searchsorted(arc, s_start))
                ka = max(k_alarm, min(int(np.searchsorted(arc, s_j + tol + 30.0)), len(arc) - 1))
                gyro_change = psi_m[ka] - psi_m[kb]
                prefix = road.xy[: max(int(np.searchsorted(road.s, s_j)), 2)]
                for e2, f2 in G.adj.get(jn, []):
                    if e2 == next_edge:
                        continue
                    br = straightest(G, jn, e2, f2, 300.0)
                    if len(br) < 3:
                        continue
                    newP = np.concatenate([prefix, br]); newP = newP[np.r_[True, np.linalg.norm(np.diff(newP, axis=0), axis=1) > 1e-6]]
                    cand = RoadPolyline(newP, 1.0)
                    first_turn = wrapd(heading_at(cand, s_j + 15) - heading_at(cand, max(s_j - 15, 0)))
                    if abs(first_turn) > 160:                           # the road we came in on
                        continue
                    path_change = unwrapped_change(cand, s_start, arc[ka])
                    cands.append(dict(cost=abs(wrapd(gyro_change - path_change)), jn=jn, s_j=s_j, edge=e2, road=cand,
                                      gyro=gyro_change, path=path_change))
            if not cands:
                rows.append(dict(ride=ride, dest_node=node, result="no candidate")); continue
            best = min(cands, key=lambda x: x["cost"])
            n_roads = len(cands)
            # truth (GPS, hidden from the method): the junction the rider actually left the route at, and the road taken
            dist_route = cKDTree(road.xy).query(seg_truth)[0]
            k_leave = int(np.argmax(dist_route > 10.0)) if np.any(dist_route > 10.0) else None
            truth_edge, truth_jn = None, None
            if k_leave is not None:
                p_leave = seg_truth[max(k_leave - 1, 0)]
                truth_jn = min(steps, key=lambda st: np.linalg.norm(G.node_xy[st[0]] - p_leave))[0]
                after = seg_truth[(s_true >= s_true[k_leave] + 10) & (s_true <= s_true[k_leave] + 60)]
                scored = []
                for e3, f3 in G.adj.get(truth_jn, []):
                    br = straightest(G, truth_jn, e3, f3, 80.0)
                    if len(after) and len(br) > 2:
                        scored.append((float(np.mean(cKDTree(br).query(after)[0])), e3))
                truth_edge = min(scored)[1] if scored else None
            correct = best["jn"] == truth_jn and best["edge"] == truth_edge
            out = dict(ride=ride, dest_node=node, mount="pocket" if "2026-08-31" in folder else "mounted",
                       candidates=n_roads, forced=n_roads == 1, correct=bool(correct),
                       gyro_change=best["gyro"], path_change=best["path"])
            for extra in (0.0, 50.0, 100.0):
                k = int(np.searchsorted(s_true, c.alarm_m + extra))
                if k >= len(arc):
                    continue
                out[f"rebound_{int(extra)}"] = float(np.linalg.norm(best["road"].at(np.array([arc[k]]))[0][0] - seg_truth[k]))
                out[f"stayed_{int(extra)}"] = float(np.linalg.norm(road.at(np.array([arc[k]]))[0][0] - seg_truth[k]))
            rows.append(out)
            print(f"{ride[:28]:28s} {out['mount']:7s} {n_roads} road(s){' FORCED' if out['forced'] else '       '} | gyro {best['gyro']:+5.0f}° vs path {best['path']:+5.0f}° | "
                  f"{'CORRECT' if correct else 'wrong  '} | at alarm {out.get('rebound_0', np.nan):4.0f}/{out.get('stayed_0', np.nan):4.0f} m"
                  f" | +50 m {out.get('rebound_50', np.nan):4.0f}/{out.get('stayed_50', np.nan):4.0f} m | +100 m {out.get('rebound_100', np.nan):4.0f}/{out.get('stayed_100', np.nan):4.0f} m  (re-bound/stayed)", flush=True)
    D = pd.DataFrame(rows)
    if "forced" in D:   # "no candidate" rows have no forced/correct values; keep the columns boolean
        D["forced"] = D["forced"].fillna(False).astype(bool); D["correct"] = D["correct"].fillna(False).astype(bool)
    D.to_csv("results/reroute.csv", index=False)
    for mount in ("mounted", "pocket"):
        M = D[D["mount"] == mount] if "mount" in D else D.iloc[0:0]
        if not len(M):
            continue
        real = M[~M.forced]
        print(f"\n== {mount}: {len(M)} caught wrong turns | correct {int(M.correct.sum())}/{len(M)} "
              f"(gyro-decided {int(real.correct.sum())}/{len(real)}, forced {int(M.forced.sum())})")
        for extra in (0, 50, 100):
            col_r, col_s = f"rebound_{extra}", f"stayed_{extra}"
            if col_r in M:
                print(f"   +{extra:3d} m after the alarm: re-bound median {M[col_r].median():4.0f} m vs left on planned route {M[col_s].median():4.0f} m")
    print("saved results/reroute.csv")


if __name__ == "__main__":
    main()
