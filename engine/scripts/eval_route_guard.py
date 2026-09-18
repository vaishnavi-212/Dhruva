#!/usr/bin/env python3
"""Route guard: navigate to a DESTINATION through a GNSS blackout, and notice a wrong turn.

The reviewer's point (internal round, 12 Sept): binding to the nearest road only says where you
are, never whether you are going the right way. This tests the fix on recorded rides:

  1. When GPS drops, plan the route from there to the destination on the OpenStreetMap road graph.
  2. Move the dot ALONG THAT ROUTE with the phone-style AI speed (look-back replica features).
  3. Watch the gyro. The route says how the heading should change with distance; if what the phone
     measures cannot be matched to the route within the along-track uncertainty, raise OFF ROUTE.

Each ride is tested twice:
  * TRUE destination (where the ride really ended) -- any alarm while the rider follows the planned
    route is a FALSE ALARM.
  * WRONG destinations: nodes whose planned route leaves the ridden path at a junction the rider
    actually passed. The alarm should fire soon after that junction; delay is measured in metres.

Ground truth comes from GPS (hidden from the detector): the rider has left the planned route when
the GPS track is more than 30 m from it for 5 s.

Only GNSS-free information is used after the cut: speed from the AI model, rotation from the gyro
(projected on the gravity sensor), and the planned route. The gyro's sign and bias are calibrated on
the 60 s BEFORE the cut, while GPS was still healthy -- which a phone can do too.

    python scripts/eval_route_guard.py
"""
from __future__ import annotations
import glob, heapq, os, sys
import numpy as np, pandas as pd, torch
from scipy.spatial import cKDTree
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_causal_features as EC
from train_eval_phone_features import frame_features
from dhruva.model import ResNet1D
from dhruva.roadgraph import RoadGraph
from dhruva.mapmatch import RoadPolyline
from dhruva.osm_road import to_xy
from dhruva.gpsclean import clean_for_truth

CUT_S = 60.0
DEV_M, DEV_HOLD_S = 30.0, 5.0          # ground truth: left the planned route
ALARM_DEG, ALARM_HOLD_S = 60.0, 2.0    # detector: heading-change mismatch that cannot be explained
TOL_M, TOL_FRAC = 15.0, 0.08           # along-track uncertainty searched when matching heading
WIN_EXTRA_M = 20.0                     # window = 2 x uncertainty + this
WRONG_PER_RIDE = 3


def model_for(folder):
    name = os.path.basename(folder)
    path = f"checkpoints/loro_phone_replica/{name}.pt"
    if not os.path.exists(path):
        path = "checkpoints/app/speed_app_replica.pt"      # 11 Sept rides: the app model never saw them
    ck = torch.load(path, map_location="cpu", weights_only=False)
    net = ResNet1D(in_ch=6, out_dim=1); net.load_state_dict(ck["model"]); net.eval()
    return net, np.asarray(ck["mean"]), np.asarray(ck["std"]), os.path.basename(path)


def nearest_on_edges(G, p):
    best = None
    for i in range(len(G.edges)):
        a, xy, hd = G.edge_points(i, 2.0)
        d = np.linalg.norm(xy - p, axis=1); j = int(np.argmin(d))
        if best is None or d[j] < best[0]:
            best = (float(d[j]), i, float(a[j]), float(hd[j]), a, xy)
    return best


def dijkstra(G, start_node, start_cost):
    dist, prev, pq = {start_node: start_cost}, {}, [(start_cost, start_node)]
    while pq:
        c, n = heapq.heappop(pq)
        if c > dist.get(n, np.inf):
            continue
        for ei, fwd in G.adj.get(n, []):
            u, v, _ = G.edges[ei]
            m = v if fwd else u
            nc = c + G.lengths[ei]
            if nc < dist.get(m, np.inf):
                dist[m] = nc; prev[m] = (n, ei, fwd); heapq.heappush(pq, (nc, m))
    return dist, prev


class Planner:
    """Shortest route from a point, leaving in the direction of travel, to any graph position."""

    def __init__(self, G, start_xy, heading):
        self.G = G
        _, self.es, self.sa, sh, self.a, self.xy = nearest_on_edges(G, start_xy)
        self.fwd = np.cos(sh - heading) >= 0
        u, v, _ = G.edges[self.es]
        self.first = v if self.fwd else u
        first_cost = G.lengths[self.es] - self.sa if self.fwd else self.sa
        self.dist, self.prev = dijkstra(G, self.first, first_cost)

    def route_to_node(self, node):
        if node not in self.dist:
            return None
        steps = []
        n = node
        while n != self.first:
            n0, ei, fwd = self.prev[n]; steps.append((ei, fwd)); n = n0
        steps.reverse()
        head = self.xy[self.a >= self.sa] if self.fwd else self.xy[self.a <= self.sa][::-1]
        parts = [head]
        for ei, fwd in steps:
            _, xy, _ = self.G.edge_points(ei, 2.0)
            parts.append(xy if fwd else xy[::-1])
        P = np.concatenate(parts)
        keep = np.r_[True, np.linalg.norm(np.diff(P, axis=0), axis=1) > 1e-6]
        P = P[keep]
        return P if len(P) >= 3 and np.linalg.norm(np.diff(P, axis=0), axis=1).sum() > 30 else None


def wrap(x):
    return np.angle(np.exp(1j * x))


def first_sustained(mask, t, hold):
    run_start = None
    for i, m in enumerate(mask):
        if m:
            run_start = t[i] if run_start is None else run_start
            if t[i] - run_start >= hold:
                return i
        else:
            run_start = None
    return None


def evaluate(route_xy, t, truth, cum, v, wz_cal, i0, i1):
    """Returns (true deviation index or None, alarm index or None, position error at end/deviation)."""
    road = RoadPolyline(route_xy, 1.0)
    fine = cKDTree(road.xy)
    dev_mask = fine.query(truth[i0:i1])[0] > DEV_M
    k_dev = first_sustained(dev_mask, t[i0:i1], DEV_HOLD_S)
    dt = float(np.median(np.diff(t)))
    arc = np.cumsum(v[i0:i1]) * dt                                  # along the planned route, no GPS
    psi_r = np.unwrap(road.heading); psi_r = psi_r - psi_r[0]
    psi_m = np.cumsum(wz_cal[i0:i1]) * dt                            # gyro heading, relative, drifts slowly
    # v2: compare heading CHANGE over a sliding window of travel, not total heading since the cut
    # (v1 let gyro drift accumulate: 40-90 deg off at its false alarms). The window is wider than twice
    # the along-track uncertainty, so every shifted copy still contains a route turn it straddles --
    # otherwise going straight through a junction where the route turns could never be noticed.
    offs = np.linspace(-1.0, 1.0, 21)
    mismatch = np.zeros(len(arc))
    for k, s in enumerate(arc):
        tol = TOL_M + TOL_FRAC * s
        win = 2.0 * tol + WIN_EXTRA_M
        if s < win:
            continue
        kb = int(np.searchsorted(arc, s - win))
        dm = psi_m[k] - psi_m[kb]
        so = s + offs * tol
        dr = (np.interp(np.clip(so, 0, road.length), road.s, psi_r)
              - np.interp(np.clip(so - win, 0, road.length), road.s, psi_r))
        mismatch[k] = np.min(np.abs(wrap(dm - dr)))
    k_alarm = first_sustained(np.degrees(mismatch) > ALARM_DEG, t[i0:i1], ALARM_HOLD_S)
    k_end = k_dev if k_dev is not None else len(arc) - 1
    pos = road.at(np.array([arc[k_end]]))[0][0]
    err = float(np.linalg.norm(pos - truth[i0 + k_end]))
    return k_dev, k_alarm, err, float(cum[i0 + k_end] - cum[i0])


def main():
    S = pd.read_csv("data/landmarks/campus_landmarks.csv"); lat0, lon0 = float(S.lat.iloc[0]), float(S.lon.iloc[0])
    G = RoadGraph("data/osm/wide_ways_v2.json", lat0, lon0)
    rides = [r.rstrip("/") for r in sorted(glob.glob("rides/bike/*/")) + sorted(glob.glob("rides/aug31/*/"))]
    rides += [f"rides/app_DhruvaRun_2026-09-11_{s}" for s in ("13-42-07", "13-46-20", "19-37-43", "19-41-28")]
    rows = []
    for folder in rides:
        net, mean, std, mname = model_for(folder)
        t, acc, gyro, grav = EC.load(folder)
        v = EC.speeds(frame_features("replica", acc, gyro, grav), t, net, mean, std, causal=True)
        L = pd.read_csv(os.path.join(folder, "Location.csv")); lt = L.seconds_elapsed.to_numpy(float)
        gps = to_xy(L.latitude.to_numpy(float), L.longitude.to_numpy(float), lat0, lon0); gps, lt = clean_for_truth(gps, lt)
        keep = (t >= lt[0]) & (t <= lt[-1])
        truth = np.column_stack([np.interp(t, lt, gps[:, 0]), np.interp(t, lt, gps[:, 1])])
        cum = np.r_[0, np.cumsum(np.linalg.norm(np.diff(truth, axis=0), axis=1))]
        dt = float(np.median(np.diff(t)))
        i0 = int(np.searchsorted(t, CUT_S)); i1 = int(np.flatnonzero(keep)[-1]) + 1

        # gyro turn rate about true vertical; sign and bias calibrated on GPS in the 60 s before the cut
        gu = grav / np.maximum(np.linalg.norm(grav, axis=1, keepdims=True), 1e-9)
        wz = np.einsum("ij,ij->i", gyro, gu)
        j = max(int(np.searchsorted(cum, cum[i0] - 15.0)), 0); d = truth[i0] - truth[j]; heading0 = np.arctan2(d[1], d[0])
        # v2: sign fixed by physics (the gravity sensor points up, so w.g is counter-clockwise yaw), and
        # no bias fit. v1 fitted sign and bias against a 1 Hz GPS course rate that was mostly noise
        # (correlation 0.01-0.18) and produced biases up to 3 deg/s; stops give no better (rider handling).
        wz_cal = wz

        planner = Planner(G, truth[i0], heading0)
        ridden = truth[i0:i1]
        # TRUE destination: the graph node nearest where the ride ended
        end_node = min(planner.dist, key=lambda n: np.linalg.norm(G.node_xy[n] - truth[i1 - 1]))
        cases = [("true", end_node)]
        # WRONG destinations: routes that follow the ridden path for a while, then leave it
        tree = cKDTree(ridden)
        cands = []
        for n, dn in planner.dist.items():
            if not (150.0 <= dn <= 600.0) or n == end_node:
                continue
            P = planner.route_to_node(n)
            if P is None:
                continue
            dist_to_ride = tree.query(P)[0]
            off = np.flatnonzero(dist_to_ride > DEV_M)
            if len(off) == 0 or off[0] < 25:          # must share >= ~50 m with the ridden path first
                continue
            leave_xy = P[off[0]]
            if np.linalg.norm(truth[i1 - 1] - leave_xy) < 60:
                continue                              # leaves right at the end of the recording: untestable
            cands.append((off[0], n, leave_xy))
        chosen, used = [], []
        for _, n, lx in sorted(cands, key=lambda c: c[0]):
            if all(np.linalg.norm(lx - u) > 60 for u in used):
                chosen.append(n); used.append(lx)
            if len(chosen) == WRONG_PER_RIDE:
                break
        cases += [("wrong", n) for n in chosen]

        for kind, node in cases:
            P = planner.route_to_node(node)
            if P is None:
                continue
            k_dev, k_alarm, err, along = evaluate(P, t, truth, cum, v, wz_cal, i0, i1)
            mount = "pocket" if "2026-08-31" in folder else "mounted"      # RESOURCES 7x: all five 31 Aug runs were pocketed
            rec = dict(ride=os.path.basename(folder), mount=mount, model=mname, case=kind, dest_node=int(node),
                       route_m=float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum()),
                       deviated=k_dev is not None, alarm=k_alarm is not None,
                       dev_m=float(cum[i0 + k_dev] - cum[i0]) if k_dev is not None else np.nan,
                       alarm_m=float(cum[i0 + k_alarm] - cum[i0]) if k_alarm is not None else np.nan,
                       pos_err_m=err, followed_m=along)
            if k_dev is None:
                rec["outcome"] = "false alarm" if k_alarm is not None else "quiet (correct)"
            elif k_alarm is None:
                rec["outcome"] = "missed"
            elif k_alarm + int(5.0 / dt) < k_dev:
                rec["outcome"] = "false alarm (early)"
            else:
                rec["outcome"] = "caught"
                rec["delay_m"] = float(cum[i0 + k_alarm] - cum[i0 + k_dev])
            rows.append(rec)
            print(f"{rec['ride'][:30]:30s} {kind:5s} route {rec['route_m']:4.0f} m | left route at "
                  f"{rec['dev_m']:5.0f} m | alarm at {rec['alarm_m']:5.0f} m | {rec['outcome']:20s}"
                  + (f" delay {rec['delay_m']:4.0f} m" if 'delay_m' in rec else "") + f" | pos err {err:4.0f} m after {along:4.0f} m", flush=True)

    D = pd.DataFrame(rows); D.to_csv("results/route_guard.csv", index=False)
    for mount in ("mounted", "pocket"):
        M = D[D.mount == mount]
        print(f"\n== {mount} phone ({M.ride.nunique()} rides)")
        for kind in ("true", "wrong"):
            X = M[M.case == kind]
            print(f"  {kind:5s} destinations: {len(X)} cases | " + ", ".join(f"{k} {int(n)}" for k, n in X.outcome.value_counts().items()))
        caught = M[M.outcome == "caught"]
        if len(caught):
            print(f"  caught wrong turns: median delay {caught.delay_m.median():.0f} m after leaving the route (max {caught.delay_m.max():.0f} m)")
        on = M[~M.deviated]
        print(f"  false alarms while on route: {int((on.outcome == 'false alarm').sum())} in {on.followed_m.sum()/1000:.1f} km followed;"
              f" position error on the followed route: median {on.pos_err_m.median():.0f} m")
    print("saved results/route_guard.csv")


if __name__ == "__main__":
    main()
