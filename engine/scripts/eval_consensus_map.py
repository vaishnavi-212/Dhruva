#!/usr/bin/env python3
"""Should a road learned by one rider go straight into the shared map? Measured with our own rides.

The system design promotes a road from a phone's personal map to the shared map only when 3 or more
contributions agree. This tests that rule on the 13 campus rides that cover the back gate <-> LHC
corridor end to end, leave-one-out: every ride in turn is the NEW rider, and the map is built from
the others.

  1 contributor        the road exactly as one earlier ride recorded it
  3 contributors       the point-by-point MEDIAN of three rides (the consensus rule)
  faulty contributor   one ride with a simulated GPS fault: a 250 m stretch displaced 40 m sideways,
                       the kind of error multipath near buildings produces. Simulated, and labelled so
  3 incl. faulty, mean     naive merging: average everything that arrives
  3 incl. faulty, median   the consensus rule with the same faulty contribution in it
  all others, median   the crowd

Scored on the new rider's own GPS (hidden from the map):
  shape    how far the new rider's track sits from the map road, median and 95th percentile
  drift    ride the map road at the new rider's true speed; worst error at 300, 450 and 600 m ridden,
           as % of distance (the map's share of navigation error; checkpoints stay inside every map)

Honest limits: the 13 rides were recorded on one phone by a few riders, so "contributors" here are
separate rides, not separate phones; and the fault is injected, not observed.

    python scripts/eval_consensus_map.py   ->  results/consensus_map.csv, results/consensus_map.png
"""
from __future__ import annotations
import glob, os, sys
import numpy as np, pandas as pd
from scipy.spatial import cKDTree
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dhruva.osm_road import to_xy
from dhruva.gpsclean import clean_for_truth
from dhruva.mapmatch import RoadPolyline

GATE, LHC = (15.3672, 75.1269), (15.3706, 75.1219)
LAT0, LON0 = GATE
FAULT_M, FAULT_LEN_M, FAULT_RAMP_M = 40.0, 250.0, 40.0
DRAWS, SEED = 20, 26168


def load_corridor(folder):
    """The ride's GPS between the back gate and LHC, oriented gate -> LHC, in local metres."""
    L = pd.read_csv(os.path.join(folder, "Location.csv"))
    if len(L) < 30:
        return None
    xy = to_xy(L.latitude.to_numpy(float), L.longitude.to_numpy(float), LAT0, LON0)
    xy, _ = clean_for_truth(xy, L.seconds_elapsed.to_numpy(float))
    g = to_xy(np.array([GATE[0]]), np.array([GATE[1]]), LAT0, LON0)[0]
    h = to_xy(np.array([LHC[0]]), np.array([LHC[1]]), LAT0, LON0)[0]
    dg, dh = np.linalg.norm(xy - g, axis=1), np.linalg.norm(xy - h, axis=1)
    ig, ih = int(np.argmin(dg)), int(np.argmin(dh))
    if dg[ig] > 60 or dh[ih] > 60:
        return None
    seg = xy[ig:ih + 1] if ig < ih else xy[ih:ig + 1][::-1]
    return RoadPolyline(seg, 2.0).xy if len(seg) > 10 else None


def length(xy):
    return float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum())


def consensus(tracks, how="median"):
    """Point-by-point median (or mean) of several tracks, stationed along the most typical one."""
    if len(tracks) == 1:
        return tracks[0]
    trees = [cKDTree(t) for t in tracks]
    # the most typical track sets the stations (the one closest, on average, to all the others)
    score = [np.mean([tr.query(t)[0].mean() for tr in trees]) for t in tracks]
    ref = tracks[int(np.argmin(score))]
    agg = np.median if how == "median" else np.mean
    for _ in range(2):                                   # re-station once on the result
        pts = np.stack([t[tr.query(ref)[1]] for t, tr in zip(tracks, trees)])   # (n, stations, 2)
        ref = agg(pts, axis=0)
        ref = RoadPolyline(ref, 2.0).xy
    return ref


def with_fault(xy, rng):
    """A simulated GPS fault: a 250 m stretch pushed 40 m sideways, with 40 m ramps in and out."""
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
    a = rng.uniform(0.25, 0.55) * s[-1]
    t = np.gradient(xy, axis=0); n = np.column_stack([-t[:, 1], t[:, 0]])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
    w = np.clip(np.minimum((s - a) / FAULT_RAMP_M, (a + FAULT_LEN_M - s) / FAULT_RAMP_M), 0, 1)
    return xy + n * (FAULT_M * w)[:, None]


CHECK_M = (300.0, 450.0, 600.0)     # all inside every map: rides are ~830 m, maps a little shorter at the ends


def score(map_xy, rider):
    """Shape: distance from the new rider's track to the map road. Drift: ride the map at the rider's
    true distance and take the worst error at 300, 450 and 600 m, as % of the distance ridden.
    Checkpoints, not the end: a map built from other rides is a few metres shorter at its ends, and
    running off the end would score the clipping, not the map."""
    road = RoadPolyline(map_xy, 1.0)
    d = cKDTree(road.xy).query(rider)[0]
    cum = np.r_[0, np.cumsum(np.linalg.norm(np.diff(rider, axis=0), axis=1))]
    s0 = road.project(rider[0])
    worst = 0.0
    for c in CHECK_M:
        if s0 + c > road.length - 5 or c > cum[-1]:
            continue
        i = int(np.searchsorted(cum, c))
        worst = max(worst, 100 * float(np.linalg.norm(road.at(np.array([s0 + c]))[0][0] - rider[i])) / c)
    return float(np.median(d)), float(np.percentile(d, 95)), worst


def main():
    folders = sorted(glob.glob("rides/bike/*/")) + sorted(glob.glob("rides/aug31/*/")) + \
              sorted(glob.glob("rides/app_DhruvaRun_2026-09-1*/"))
    rides = {os.path.basename(f.rstrip("/")): t for f in folders if (t := load_corridor(f)) is not None}
    names = list(rides)
    print(f"{len(names)} rides cover the corridor: median length {np.median([length(rides[n]) for n in names]):.0f} m")
    rng = np.random.default_rng(SEED)
    rows = []
    for held in names:
        rider = rides[held]
        others = [rides[n] for n in names if n != held]
        for k in range(DRAWS):
            pick = rng.choice(len(others), 3, replace=False)
            good1, good3 = [others[pick[0]]], [others[i] for i in pick]
            bad = with_fault(others[pick[2]], rng)
            scen = {
                "1 contributor": consensus(good1),
                "3 contributors": consensus(good3),
                "faulty contributor alone": bad,
                "3 incl. faulty, averaged": consensus([others[pick[0]], others[pick[1]], bad], "mean"),
                "3 incl. faulty, median": consensus([others[pick[0]], others[pick[1]], bad], "median"),
            }
            if k == 0:
                scen["all others, median"] = consensus(others)
            for name, m in scen.items():
                med, p95, drift = score(m, rider)
                rows.append(dict(held_out=held, draw=k, scenario=name, shape_median_m=med, shape_p95_m=p95, drift_pct=drift))
    D = pd.DataFrame(rows)
    os.makedirs("results", exist_ok=True); D.to_csv("results/consensus_map.csv", index=False)
    order = ["1 contributor", "3 contributors", "all others, median", "faulty contributor alone",
             "3 incl. faulty, averaged", "3 incl. faulty, median"]
    print(f"\n{'map built from':28s} {'shape median':>13s} {'shape p95':>10s} {'drift median':>13s} {'drift p90':>10s}")
    for s in order:
        X = D[D.scenario == s]
        print(f"{s:28s} {X.shape_median_m.median():11.1f} m {X.shape_p95_m.median():8.1f} m "
              f"{X.drift_pct.median():11.2f} % {np.percentile(X.drift_pct, 90):8.2f} %")
    print("saved results/consensus_map.csv")


if __name__ == "__main__":
    main()
