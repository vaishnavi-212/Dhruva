"""Pre-flight: is this ride inside our downloaded road map?

A ride that leaves the OSM extract fails SILENTLY — the 1-D tracker follows roads
the rider never took and reports a model-sized error (25.3% on 5 Sept) that looks
like the model broke. It did not. The map was missing.

Run this BEFORE the pipeline, on every new ride:

    python3 scripts/check_coverage.py rides/<ride>/Location.csv

Exit code 0 = safe to process, 1 = refetch OSM first.
"""
from __future__ import annotations
import json, math, sys, glob, os
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

OSM = os.environ.get("OSM", "data/osm/wide_ways_v2.json")

# Thresholds, set from the 17 rides we have already validated:
#   every good ride sits at p90 <= 58 m, max <= 94 m
#   the ride that failed sat at p90 337 m, max 465 m
WARN_P90, FAIL_P90 = 80.0, 150.0
WARN_MAX, FAIL_MAX = 150.0, 300.0


def _road_points(path):
    d = json.load(open(path))
    pts = [(g["lat"], g["lon"])
           for e in d["elements"]
           if e.get("type") == "way" and "geometry" in e
           for g in e["geometry"]]
    if not pts:
        sys.exit(f"no ways in {path}")
    return np.asarray(pts, float)


def _gps(path):
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return np.empty(0), np.empty(0)
    c = {x.lower(): x for x in df.columns}
    la = c.get("latitude") or c.get("lat")
    lo = c.get("longitude") or c.get("lon")
    if not (la and lo):
        return np.empty(0), np.empty(0)
    lat = df[la].to_numpy(float)
    lon = df[lo].to_numpy(float)
    ok = np.isfinite(lat) & np.isfinite(lon)
    return lat[ok], lon[ok]


def check(csv_path, osm_path=OSM, verbose=True):
    A = _road_points(osm_path)
    lat, lon = _gps(csv_path)
    if len(lat) < 10:
        if verbose:
            print(f"\n  ride     {csv_path}\n  SKIP — only {len(lat)} usable fixes")
        return dict(p50=0.0, p90=0.0, max=0.0, margin=0.0,
                    fail=False, warn=False, skipped=True)

    m, k = 111320.0, math.cos(math.radians(float(lat.mean())))
    tree = cKDTree(np.c_[A[:, 1] * m * k, A[:, 0] * m])
    dist, _ = tree.query(np.c_[lon * m * k, lat * m])

    p50, p90, mx = (float(np.percentile(dist, 50)),
                    float(np.percentile(dist, 90)),
                    float(dist.max()))
    # distance to the map's bounding box, so "outside the box" is named as such
    S, N = A[:, 0].min(), A[:, 0].max()
    W, E = A[:, 1].min(), A[:, 1].max()
    margin = min((lat.min() - S) * m, (N - lat.max()) * m,
                 (lon.min() - W) * m * k, (E - lon.max()) * m * k)

    bad = p90 > FAIL_P90 or mx > FAIL_MAX or margin < 0
    warn = not bad and (p90 > WARN_P90 or mx > WARN_MAX or margin < 200)

    if verbose:
        name = os.path.basename(os.path.dirname(csv_path)) or csv_path
        print(f"\n  ride     {name}")
        print(f"  map      {osm_path}")
        print(f"  fixes    {len(lat)}")
        print(f"  gap to nearest mapped road   p50 {p50:6.1f} m   "
              f"p90 {p90:6.1f} m   max {mx:6.1f} m")
        print(f"  margin inside the map box    {margin:6.0f} m")
        if bad:
            print("\n  FAIL — this ride is outside the downloaded map.")
            print("         Do NOT trust any drift number from it.")
            print("         Refetch OSM over a bbox covering this ride, then re-run.")
        elif warn:
            print("\n  WARN — close to the edge of what we have mapped.")
            print("         Results are probably fine; widen the map before the next ride.")
        else:
            print("\n  OK — fully covered. Safe to process.")
    return dict(p50=p50, p90=p90, max=mx, margin=margin, fail=bad, warn=warn)


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        args = sorted(glob.glob("rides/**/Location.csv", recursive=True))
        print(f"no path given — checking all {len(args)} rides")
    bad = 0
    for a in args:
        if check(a)["fail"]:
            bad += 1
    print()
    sys.exit(1 if bad else 0)
