"""Export a ride's route as a plain lat/lon polyline for the phone.

The phone does not need OSM, a road graph, or any of the pipeline. It needs one
list of points: the road it is on. This writes that.

    python3 scripts/export_route.py rides/<ride>/ app/src/main/assets/route.json
"""
from __future__ import annotations
import json, os, sys
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dhruva.osm_road import OsmNetwork, road_from_osm, to_xy
from dhruva.gpsclean import clean_for_truth

OSM = os.environ.get("OSM", "data/osm/wide_ways_v2.json")


def export(ride_dir, out_path, pad_m=3000.0):
    M = pd.read_csv("data/landmarks/campus_landmarks.csv")
    lat0, lon0 = float(M.lat.iloc[0]), float(M.lon.iloc[0])
    net = OsmNetwork(OSM, lat0, lon0)

    L = pd.read_csv(os.path.join(ride_dir, "Location.csv"))
    t = L["seconds_elapsed"].to_numpy(float)
    gps = to_xy(L["latitude"].to_numpy(float), L["longitude"].to_numpy(float), lat0, lon0)
    gps, t = clean_for_truth(gps, t)

    xy, _ = road_from_osm(net, gps)
    from evaluate_system import _extend
    xy = _extend(np.asarray(xy, float), pad_m)

    # back to lat/lon so the phone needs no projection constants
    m = 111320.0
    k = np.cos(np.radians(lat0))
    lat = lat0 + xy[:, 1] / m
    lon = lon0 + xy[:, 0] / (m * k)
    pts = [[round(float(a), 7), round(float(b), 7)] for a, b in zip(lat, lon)]

    seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    doc = {
        "name": os.path.basename(ride_dir.rstrip("/")),
        "source": OSM,
        "length_m": round(float(seg.sum()), 1),
        "n_points": len(pts),
        "points": pts,
    }
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(doc, f)
    print(f"{out_path}: {len(pts)} points, {doc['length_m']:.0f} m "
          f"({os.path.getsize(out_path)/1024:.0f} KB)")
    return doc


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    export(sys.argv[1], sys.argv[2])
