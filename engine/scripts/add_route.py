"""Add a ridden circuit to the app's routes.json -- real OSM road only, no padding.

    python3 scripts/add_route.py <ride_folder> <route_name> <path/to/routes.json>

Checks coverage first and refuses if the ride is outside the downloaded map.
Existing route with the same name is replaced.
"""
from __future__ import annotations
import json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dhruva.osm_road import OsmNetwork, road_from_osm, to_xy
from dhruva.gpsclean import clean_for_truth
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_coverage import check

OSM = os.environ.get("OSM", "data/osm/wide_ways_v2.json")

def main(ride, name, routes_path):
    loc = os.path.join(ride, "Location.csv")
    if not os.path.exists(loc) or os.path.getsize(loc) < 200:
        sys.exit(f"REFUSED: {loc} is missing or empty -- was the zip shared before Stop?")
    cov = check(loc, OSM)
    if cov.get("fail"):
        sys.exit("REFUSED: ride is outside the downloaded map (see above)")

    M = pd.read_csv("data/landmarks/campus_landmarks.csv")
    lat0, lon0 = float(M.lat.iloc[0]), float(M.lon.iloc[0])
    L = pd.read_csv(loc)
    t = L["seconds_elapsed"].to_numpy(float)
    gps = to_xy(L["latitude"].to_numpy(float), L["longitude"].to_numpy(float), lat0, lon0)
    gps, t = clean_for_truth(gps, t)
    xy, _ = road_from_osm(OsmNetwork(OSM, lat0, lon0), gps)
    xy = np.asarray(xy, float)
    xy = xy[np.r_[True, np.linalg.norm(np.diff(xy, axis=0), axis=1) > 0.5]]
    m = 111320.0; k = np.cos(np.radians(lat0))
    pts = [[round(float(lat0 + y / m), 7), round(float(lon0 + x / (m * k)), 7)] for x, y in xy]
    length = float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum())
    ride_len = float(np.linalg.norm(np.diff(gps, axis=0), axis=1).sum())

    doc = json.load(open(routes_path))
    doc["routes"] = [r for r in doc["routes"] if r["name"] != name]
    doc["routes"].insert(0, {"name": name, "length_m": round(length, 1), "n_points": len(pts), "points": pts})
    json.dump(doc, open(routes_path, "w"))
    print(f"\n  added '{name}': {len(pts)} pts, route {length:.0f} m vs ridden {ride_len:.0f} m "
          f"({100*(length-ride_len)/max(ride_len,1):+.1f}%), start->end {np.linalg.norm(xy[-1]-xy[0]):.0f} m")
    print(f"  {routes_path} now holds {len(doc['routes'])} routes: {', '.join(r['name'] for r in doc['routes'])}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    main(*sys.argv[1:])
