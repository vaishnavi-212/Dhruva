#!/usr/bin/env python3
"""OpenStreetMap roads along an IO-VNBD drive, for road binding on the held-out test drives.

Fetches every car road within 40 m of the drive's real GPS fixes (Overpass `around` a polyline),
including motorways, trunk roads and slip roads, which the campus loader's DRIVABLE set leaves out.
Glitch fixes (implied speed > 60 m/s) are dropped first.

    python scripts/fetch_osm_iovnbd.py S-Vw2 S-Vta1a ...   ->  data/osm/iovnbd_<name>.json
"""
from __future__ import annotations
import json, os, sys, time, urllib.parse, urllib.request
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from harness import dataset as ds

MIRRORS = ["https://overpass.kumi.systems/api/interpreter", "https://overpass-api.de/api/interpreter"]
HIGHWAYS = ("motorway|trunk|primary|secondary|tertiary|unclassified|residential|service|living_street|road|"
            "motorway_link|trunk_link|primary_link|secondary_link|tertiary_link")
USER_AGENT = "Dhruva-SIH26168-research/1.0 (one-off fetch for evaluation)"


def clean_fixes(tr, vmax=60.0):
    idx = np.asarray(tr.fix_idx if tr.fix_idx is not None else np.arange(len(tr.t)))
    keep = [idx[0]]
    for i in idx[1:]:
        d = np.linalg.norm(tr.xy[i] - tr.xy[keep[-1]]); dt = tr.t[i] - tr.t[keep[-1]]
        if dt > 0 and d / dt <= vmax:
            keep.append(i)
    return np.array(keep)


def main(names):
    os.makedirs("data/osm", exist_ok=True)
    for name in names:
        out = f"data/osm/iovnbd_{name}.json"
        if os.path.exists(out):
            print(f"{name}: cached {out}"); continue
        tr = ds.load(f"data/{name}.csv", verbose=False)
        k = clean_fixes(tr)
        pts, last = [], None                      # a point every ~150 m of the drive keeps the query small
        for i in k:
            if last is None or np.linalg.norm(tr.xy[i] - tr.xy[last]) > 150:
                pts.append((float(tr.lat[i]), float(tr.lon[i]))); last = i
        coords = ",".join(f"{la:.6f},{lo:.6f}" for la, lo in pts)
        q = f'[out:json][timeout:300];way(around:40,{coords})["highway"~"^({HIGHWAYS})$"];out tags geom;'
        for url in MIRRORS:
            try:
                req = urllib.request.Request(url, data=urllib.parse.urlencode({"data": q}).encode(),
                                             headers={"User-Agent": USER_AGENT})
                raw = urllib.request.urlopen(req, timeout=330).read()
                d = json.loads(raw)
                if not d.get("elements"):
                    raise RuntimeError("zero elements")
                json.dump(d, open(out, "w"))
                print(f"{name}: {len(pts)} corridor points -> {len(d['elements'])} ways, {len(raw)/1e6:.1f} MB from {url.split('/')[2]}")
                break
            except Exception as e:
                print(f"{name}: {url.split('/')[2]} failed: {str(e)[:120]}")
                time.sleep(5)
        time.sleep(3)


if __name__ == "__main__":
    main(sys.argv[1:] or ["S-S2", "S-S3a", "S-Vta1a", "S-Vw2", "S-Y1"])
