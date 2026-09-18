#!/usr/bin/env python3
"""Turn a walking survey into a landmark map.

You walk the route with Sensor Logger recording Location, stand on each speed
breaker for ~30 s and tap the annotation button. This averages the GPS fixes
around each annotation into one accurate coordinate per breaker.

    python scripts/extract_breakers.py rides/survey/

Writes breakers.csv and breakers.geojson next to the input.

If the annotation button wasn't used, pass --dwell to find the stops instead:
    python scripts/extract_breakers.py rides/survey/ --dwell
"""
from __future__ import annotations
import argparse, json, os
import numpy as np
import pandas as pd

R = 6378137.0


def load_location(d):
    p = os.path.join(d, "Location.csv")
    if not os.path.exists(p) or os.path.getsize(p) == 0:
        raise SystemExit(f"[survey] {p} is empty -- Location was not recorded.\n"
                         "         Enable Location INSIDE Sensor Logger's sensor list,\n"
                         "         grant 'Allow all the time' + 'Precise', and re-record.")
    df = pd.read_csv(p)
    t = df["seconds_elapsed"].to_numpy(float)
    lat = df["latitude"].to_numpy(float)
    lon = df["longitude"].to_numpy(float)
    acc = df["horizontalAccuracy"].to_numpy(float) if "horizontalAccuracy" in df else np.full(len(t), np.nan)
    print(f"[survey] {len(df)} GPS fixes over {t[-1]-t[0]:.0f}s "
          f"({len(df)/(t[-1]-t[0]):.2f} Hz), median accuracy "
          f"{np.nanmedian(acc):.1f} m")
    return t, lat, lon, acc


def from_annotations(d, t, lat, lon, acc, window=20.0):
    p = os.path.join(d, "Annotation.csv")
    if not os.path.exists(p) or os.path.getsize(p) == 0:
        return None
    ann = pd.read_csv(p)
    if "seconds_elapsed" not in ann.columns or len(ann) == 0:
        return None
    print(f"[survey] {len(ann)} annotations found")
    rows = []
    for i, ts in enumerate(ann["seconds_elapsed"].to_numpy(float), 1):
        m = np.abs(t - ts) <= window
        if m.sum() < 2:
            m = np.abs(t - ts) <= window * 3
        if m.sum() == 0:
            continue
        w = 1.0 / np.maximum(acc[m], 1.0) ** 2 if np.isfinite(acc[m]).all() else np.ones(m.sum())
        rows.append({"id": i, "t": round(float(ts), 1),
                     "lat": float(np.average(lat[m], weights=w)),
                     "lon": float(np.average(lon[m], weights=w)),
                     "n_fixes": int(m.sum()),
                     "spread_m": float(_spread(lat[m], lon[m]))})
    return rows


def from_dwell(t, lat, lon, acc, min_stop=15.0, max_move=6.0):
    """Fallback: find where the walker stood still."""
    x = (np.deg2rad(lon) - np.deg2rad(lon[0])) * np.cos(np.deg2rad(lat[0])) * R
    y = (np.deg2rad(lat) - np.deg2rad(lat[0])) * R
    speed = np.r_[0, np.linalg.norm(np.diff(np.c_[x, y], axis=0), axis=1) / np.maximum(np.diff(t), 1e-3)]
    still = pd.Series(speed).rolling(5, center=True, min_periods=1).median().to_numpy() < 0.4
    rows, i, k = [], 0, 0
    while i < len(t):
        if still[i]:
            j = i
            while j < len(t) and still[j]:
                j += 1
            if t[j-1] - t[i] >= min_stop:
                seg = slice(i, j)
                if _spread(lat[seg], lon[seg]) < max_move:
                    k += 1
                    rows.append({"id": k, "t": round(float(t[i]), 1),
                                 "lat": float(np.mean(lat[seg])), "lon": float(np.mean(lon[seg])),
                                 "n_fixes": int(j - i), "spread_m": float(_spread(lat[seg], lon[seg]))})
            i = j
        else:
            i += 1
    print(f"[survey] {len(rows)} dwell stops >= {min_stop}s detected")
    return rows


def _spread(lat, lon):
    if len(lat) < 2:
        return 0.0
    x = (np.deg2rad(lon) - np.deg2rad(lon.mean())) * np.cos(np.deg2rad(lat.mean())) * R
    y = (np.deg2rad(lat) - np.deg2rad(lat.mean())) * R
    return float(np.sqrt(np.mean(x**2 + y**2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir"); ap.add_argument("--dwell", action="store_true")
    a = ap.parse_args()

    t, lat, lon, acc = load_location(a.dir)
    rows = None if a.dwell else from_annotations(a.dir, t, lat, lon, acc)
    if not rows:
        print("[survey] no usable annotations -> falling back to dwell detection")
        rows = from_dwell(t, lat, lon, acc)
    if not rows:
        raise SystemExit("[survey] found nothing. Did you stand still on each breaker?")

    # split odd/even so corrections and scoring stay independent
    for r in rows:
        r["role"] = "landmark" if r["id"] % 2 == 1 else "checkpoint"
        r["type"] = ""          # fill in from your notes: bump|hump|table|rumble

    df = pd.DataFrame(rows)
    out = os.path.join(a.dir, "breakers.csv")
    df.to_csv(out, index=False)
    print(f"\n{df.to_string(index=False)}")
    print(f"\n[survey] -> {out}")

    gj = {"type": "FeatureCollection", "features": [
        {"type": "Feature",
         "geometry": {"type": "Point", "coordinates": [r["lon"], r["lat"]]},
         "properties": {k: r[k] for k in ("id", "role", "type", "spread_m")}}
        for r in rows]}
    gout = os.path.join(a.dir, "breakers.geojson")
    json.dump(gj, open(gout, "w"), indent=1)
    print(f"[survey] -> {gout}   (drag onto geojson.io to eyeball it)")
    print("\nNow open breakers.csv and fill the 'type' column from your notes.")


if __name__ == "__main__":
    main()
