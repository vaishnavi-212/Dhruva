#!/usr/bin/env python3
"""Turn a GPS ride + the surveyed landmark map into LABELLED training windows.

This is the piece that makes a ride trainable. For each surveyed landmark we
find every moment the vehicle's GPS position passed closest to it -- one per lap
-- and label the IMU window around that moment with the landmark's surveyed type
(bump / hump / table / rumble). Everything else becomes a negative.

    python scripts/label_ride.py rides/<ride_folder> \
           --map data/landmarks/campus_landmarks.csv --out data/detector

Without ride GPS this cannot work, which is exactly why the 28 Aug rides are
unusable for training: Location.csv was empty in all four.
"""
from __future__ import annotations
import argparse, os, json
import numpy as np
import pandas as pd

R = 6378137.0


def local_xy(lat, lon, lat0, lon0):
    x = (np.deg2rad(lon) - np.deg2rad(lon0)) * np.cos(np.deg2rad(lat0)) * R
    y = (np.deg2rad(lat) - np.deg2rad(lat0)) * R
    return np.column_stack([x, y])


def load_ride(folder):
    a = pd.read_csv(os.path.join(folder, "Accelerometer.csv"))
    t = a["seconds_elapsed"].to_numpy(float)
    acc = a[["x", "y", "z"]].to_numpy(float)
    hz = 1.0 / float(np.median(np.diff(t)))
    loc_p = os.path.join(folder, "Location.csv")
    if os.path.getsize(loc_p) < 60:
        raise SystemExit(f"{folder}: Location.csv is empty -- ride cannot be labelled.")
    L = pd.read_csv(loc_p)
    return t, acc, hz, L


def crossings(L, landmarks, max_dist_m=12.0, min_gap_s=8.0):
    """Every time the ride passed within max_dist_m of each landmark."""
    lt = L["seconds_elapsed"].to_numpy(float)
    lat0, lon0 = landmarks.lat.iloc[0], landmarks.lon.iloc[0]
    ride = local_xy(L["latitude"].to_numpy(float), L["longitude"].to_numpy(float), lat0, lon0)
    out = []
    for lm in landmarks.itertuples():
        p = local_xy(np.array([lm.lat]), np.array([lm.lon]), lat0, lon0)[0]
        d = np.linalg.norm(ride - p, axis=1)
        near = d < max_dist_m
        if not near.any():
            continue
        # split into separate passes
        idx = np.where(near)[0]
        splits = np.split(idx, np.where(np.diff(lt[idx]) > min_gap_s)[0] + 1)
        for grp in splits:
            if len(grp) == 0:
                continue
            j = grp[int(np.argmin(d[grp]))]
            out.append({"landmark_id": int(lm.id), "type": lm.type,
                        "t": float(lt[j]), "closest_m": float(d[j])})
    out.sort(key=lambda r: r["t"])
    return out


def build_windows(t, acc, hz, cross, window_s=3.0, stride_s=0.15, pos_tol_s=0.25):
    """Windows centred on a grid; positive if a crossing sits near the centre."""
    w = int(window_s * hz)
    step = max(int(stride_s * hz), 1)
    ct = np.array([c["t"] for c in cross]) if cross else np.array([])
    ctypes = [c["type"] for c in cross]
    X, y, ty, tc = [], [], [], []
    for i in range(0, len(t) - w, step):
        centre = t[i + w // 2]
        lab, typ = 0, "none"
        if len(ct):
            j = int(np.argmin(np.abs(ct - centre)))
            if abs(ct[j] - centre) <= pos_tol_s:
                lab, typ = 1, ctypes[j]
        X.append(acc[i:i + w]); y.append(lab); ty.append(typ); tc.append(float(centre))
    return np.asarray(X, np.float32), np.asarray(y, np.int64), ty, np.asarray(tc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ride"); ap.add_argument("--map", default="data/landmarks/campus_landmarks.csv")
    ap.add_argument("--out", default="data/detector"); ap.add_argument("--window", type=float, default=3.0)
    a = ap.parse_args()

    t, acc, hz, L = load_ride(a.ride)
    M = pd.read_csv(a.map)
    print(f"[label] ride {os.path.basename(a.ride)}: {t[-1]:.0f}s @ {hz:.0f} Hz, "
          f"{len(L)} GPS fixes")
    cross = crossings(L, M)
    print(f"[label] {len(cross)} crossings of {M.id.nunique()} surveyed landmarks")
    if cross:
        per = pd.Series([c['landmark_id'] for c in cross]).value_counts().sort_index()
        print(f"[label]   passes per landmark: {dict(per)}")
        print(f"[label]   median closest approach: "
              f"{np.median([c['closest_m'] for c in cross]):.1f} m")
    X, y, ty, tc = build_windows(t, acc, hz, cross, window_s=a.window)
    print(f"[label] {len(X)} windows | {int(y.sum())} positive ({y.mean()*100:.2f}%)")
    from collections import Counter
    print(f"[label]   type balance: {dict(Counter(t_ for t_ in ty if t_!='none'))}")

    os.makedirs(a.out, exist_ok=True)
    tag = os.path.basename(a.ride.rstrip("/"))
    np.savez_compressed(os.path.join(a.out, tag + ".npz"),
                        X=X, y=y, types=np.array(ty), t=tc, hz=hz)
    json.dump(cross, open(os.path.join(a.out, tag + "_crossings.json"), "w"), indent=1)
    print(f"[label] -> {a.out}/{tag}.npz")


if __name__ == "__main__":
    main()
