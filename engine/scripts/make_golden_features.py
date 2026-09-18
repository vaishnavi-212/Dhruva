#!/usr/bin/env python3
"""Golden files for the phone's speed-model inputs (ImuFeatures.kt).

Takes 60 s of RAW sensor events from a recorded app ride, writes them exactly as the phone
would receive them, and writes the features the laptop computes from those same events --
10 Hz grid by numpy.interp, trailing 8 s level frame, 6 channels -- for both candidate frames.
ImuFeaturesGoldenTest feeds the raw events through the Kotlin code and must reproduce these.

Grid convention shared with the Kotlin code: starts at the latest first event of the three
sensors, steps 0.1 s, and includes every grid time up to the earliest last event.

    python scripts/make_golden_features.py --out <app>/app/src/test/resources/golden
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_eval_phone_features import frame_features

RIDE = "rides/app_DhruvaRun_2026-09-11_19-41-28"
START_S, SECONDS = 40.0, 60.0            # includes riding before and after that ride's cut at 48 s


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    ev = {}
    for key, name in (("acc", "Accelerometer"), ("gyro", "Gyroscope"), ("grav", "Gravity")):
        d = pd.read_csv(os.path.join(RIDE, name + ".csv"))
        m = (d.seconds_elapsed >= START_S) & (d.seconds_elapsed <= START_S + SECONDS)
        t_ns = np.round(d.seconds_elapsed[m].to_numpy(float) * 1e9).astype(np.int64)
        v = d.loc[m, ["x", "y", "z"]].to_numpy(float)
        with open(os.path.join(a.out, f"raw_{key}.csv"), "w") as f:
            f.write("t_ns,x,y,z\n")
            for tn, (x, y, z) in zip(t_ns, v):
                f.write(f"{int(tn)},{float(x)!r},{float(y)!r},{float(z)!r}\n")   # plain floats: numpy 2 reprs as np.float64(...)
        back = pd.read_csv(os.path.join(a.out, f"raw_{key}.csv"))          # compute from exactly what was written
        ev[key] = (back.t_ns.to_numpy(np.int64) / 1e9, back[["x", "y", "z"]].to_numpy(float))
    t0 = max(ev[k][0][0] for k in ev); t1 = min(ev[k][0][-1] for k in ev)
    K = int(np.floor((t1 - t0) / 0.1 + 1e-9))
    grid = t0 + np.arange(K + 1) * 0.1
    grid = grid[grid <= t1]
    rs = lambda k: np.column_stack([np.interp(grid, ev[k][0], ev[k][1][:, i]) for i in range(3)])
    acc, gyro, grav = rs("acc"), rs("gyro"), rs("grav")
    for frame in ("replica", "gravity"):
        F = frame_features(frame, acc, gyro, grav)
        with open(os.path.join(a.out, f"expected_{frame}.csv"), "w") as f:
            f.write("t_s,f0,f1,f2,f3,f4,f5\n")
            for g, row in zip(grid, F):
                f.write(f"{float(g)!r}," + ",".join(f"{float(x):.9g}" for x in row) + "\n")
    print(f"golden: {sum(len(ev[k][0]) for k in ev)} raw events, {len(grid)} rows per frame -> {a.out}")


if __name__ == "__main__":
    main()
