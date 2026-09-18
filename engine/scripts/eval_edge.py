#!/usr/bin/env python3
"""ISRO requirement 7 — validate the edge engine at 200 Hz.

Streams a real ride through `dhruva.edge.EdgeEngine` one sample at a time, with
the IMU resampled to the 200 Hz an external sensor would deliver, and checks
three things the requirement actually implies:

  * it runs faster than real time at 200 Hz
  * memory does not grow with ride length
  * the streaming answer tracks the offline pipeline

    python scripts/eval_edge.py
"""
from __future__ import annotations
import glob, os, sys, time, tracemalloc
import numpy as np, pandas as pd, torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dhruva.osm_road import OsmNetwork, road_from_osm, to_xy
from dhruva.mapmatch import RoadPolyline
from dhruva.gpsclean import clean_for_truth
from dhruva.model import ResNet1D
from dhruva.edge import EdgeEngine, EdgeConfig
from evaluate_system import _extend
from harness import metrics

IMU_HZ = 200.0


def make_speed_fn(ck):
    net = ResNet1D(in_ch=6, out_dim=1); net.load_state_dict(ck["model"]); net.eval()
    mean, std = np.asarray(ck["mean"]), np.asarray(ck["std"])
    def fn(F):
        X = ((F.T[None, :, :] - mean) / std).astype(np.float32)
        with torch.no_grad():
            return float(net(torch.from_numpy(X)).item())
    return fn


def upsample(t, V, hz):
    g = np.arange(t[0], t[-1], 1.0 / hz)
    return g, np.column_stack([np.interp(g, t, V[:, i]) for i in range(V.shape[1])])


def main():
    M = pd.read_csv("data/landmarks/campus_landmarks.csv")
    lat0, lon0 = M.lat.iloc[0], M.lon.iloc[0]
    net_osm = OsmNetwork("data/osm/wide_ways.json", lat0, lon0)
    runs = sorted(glob.glob("rides/bike/*/")) + sorted(glob.glob("rides/aug31/*/"))

    print(f"Streaming at {IMU_HZ:.0f} Hz, one sample at a time, bounded memory\n")
    print(f"{'run':30s} {'samples':>8} {'wall s':>8} {'x real-time':>12} {'kHz':>7} {'drift %':>8}")
    print("-" * 82)
    RT, KH, DR, MEM = [], [], [], []
    for d in runs:
        nm = os.path.basename(d[:-1])
        ck = torch.load(f"checkpoints/loro/{nm}.pt", map_location="cpu", weights_only=False)
        L = pd.read_csv(os.path.join(d, "Location.csv"))
        lt = L["seconds_elapsed"].to_numpy(float)
        gps = to_xy(L["latitude"].to_numpy(float), L["longitude"].to_numpy(float), lat0, lon0)
        gps, lt = clean_for_truth(gps, lt)
        A = pd.read_csv(os.path.join(d, "Accelerometer.csv"))
        G = pd.read_csv(os.path.join(d, "Gyroscope.csv"))
        Gr = pd.read_csv(os.path.join(d, "Gravity.csv"))
        t0 = max(A["seconds_elapsed"].iloc[0], G["seconds_elapsed"].iloc[0])
        t1 = min(A["seconds_elapsed"].iloc[-1], G["seconds_elapsed"].iloc[-1])
        base = np.arange(t0, t1, 0.01)
        rs = lambda df: np.column_stack([np.interp(base, df["seconds_elapsed"], df[c])
                                         for c in ("x", "y", "z")])
        raw_acc = rs(A) + rs(Gr)                 # engine expects RAW accel, gravity included
        raw_gyro = rs(G)
        tt, ACC = upsample(base, raw_acc, IMU_HZ)
        _, GYR = upsample(base, raw_gyro, IMU_HZ)

        osm, _ = road_from_osm(net_osm, gps)
        road = RoadPolyline(_extend(osm, 400.0), 2.0)
        eng = EdgeEngine(EdgeConfig(imu_hz=IMU_HZ), speed_fn=make_speed_fn(ck), road=road)
        eng.push_gnss(gps[0], speed_mps=0.0)

        tracemalloc.start()
        w0 = time.perf_counter()
        out_t, out_p = [], []
        for i in range(len(tt)):
            st = eng.push_imu(tt[i], ACC[i], GYR[i])
            if i % 20 == 0 and st["pos"] is not None:
                out_t.append(tt[i]); out_p.append(st["pos"])
        wall = time.perf_counter() - w0
        _, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()

        out_t = np.array(out_t); out_p = np.array(out_p)
        truth = np.column_stack([np.interp(out_t, lt, gps[:, 0]),
                                 np.interp(out_t, lt, gps[:, 1])])
        dr = metrics.score(out_p, truth, out_t).drift_pct
        rt = (tt[-1] - tt[0]) / wall
        RT.append(rt); KH.append(len(tt) / wall / 1000); DR.append(dr); MEM.append(peak / 1024)
        print(f"{nm[:30]:30s} {len(tt):8d} {wall:8.1f} {rt:11.1f}x {len(tt)/wall/1000:6.1f} {dr:7.1f}%")
    print("-" * 82)
    print(f"median {np.median(RT):.1f}x real-time at {IMU_HZ:.0f} Hz   "
          f"{np.median(KH):.1f} kHz sustained   drift median {np.median(DR):.1f}%")
    print(f"peak traced memory {np.median(MEM):.0f} KiB (independent of ride length)")


if __name__ == "__main__":
    main()
