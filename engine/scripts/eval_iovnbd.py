#!/usr/bin/env python3
"""IO-VNBD position results for the SIH proposal: the preliminary AI model on drives it never saw.

The problem statement: "Teams are required to include the preliminary AI models and the results of
the position plot inferenced from the subset of IO-VNBD dataset."

Model: checkpoints/speed_best.pt, trained on 67 IO-VNBD smartphone files. Test drives: the 5 files held
out of that training (S-S2, S-S3a, S-Vta1a, S-Vw2, S-Y1). Input pipeline checked first: the held-out
window error reproduces the checkpoint's recorded 110.2 m (see --check).

Blackouts are cut every 250 m along each drive at 500, 1000 and 2000 m, both ends snapped to REAL GPS
fixes (IO-VNBD GPS updates only every ~9 s; glitch fixes over 60 m/s are dropped). Every blackout is
run four ways:
  frozen          the dot stops where GPS was lost (what a phone navigation app shows)
  AI + gyro       distance from the AI model, heading from the gyroscope (no map). The gyro sign is
                  fitted per drive on its own GPS: hindsight, so this baseline is at its best
  AI + road       distance from the AI model along the OpenStreetMap road the drive used (Dhruva)
  perfect + road  GPS distance along the same road: what remains is map error only
  AI cal + road   AI + road, with the model's distance scaled by GPS distance / AI distance over the
                  120 s BEFORE the cut (GPS still healthy), limited to 0.8-1.25. Settings fixed before
                  any result was seen (18-19 Sept), run once, not tuned afterwards.

The road is chosen from the drive's own GPS (the "known route" assumption of the campus benchmark),
snapped to OpenStreetMap ways fetched by scripts/fetch_osm_iovnbd.py.

    python scripts/eval_iovnbd.py --check     # reproduce the checkpoint's held-out error
    python scripts/eval_iovnbd.py             # results/iovnbd/*.png, results/iovnbd/iovnbd_blackouts.csv
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np, pandas as pd, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import dataset as ds
from dhruva.dataset_iovnbd import build_features
from dhruva.model import ResNet1D
from dhruva import align
from dhruva.osm_road import OsmNetwork, road_from_osm
from dhruva.mapmatch import RoadPolyline
from evaluate_system import _extend
from fetch_osm_iovnbd import clean_fixes

HELD_OUT = ["S-S2", "S-S3a", "S-Vta1a", "S-Vw2", "S-Y1"]
TARGETS = (500.0, 1000.0, 2000.0)
CAL_S, CAL_MIN, CAL_MAX = 120.0, 0.8, 1.25       # pre-registered: do not tune on these results
STRIDE_M = 250.0
W = 300
OUT = "results/iovnbd"


def model():
    ck = torch.load("checkpoints/speed_best.pt", map_location="cpu", weights_only=False)
    net = ResNet1D(in_ch=6, out_dim=1); net.load_state_dict(ck["model"]); net.eval()
    return net, ck["mean"], ck["std"]


def window_metres(net, mean, std, F, step):
    idx = np.arange(0, len(F) - W, step)
    X = ((np.stack([F[i:i + W] for i in idx]).transpose(0, 2, 1) - mean) / std).astype(np.float32)
    with torch.no_grad():
        p = np.concatenate([net(torch.from_numpy(X[k:k + 1024])).numpy().ravel() for k in range(0, len(X), 1024)])
    return idx, p


def check():
    net, mean, std = model(); errs = []
    for f in HELD_OUT:
        tr = ds.load(f"data/{f}.csv", verbose=False)
        F, cum = build_features(tr, invariant=False)
        idx, p = window_metres(net, mean, std, F, 8)
        y = np.array([cum[i + W - 1] - cum[i] for i in idx]); ok = np.isfinite(y) & (y >= 0)
        errs.append(p[ok] - y[ok])
    e = np.concatenate(errs)
    print(f"held-out window RMSE {np.sqrt(np.mean(e ** 2)):.1f} m (checkpoint records 110.2 m)")


def prepare(name, net, mean, std):
    tr = ds.load(f"data/{name}.csv", verbose=False)
    F, _ = build_features(tr, invariant=False)
    idx, p = window_metres(net, mean, std, F, 1)
    v_ai = np.clip(np.interp(tr.t, tr.t[idx + W // 2], p / (W / tr.hz)), 0, None)   # m/s, window centre

    fx = clean_fixes(tr)                                          # real, glitch-free GPS fixes
    truth = np.column_stack([np.interp(tr.t, tr.t[fx], tr.xy[fx, 0]), np.interp(tr.t, tr.t[fx], tr.xy[fx, 1])])
    cum = np.r_[0, np.cumsum(np.linalg.norm(np.diff(truth, axis=0), axis=1))]

    R = align.estimate_tilt(tr.acc_raw if tr.acc_raw is not None else tr.acc, tr.hz)
    wz = align.apply(R, tr.gyro)[:, 2]
    # gyro sign by the drive's own GPS course (hindsight: the no-map baseline at its best)
    c = np.unwrap(np.arctan2(*np.gradient(truth, axis=0)[:, ::-1].T))
    rate = np.gradient(c) * tr.hz; mv = np.linalg.norm(np.gradient(truth, axis=0), axis=1) * tr.hz > 3
    sgn = 1.0 if np.corrcoef(wz[mv], rate[mv])[0, 1] >= 0 else -1.0

    lat0, lon0 = float(tr.lat[0]), float(tr.lon[0])
    net_osm = OsmNetwork(f"data/osm/iovnbd_{name}.json", lat0, lon0, drivable_only=False)
    return dict(tr=tr, truth=truth, cum=cum, v_ai=v_ai, wz=sgn * wz, fx=fx, net=net_osm, dt=1.0 / tr.hz)


def blackouts(P):
    fx, cum = P["fx"], P["cum"]; out = []
    for target in TARGETS:
        start = 0.0
        while True:
            a = fx[np.searchsorted(cum[fx], start)] if np.searchsorted(cum[fx], start) < len(fx) else None
            if a is None: break
            bi = np.searchsorted(cum[fx], cum[a] + target)
            if bi >= len(fx): break
            b = fx[bi]
            dur = P["tr"].t[b] - P["tr"].t[a]
            if dur > 0 and (cum[b] - cum[a]) / dur > 5 / 3.6:
                out.append((target, int(a), int(b)))
            start += STRIDE_M
    return out


def run_one(P, a, b):
    truth, cum, dt = P["truth"], P["cum"], P["dt"]
    seg = slice(a, b + 1); tr_seg = truth[seg]; dist = float(cum[b] - cum[a])
    end = truth[b]
    res = {"frozen": float(np.linalg.norm(truth[a] - end))}

    # heading at the cut from the last ~30 m of GPS before it
    j = max(int(np.searchsorted(cum, cum[a] - 30.0)), 0)
    d0 = truth[a] - truth[j] if a > j else truth[min(a + 5, b)] - truth[a]
    psi = np.arctan2(d0[1], d0[0]) + np.cumsum(P["wz"][seg]) * dt
    v = P["v_ai"][seg]
    free = truth[a] + np.column_stack([np.cumsum(v * np.cos(psi)), np.cumsum(v * np.sin(psi))]) * dt
    res["AI + gyro"] = float(np.linalg.norm(free[-1] - end))

    lo = max(int(np.searchsorted(cum, cum[a] - 200.0)), 0)
    track = truth[lo:b + 1:max(int(1.0 / dt), 1)]                 # the drive's own GPS, once a second
    osm, _ = road_from_osm(P["net"], track)
    road = RoadPolyline(_extend(np.asarray(osm, float), 500.0), 2.0)
    s0 = road.project(truth[a])
    res["AI + road"] = float(np.linalg.norm(road.at(np.array([s0 + np.sum(v) * dt]))[0][0] - end))
    res["perfect + road"] = float(np.linalg.norm(road.at(np.array([s0 + dist]))[0][0] - end))
    path_ai = road.at(s0 + np.cumsum(v) * dt)[0]

    # real-time correction: calibrate the model against GPS while GPS was still healthy
    c0 = max(int(np.searchsorted(P["tr"].t, P["tr"].t[a] - CAL_S)), 0)
    ai_before = float(np.sum(P["v_ai"][c0:a]) * dt); gps_before = float(cum[a] - cum[c0])
    scale = float(np.clip(gps_before / ai_before, CAL_MIN, CAL_MAX)) if ai_before > 50.0 else 1.0
    res["AI cal + road"] = float(np.linalg.norm(road.at(np.array([s0 + scale * np.sum(v) * dt]))[0][0] - end))
    res["scale"] = scale
    return dist, res, dict(free=free, road_path=path_ai, truth=tr_seg, road=road.xy)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--check", action="store_true")
    ap.add_argument("--drives", default=",".join(HELD_OUT)); a = ap.parse_args()
    drives = [d for d in a.drives.split(",") if d]
    assert set(drives) <= set(HELD_OUT), "only the held-out drives are fair tests"
    if a.check:
        return check()
    os.makedirs(OUT, exist_ok=True)
    net, mean, std = model()
    rows, keep = [], {}
    for name in drives:
        P = prepare(name, net, mean, std)
        for target, i0, i1 in blackouts(P):
            dist, res, paths = run_one(P, i0, i1)
            dur = P["tr"].t[i1] - P["tr"].t[i0]
            scale = res.pop("scale")
            rows.append(dict(drive=name, target_m=target, i0=i0, i1=i1, dist_m=dist, dur_s=dur,
                             speed_kmh=3.6 * dist / dur, cal_scale=scale, **{f"err_{k}": v for k, v in res.items()}))
            keep[(name, i0, i1)] = paths
        D = pd.DataFrame([r for r in rows if r["drive"] == name])
        print(f"{name:8s} {len(D):4d} blackouts | median drift % " + "  ".join(
            f"{m}: {100 * np.median(D[f'err_{m}'] / D.dist_m):6.1f}" for m in ("frozen", "AI + gyro", "AI + road", "AI cal + road", "perfect + road")), flush=True)
    D = pd.DataFrame(rows); D.to_csv(f"{OUT}/iovnbd_blackouts.csv", index=False)
    methods = ("frozen", "AI + gyro", "AI + road", "AI cal + road", "perfect + road")
    print("\nall held-out drives, drift % (median, share under 10%):")
    for t in TARGETS:
        X = D[D.target_m == t]
        print(f"  {int(t):5d} m ({len(X):3d} blackouts): " + "  ".join(
            f"{m} {100 * np.median(X[f'err_{m}'] / X.dist_m):5.1f}% ({100 * np.mean(X[f'err_{m}'] / X.dist_m < 0.1):3.0f}%)" for m in methods))
    fast = D[(D.target_m == 1000.0) & (D.speed_kmh >= 50)]
    print(f"\nPS example '1 km at 60 km/h': {len(fast)} one-km blackouts at >= 50 km/h (median {fast.speed_kmh.median():.0f} km/h): "
          + "  ".join(f"{m} median error {fast[f'err_{m}'].median():5.0f} m, under 100 m {int((fast[f'err_{m}'] < 100).sum())}/{len(fast)}" for m in methods))
    json.dump({f"{k[0]}|{k[1]}|{k[2]}": {kk: vv.tolist() for kk, vv in v.items()} for k, v in keep.items()},
              open(f"{OUT}/paths.json", "w"))
    print(f"saved {OUT}/iovnbd_blackouts.csv and paths.json")


if __name__ == "__main__":
    main()
