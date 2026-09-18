"""Error-vs-distance curves and paths for the figures. Same model for every map state
on a ride, so only the map changes (11 Sept: the model at the 11-model median of the
learned-from-previous-ride drift; benchmark: each ride's own held-out model)."""
import glob, os, sys, json, numpy as np, pandas as pd, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import selfheal_eval as SE
from dhruva.osm_road import OsmNetwork, road_from_osm, to_xy
from dhruva.mapmatch import RoadPolyline
from dhruva.gpsclean import clean_for_truth
from dhruva.infer_speed import load_imu, features
from evaluate_system import speed_from_imu, _extend

def free_path(truth, cum, t, wz, v, dt):
    i = int(np.searchsorted(cum, 15.0)); d = truth[min(i, len(truth)-1)] - truth[0]; psi0 = np.arctan2(d[1], d[0])
    vg = np.r_[0, np.diff(cum)] / dt; still = vg < 0.3
    cands = [0.0] + ([float(np.median(wz[still]))] if still.sum() > 30 else [])
    hd = np.unwrap(np.arctan2(*np.gradient(truth, axis=0)[:, ::-1].T)); m = cum > 10; best = None
    for b0 in cands:   # same best-case rule as selfheal_eval.free_dr
        res = {s: np.median(np.abs(np.angle(np.exp(1j*((psi0 + s*np.cumsum(wz-b0)*dt) - hd)[m])))) for s in (1, -1)}
        sg = min(res, key=res.get); pc = psi0 + sg*np.cumsum(wz-b0)*dt
        e = float(np.linalg.norm(truth[0] + np.array([np.sum(vg*np.cos(pc)), np.sum(vg*np.sin(pc))])*dt - truth[-1]))
        if best is None or e < best[0]: best = (e, sg, b0)
    psi = psi0 + best[1]*np.cumsum(wz-best[2])*dt
    return truth[0] + np.column_stack([np.cumsum(v*np.cos(psi)), np.cumsum(v*np.sin(psi))])*dt

def bound(poly, truth, v, dt, pad=3000.0, start=None):
    # benchmark rides start at the first GPS fix, exactly as loro_eval.py does
    road = RoadPolyline(_extend(poly, pad), 2.0)
    return road.at(road.project(truth[0] if start is None else start) + np.cumsum(v)*dt)[0]

D = {}
# benchmark rides
net1 = OsmNetwork("data/osm/wide_ways.json", SE.lat0, SE.lon0)
for r in sorted(glob.glob("rides/bike/*/")) + sorted(glob.glob("rides/aug31/*/")):
    name = os.path.basename(r[:-1])
    ck = torch.load(f"checkpoints/loro/{name}.pt", map_location="cpu", weights_only=False)
    L = pd.read_csv(os.path.join(r, "Location.csv")); lt = L.seconds_elapsed.to_numpy(float)
    gps, lt = clean_for_truth(to_xy(L.latitude.to_numpy(float), L.longitude.to_numpy(float), SE.lat0, SE.lon0), lt)
    t, v = speed_from_imu(r, ck); v = np.clip(v, 0, None); dt = float(np.median(np.diff(t)))
    wz = features(*load_imu(r)[1:])[:, 2]
    truth = np.column_stack([np.interp(t, lt, gps[:, 0]), np.interp(t, lt, gps[:, 1])])
    cum = np.r_[0, np.cumsum(np.linalg.norm(np.diff(truth, axis=0), axis=1))]
    pr = bound(road_from_osm(net1, gps)[0], truth, v, dt, 400.0, start=gps[0]); pn = free_path(truth, cum, t, wz, v, dt)
    k = np.arange(0, len(t), 10)
    D[f"bench|{name}|dist"] = cum[k]; D[f"bench|{name}|road"] = np.linalg.norm(pr-truth, axis=1)[k]
    D[f"bench|{name}|none"] = np.linalg.norm(pn-truth, axis=1)[k]
    print(name, f"road {100*np.linalg.norm(pr[-1]-truth[-1])/cum[-1]:.1f}% none {100*np.linalg.norm(pn[-1]-truth[-1])/cum[-1]:.1f}%", flush=True)

# 11 Sept rides
cache = np.load(os.path.join(HERE, "speed_cache.npz"))
models = sorted({k.split("|")[1] for k in cache.files if k.split("|")[1] not in ("t", "wz")})
tracks = [SE.load(f) for _, f in SE.PASSES]
for ti in range(8, 12):
    label, folder = SE.PASSES[ti]; stamp = folder.split("_")[-1]
    gps, lt = tracks[ti]; t = cache[f"{stamp}|t"]; keep = (t >= lt[0]) & (t <= lt[-1])
    t = t[keep]; wz = cache[f"{stamp}|wz"][keep]; dt = float(np.median(np.diff(t)))
    truth = np.column_stack([np.interp(t, lt, gps[:, 0]), np.interp(t, lt, gps[:, 1])])
    cum = np.r_[0, np.cumsum(np.linalg.norm(np.diff(truth, axis=0), axis=1))]
    ref = SE.resample(gps, 1.0)
    prev = SE.orient(SE.fuse([tracks[ti-1][0]]), ref)
    osm = np.asarray(road_from_osm(SE.net, gps)[0], float)
    V = [np.clip(cache[f"{stamp}|{m}"][keep], 0, None) for m in models]
    dr = [100*np.linalg.norm(bound(prev, truth, v, dt)[-1]-truth[-1])/cum[-1] for v in V]
    mi = int(np.argmin(np.abs(np.array(dr) - np.median(dr)))); v = V[mi]
    paths = dict(truth=truth, prev=bound(prev, truth, v, dt), osm=bound(osm, truth, v, dt), none=free_path(truth, cum, t, wz, v, dt))
    k = np.arange(0, len(t), 10)
    D[f"sep|{stamp}|dist"] = cum[k]; D[f"sep|{stamp}|prevmap"] = prev; D[f"sep|{stamp}|prev_label"] = np.array(SE.PASSES[ti-1][0])
    for n, p in paths.items():
        D[f"sep|{stamp}|{n}"] = p[k]
    print(label, "model", models[mi], {n: round(100*np.linalg.norm(p[-1]-truth[-1])/cum[-1], 1) for n, p in paths.items() if n != "truth"}, flush=True)
np.savez(os.path.join(HERE, "curves.npz"), **D); print("saved")
