"""Self-healing map, measured: does a road learned from earlier GPS rides make a later
GNSS-denied ride better, and does it keep improving as more rides are added?

Corridor: the surveyed campus path, 12 full passes (30 Aug x6, 31 Aug x2, 11 Sept x4),
both directions, 3 phones. Every map is built from GPS ONLY and never from the ride it
is scored on.

Map states compared on the same held-out ride:
  none      free dead reckoning: gyro heading (bias removed, sign fitted: generous) + speed
  osm       public OpenStreetMap centreline (the benchmark pipeline)
  learned-k road learned from k OTHER passes (all subsets up to 20, median over subsets)

Metrics:
  oracle drift  perfect speed from the held-out ride's own GPS, run along the map.
                Speed error is removed, so what remains is error the MAP causes.
  xtrack        median distance from the held-out GPS track to the map (m)
  len_err       map length between the ride's start and end vs distance ridden (%)
  model drift   the real pipeline: 11 held-out speed models, median (11 Sept rides only,
                where none of the 11 models ever saw the ride)
"""
from __future__ import annotations
import glob, itertools, json, os, sys
import numpy as np, pandas as pd
from scipy.spatial import cKDTree
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))   # repo root
sys.path.insert(0, ROOT); sys.path.insert(0, ROOT + "/scripts")
os.chdir(ROOT)
from dhruva.osm_road import OsmNetwork, road_from_osm, to_xy
from dhruva.mapmatch import RoadPolyline
from dhruva.gpsclean import clean_for_truth
from evaluate_system import _extend

HERE = os.path.dirname(os.path.abspath(__file__))
PASSES = [  # chronological
    ("30 Aug 16:31", "rides/bike/948G_RVV-2026-08-30_16-31-01"),
    ("30 Aug 16:34", "rides/bike/94CC_6RQ-2026-08-30_16-34-04"),
    ("30 Aug 16:36", "rides/bike/new_senior-2026-08-30_16-36-55"),
    ("30 Aug 16:39", "rides/bike/94CC_6RQ-2026-08-30_16-39-43"),
    ("30 Aug 16:43", "rides/bike/new_senior-2026-08-30_16-43-04"),
    ("30 Aug 16:46", "rides/bike/94CC_6RQ-2026-08-30_16-46-01"),
    ("31 Aug 08:26", "rides/aug31/948G_RVV-2026-08-31_08-26-28"),
    ("31 Aug 08:29", "rides/aug31/94CC_6RQ-2026-08-31_08-29-54"),
    ("11 Sep 13:42", "rides/app_DhruvaRun_2026-09-11_13-42-07"),
    ("11 Sep 13:46", "rides/app_DhruvaRun_2026-09-11_13-46-20"),
    ("11 Sep 19:37", "rides/app_DhruvaRun_2026-09-11_19-37-43"),
    ("11 Sep 19:41", "rides/app_DhruvaRun_2026-09-11_19-41-28"),
]
KS = [1, 2, 3, 5, 8, 11]
MAX_SUBSETS = 20
MATCH_M = 20.0

S = pd.read_csv("data/landmarks/campus_landmarks.csv")
lat0, lon0 = float(S.lat.iloc[0]), float(S.lon.iloc[0])
net = OsmNetwork("data/osm/wide_ways_v2.json", lat0, lon0)


def resample(xy, step):
    xy = np.asarray(xy, float)
    seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    keep = np.r_[True, seg > 1e-6]; xy = xy[keep]
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
    g = np.arange(0, s[-1], step)
    return np.column_stack([np.interp(g, s, xy[:, 0]), np.interp(g, s, xy[:, 1])])


def load(folder):
    L = pd.read_csv(os.path.join(folder, "Location.csv"))
    lt = L.seconds_elapsed.to_numpy(float)
    g = to_xy(L.latitude.to_numpy(float), L.longitude.to_numpy(float), lat0, lon0)
    g, lt = clean_for_truth(g, lt)
    return g, lt


def arc_of(poly, p):
    return float(np.argmin(np.linalg.norm(poly - p, axis=1)))


def orient(tr, ref):
    return tr if arc_of(ref, tr[-1]) >= arc_of(ref, tr[0]) else tr[::-1]


def fuse(traces, iters=4):
    """Average several GPS passes into one centreline (nearest-point averaging)."""
    tr = [resample(t, 1.0) for t in traces]
    ref = max(tr, key=len)
    tr = [orient(t, ref) for t in tr]
    for _ in range(iters):
        R = resample(ref, 2.0)
        acc = np.zeros_like(R); cnt = np.zeros(len(R))
        for t in tr:
            d, j = cKDTree(t).query(R)
            ok = (d < MATCH_M) & (j > 0) & (j < len(t) - 1)   # never pull onto a pass's end
            acc[ok] += t[j[ok]]; cnt[ok] += 1
        keep = cnt > 0
        new = acc[keep] / cnt[keep, None]
        k = 5; pad = k // 2
        new = np.column_stack([np.convolve(np.r_[np.full(pad, new[0, i]), new[:, i], np.full(pad, new[-1, i])],
                                           np.ones(k) / k, mode="valid") for i in (0, 1)])
        ref = new
    return resample(ref, 2.0)


def score_map(poly, truth, cum, speeds, dt):
    """poly oriented to the ride. Returns metrics dict."""
    road = RoadPolyline(_extend(poly, 3000.0), 2.0)
    fine = resample(road.xy, 0.5)
    xt = cKDTree(fine).query(truth)[0]
    s0, s1 = road.project(truth[0]), road.project(truth[-1])
    len_err = 100 * ((s1 - s0) - cum[-1]) / cum[-1]
    orc = road.at(s0 + cum)[0]
    oerr = np.linalg.norm(orc - truth, axis=1)
    out = dict(xtrack_med=float(np.median(xt)), xtrack_p90=float(np.percentile(xt, 90)), len_err=float(len_err),
               oracle_drift=float(100 * oerr[-1] / cum[-1]), oracle_mean=float(oerr.mean()))
    if speeds is not None:
        dr, me = [], []
        for v in speeds:
            p = road.at(s0 + np.cumsum(v) * dt)[0]
            e = np.linalg.norm(p - truth, axis=1)
            dr.append(100 * e[-1] / cum[-1]); me.append(e.mean())
        dr = np.array(dr)
        out.update(model_drift=float(np.median(dr)), model_pass=int((dr < 10).sum()),
                   model_mean_err=float(np.median(me)), model_drifts=dr.round(2).tolist())
    return out


def free_dr(truth, cum, lt_speed_gps, t, wz, speeds, dt):
    """No map: heading integrated from the levelled gyro."""
    # sign convention and bias fitted with GPS: the most generous version of this baseline
    i = int(np.searchsorted(cum, 15.0)); d = truth[min(i, len(truth) - 1)] - truth[0]
    psi0 = np.arctan2(d[1], d[0])
    # bias: none, or the median while GPS says stopped (a rider handling the phone at a
    # stop corrupts that estimate, 8ah). BEST CASE: keep whichever ends closer with
    # perfect speed -- hindsight, so the baseline can never be called a strawman.
    still = lt_speed_gps < 0.3
    cands = [0.0] + ([float(np.median(wz[still]))] if still.sum() > 30 else [])
    hd = np.unwrap(np.arctan2(*np.gradient(truth, axis=0)[:, ::-1].T))
    m = cum > 10; v_true = np.r_[0, np.diff(cum)] / dt; best = None
    for b0 in cands:
        res = {s: np.median(np.abs(np.angle(np.exp(1j * ((psi0 + s * np.cumsum(wz - b0) * dt) - hd)[m])))) for s in (1, -1)}
        sg = min(res, key=res.get)                     # sign is a mounting convention
        pc = psi0 + sg * np.cumsum(wz - b0) * dt
        end = truth[0] + np.array([np.sum(v_true * np.cos(pc)), np.sum(v_true * np.sin(pc))]) * dt
        e = float(np.linalg.norm(end - truth[-1]))
        if best is None or e < best[0]: best = (e, sg, b0, res[sg])
    _, sgn, bias, r = best
    best = (r, sgn)
    psi = psi0 + sgn * np.cumsum(wz - bias) * dt
    out = {}
    for name, vs in (("oracle", [np.r_[0, np.diff(cum)] / dt]), ("model", speeds)):
        if vs is None: continue
        dr = []
        for v in vs:
            p = truth[0] + np.column_stack([np.cumsum(v * np.cos(psi)), np.cumsum(v * np.sin(psi))]) * dt
            dr.append(100 * np.linalg.norm(p[-1] - truth[-1]) / cum[-1])
        out[name] = float(np.median(dr))
    out["bias_dps"] = float(np.degrees(bias)); out["heading_err_deg"] = float(np.degrees(best[0]))
    return out


def main():
    tracks = [load(f) for _, f in PASSES]
    cache = np.load(os.path.join(HERE, "speed_cache.npz"))
    models = sorted({k.split("|")[1] for k in cache.files if k.split("|")[1] not in ("t", "wz")})
    results = []
    for ti, (label, folder) in enumerate(PASSES):
        gps, lt = tracks[ti]
        stamp = folder.split("_")[-1]
        has_model = f"{stamp}|t" in cache.files
        if has_model:
            t = cache[f"{stamp}|t"]; wz = cache[f"{stamp}|wz"]
            keep = (t >= lt[0]) & (t <= lt[-1])
            t, wz = t[keep], wz[keep]
            speeds = [np.clip(cache[f"{stamp}|{m}"][keep], 0, None) for m in models]
        else:
            t = np.arange(lt[0], lt[-1], 0.1); wz = None; speeds = None
        dt = float(np.median(np.diff(t)))
        truth = np.column_stack([np.interp(t, lt, gps[:, 0]), np.interp(t, lt, gps[:, 1])])
        cum = np.r_[0, np.cumsum(np.linalg.norm(np.diff(truth, axis=0), axis=1))]
        rec = dict(ride=label, folder=folder, dist_m=float(cum[-1]), has_model=has_model)

        rec["osm"] = score_map(np.asarray(road_from_osm(net, gps)[0], float), truth, cum, speeds, dt)
        if has_model:
            vg = np.r_[0, np.linalg.norm(np.diff(truth, axis=0), axis=1)] / dt
            rec["none"] = free_dr(truth, cum, vg, t, wz, speeds, dt)

        ref = resample(gps, 1.0)
        others = [j for j in range(len(PASSES)) if j != ti]
        prior = [j for j in others if j < ti]
        rec["learned"] = {}
        for pool_name, pool in (("any", others), ("prior", prior)):
            for k in KS:
                if k > len(pool): continue
                subs = list(itertools.combinations(pool, k))
                if len(subs) > MAX_SUBSETS:
                    rng = np.random.default_rng(1000 * ti + k)
                    subs = [subs[i] for i in rng.choice(len(subs), MAX_SUBSETS, replace=False)]
                rows = []
                for sub in subs:
                    poly = orient(fuse([tracks[j][0] for j in sub]), ref)
                    rows.append(score_map(poly, truth, cum, speeds, dt))
                agg = {m: float(np.median([r[m] for r in rows])) for m in rows[0] if m != "model_drifts"}
                agg["n_subsets"] = len(rows)
                agg["oracle_drift_all"] = [round(r["oracle_drift"], 2) for r in rows]
                if has_model:
                    agg["model_drift_all"] = [round(r["model_drift"], 2) for r in rows]
                rec["learned"][f"{pool_name}_{k}"] = agg
        # the story case: the ride immediately before this one, alone
        if ti > 0:
            poly = orient(fuse([tracks[ti - 1][0]]), ref)
            rec["learned"]["previous_ride"] = score_map(poly, truth, cum, speeds, dt)
            rec["learned"]["previous_ride"]["from"] = PASSES[ti - 1][0]
        results.append(rec)

        o = rec["osm"]; L = rec["learned"]
        line = f"{label:13s} {cum[-1]:5.0f}m | OSM orc {o['oracle_drift']:5.1f}% xt {o['xtrack_med']:4.1f}m len {o['len_err']:+5.1f}%"
        for k in (1, 3, 11):
            if f"any_{k}" in L:
                line += f" | L{k} orc {L[f'any_{k}']['oracle_drift']:4.1f}% xt {L[f'any_{k}']['xtrack_med']:4.1f} len {L[f'any_{k}']['len_err']:+5.1f}"
        print(line, flush=True)
        if has_model:
            print(f"{'':20s} model: none {rec['none']['model']:6.1f}% (orc-speed {rec['none']['oracle']:5.1f}%, bias {rec['none']['bias_dps']:.2f} dps)"
                  f" | OSM {o['model_drift']:5.1f}% {o['model_pass']}/11 mean {o['model_mean_err']:5.1f}m"
                  + "".join(f" | L{k} {L[f'any_{k}']['model_drift']:5.1f}% mean {L[f'any_{k}']['model_mean_err']:5.1f}m" for k in (1, 3, 11))
                  + (f" | prev({L['previous_ride']['from']}) {L['previous_ride']['model_drift']:.1f}% {L['previous_ride']['model_pass']}/11" if 'previous_ride' in L else ""), flush=True)
    with open(os.path.join(HERE, "selfheal_results.json"), "w") as f:
        json.dump(dict(passes=[p[0] for p in PASSES], models=models, results=results), f, indent=1)
    print("saved")


if __name__ == "__main__":
    main()
