"""GPS on at the start, off mid-journey: the two 11 Sept evening demo rides, cut exactly where
the rider flipped the blackout switch (reconstructed from the phone card, RESOURCES 8aj) and
scored to the moment Finish Run was pressed.

Map states on the same stretch, same 11 held-out speed models:
  none     gyro heading from the last GPS bearing before the cut (best-case bias, sign fitted)
  learned  road learned from the GPS of the ride before this one
  osm      OpenStreetMap (reproduces the demo dashboards' cut numbers: 3.4% and 4.5%)
"""
import json, os, sys, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import selfheal_eval as SE
from dhruva.osm_road import road_from_osm
from dhruva.mapmatch import RoadPolyline
from evaluate_system import _extend

RIDES = [("19-37-43", "7:40 ride", 55.0, 172.0, 9), ("19-41-28", "7:44 ride", 48.0, 137.0, 10)]
cache = np.load(os.path.join(HERE, "speed_cache.npz"))
models = sorted({k.split("|")[1] for k in cache.files if k.split("|")[1] not in ("t", "wz")})
tracks = [SE.load(f) for _, f in SE.PASSES]
OUT, P = {}, {}

for stamp, name, cut_s, fin_s, prev_i in RIDES:
    ti = next(i for i, (_, f) in enumerate(SE.PASSES) if f.endswith(stamp))
    gps, lt = tracks[ti]
    t = cache[f"{stamp}|t"]; keep = (t >= lt[0]) & (t <= lt[-1]); t = t[keep]
    wz = cache[f"{stamp}|wz"][keep]; dt = float(np.median(np.diff(t)))
    V = [np.clip(cache[f"{stamp}|{m}"][keep], 0, None) for m in models]
    truth = np.column_stack([np.interp(t, lt, gps[:, 0]), np.interp(t, lt, gps[:, 1])])
    cum = np.r_[0, np.cumsum(np.linalg.norm(np.diff(truth, axis=0), axis=1))]
    i0 = int(np.searchsorted(t, cut_s)); i1 = min(int(np.searchsorted(t, fin_s)) + 1, len(t))
    tr = truth[i0:i1]; c = cum[i0:i1] - cum[i0]; dist = float(c[-1]); w = wz[i0:i1]
    vt = np.r_[0, np.diff(c)] / dt

    roads = {"learned": RoadPolyline(_extend(SE.orient(SE.fuse([tracks[prev_i][0]]), SE.resample(gps, 1.0)), 3000.0), 2.0),
             "osm": RoadPolyline(_extend(np.asarray(road_from_osm(SE.net, gps)[0], float), 3000.0), 2.0)}

    def on_road(road, v):
        return road.at(road.project(tr[0]) + np.cumsum(v[i0:i1]) * dt)[0]

    # no map: heading starts from the phone's last GPS bearing (15 m before the cut)
    j = min(int(np.searchsorted(cum, cum[i0] - 15.0)), i0 - 1); d = truth[i0] - truth[j]; psi0 = np.arctan2(d[1], d[0])
    vg = np.r_[0, np.diff(cum)] / dt; still = vg < 0.3
    cands = [0.0] + ([float(np.median(wz[still]))] if still.sum() > 30 else [])
    hd = np.unwrap(np.arctan2(*np.gradient(tr, axis=0)[:, ::-1].T)); m = c > 10; best = None
    for b0 in cands:
        res = {s: np.median(np.abs(np.angle(np.exp(1j * ((psi0 + s * np.cumsum(w - b0) * dt) - hd)[m])))) for s in (1, -1)}
        sg = min(res, key=res.get); pc = psi0 + sg * np.cumsum(w - b0) * dt
        e = float(np.linalg.norm(tr[0] + np.array([np.sum(vt * np.cos(pc)), np.sum(vt * np.sin(pc))]) * dt - tr[-1]))
        if best is None or e < best[0]: best = (e, sg, b0)
    psi = psi0 + best[1] * np.cumsum(w - best[2]) * dt

    def free(vv):
        return tr[0] + np.column_stack([np.cumsum(vv * np.cos(psi)), np.cumsum(vv * np.sin(psi))]) * dt

    def drift(p):
        return 100 * float(np.linalg.norm(p[-1] - tr[-1])) / dist

    rec = dict(ride=name, recording=stamp, cut_s=cut_s, finish_s=fin_s, gps_on_m=float(cum[i0]), gps_off_m=dist,
               learned_from=SE.PASSES[prev_i][0])
    for state in ("none", "learned", "osm"):
        f = free if state == "none" else (lambda v, r=roads[state]: on_road(r, v))
        dr = np.array([drift(f(v) if state == "none" else f(v)) for v in (V if state != "none" else [v[i0:i1] for v in V])])
        perfect = drift(free(vt) if state == "none" else roads[state].at(roads[state].project(tr[0]) + c)[0])
        rec[state] = dict(median=float(np.median(dr)), passes=int((dr < 10).sum()), lo=float(dr.min()), hi=float(dr.max()),
                          perfect_speed=float(perfect))
    OUT[stamp] = rec
    lrn = np.array([drift(on_road(roads["learned"], v)) for v in V])
    mi = int(np.argmin(np.abs(lrn - np.median(lrn)))); v = V[mi]
    k = np.arange(0, len(t), 5)
    P[f"{stamp}|truth"] = truth[k]; P[f"{stamp}|cut_xy"] = truth[i0]; P[f"{stamp}|fin_xy"] = tr[-1]
    P[f"{stamp}|t"] = t[k]; P[f"{stamp}|cut_s"] = cut_s; P[f"{stamp}|fin_s"] = fin_s
    kk = np.arange(0, i1 - i0, 5)
    P[f"{stamp}|learned"] = on_road(roads["learned"], v)[kk]; P[f"{stamp}|osm"] = on_road(roads["osm"], v)[kk]
    P[f"{stamp}|none"] = free(v[i0:i1])[kk]
    P[f"{stamp}|learned_end"] = on_road(roads["learned"], v)[-1]; P[f"{stamp}|none_end"] = free(v[i0:i1])[-1]
    P[f"{stamp}|prevmap"] = SE.orient(SE.fuse([tracks[prev_i][0]]), SE.resample(gps, 1.0))
    print(f"{name} ({stamp}): GPS on {cum[i0]:.0f} m, off {dist:.0f} m | "
          + " | ".join(f"{s} {rec[s]['median']:.1f}% [{rec[s]['lo']:.1f}-{rec[s]['hi']:.1f}] {rec[s]['passes']}/11 perfect {rec[s]['perfect_speed']:.1f}%" for s in ("none", "learned", "osm"))
          + f" | map model {models[mi]} learned {lrn[mi]:.1f}%", flush=True)
json.dump(OUT, open(os.path.join(HERE, "cut_results.json"), "w"), indent=1)
np.savez(os.path.join(HERE, "cut_paths.npz"), **P); print("saved")
