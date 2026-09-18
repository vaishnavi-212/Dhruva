"""Replay the phone app's blackout logic on our two on-foot survey walks (29 Aug, handheld).

Mirrors NavigateActivity/RoadBinder/GyroBias as of main c379d25:
  * switch arms only if held speed >= 1.5 m/s; held speed = median of fixes with speed > 1.5 in last 30 s
  * free mode heading: gravity-projected yaw, bias learned while GPS speed < 0.5 and |w| < 0.05 rad/s
    (>= 150 samples, rejected if > 1 deg/s); start heading = last fix bearing
  * free mode speed: tested both as the held speed and as the last fix's own speed
  * road mode: advance along a route at the held speed, clamped at the route end
Blackouts start every 15 s from 45 s in, lasting 60 s and 90 s (a lap round a building).
"""
import os, sys, numpy as np, pandas as pd
from math import radians, cos
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))   # repo root
sys.path.insert(0, ROOT)
from dhruva.gpsclean import clean_for_truth
from dhruva.mapmatch import RoadPolyline

WALKS = {"brisk (94CC)": os.path.join(ROOT, "rides/survey/94CC_6RQ-2026-08-29_14-04-28"),
         "normal (Official_1)": os.path.join(ROOT, "rides/survey/Official_1-2026-08-29_13-11-19")}
LAT0, LON0 = 15.3669, 75.1272


def xy(la, lo):
    return np.column_stack([(lo - LON0) * 111320 * cos(radians(LAT0)), (la - LAT0) * 111320])


def load(d):
    L = pd.read_csv(os.path.join(d, "Location.csv")); G = pd.read_csv(os.path.join(d, "Gyroscope.csv"))
    Gr = pd.read_csv(os.path.join(d, "Gravity.csv"))
    lt = L.seconds_elapsed.to_numpy(float); p = xy(L.latitude.to_numpy(), L.longitude.to_numpy())
    tg = G.seconds_elapsed.to_numpy(float); w = G[["x", "y", "z"]].to_numpy(float)
    gr = np.column_stack([np.interp(tg, Gr.seconds_elapsed, Gr[c]) for c in "xyz"])
    gu = gr / np.maximum(np.linalg.norm(gr, axis=1, keepdims=True), 1e-6)
    return dict(lt=lt, p=p, spd=L.speed.to_numpy(float), brg=L.bearing.to_numpy(float), tg=tg, w=w, gu=gu)


W = {k: load(v) for k, v in WALKS.items()}
routes = {}
for k, v in W.items():
    cp, ct = clean_for_truth(v["p"], v["lt"]); routes[k] = RoadPolyline(cp, 2.0)

for name, v in W.items():
    other = [k for k in W if k != name][0]
    road = routes[other]                     # route learned from the OTHER walk
    lt, p, spd = v["lt"], v["p"], v["spd"]
    tg, w, gu = v["tg"], v["w"], v["gu"]
    dtg = np.r_[np.median(np.diff(tg)), np.diff(tg)]
    rows, refused = {60: [], 90: []}, 0
    starts = np.arange(45, lt[-1] - 95, 15)
    for c0 in starts:
        pre = (lt <= c0) & (lt > c0 - 30) & (spd > 1.5)
        if pre.sum() == 0:
            refused += 1; continue
        held = float(np.median(spd[pre]))
        ic = int(np.searchsorted(lt, c0)) - 1
        # gyro bias, as GyroBias.observe: GPS-stationary and phone still, before the cut
        gsp = np.interp(tg, lt, spd); pre_g = tg <= c0
        still = pre_g & (gsp < 0.5) & (np.linalg.norm(w, axis=1) < 0.05)
        bias = w[still].mean(axis=0) if still.sum() >= 150 else np.zeros(3)
        if np.degrees(np.linalg.norm(bias)) > 1.0: bias = np.zeros(3)
        yaw = np.einsum("ij,ij->i", w - bias, gu)
        for dur in (60, 90):
            m = (tg > c0) & (tg <= c0 + dur)
            truth_end = np.array([np.interp(c0 + dur, lt, p[:, 0]), np.interp(c0 + dur, lt, p[:, 1])])
            seg = (lt >= c0) & (lt <= c0 + dur)
            tp = p[seg]; dist = float(np.hypot(*np.diff(tp, axis=0).T).sum())
            if dist < 30: continue
            start = p[ic]; psi0 = np.radians(90 - v["brg"][ic])
            res = {}
            for sgn in (1, -1):
                psi = psi0 + sgn * np.cumsum(yaw[m] * dtg[m])
                for sname, s in (("held", held), ("lastfix", max(float(spd[ic]), 0))):
                    e = start + np.array([np.sum(s * np.cos(psi) * dtg[m]), np.sum(s * np.sin(psi) * dtg[m])])
                    res[(sgn, sname)] = 100 * np.linalg.norm(e - truth_end) / dist
            s0 = road.project(start)
            # direction along the other walk's route: whichever way the pre-cut track moved
            back = p[max(ic - 10, 0)]; direction = 1 if road.project(start) >= road.project(back) else -1
            re = road.at(s0 + direction * held * dur)[0][0]
            road_d = 100 * np.linalg.norm(re - truth_end) / dist
            fs = spd[seg]; stopped = float(np.sum(np.diff(lt[seg])[fs[1:] < 0.3]))
            rows[dur].append(dict(held=held, true=dist / dur, still=int(still.sum()), road=road_d, stopped=stopped,
                                  free_held=min(res[(1, "held")], res[(-1, "held")]), free_held_plus=res[(1, "held")],
                                  free_last=res[(1, "lastfix")]))
    print(f"\n== {name}: {len(starts)} possible cuts, switch REFUSED {refused} ({100*refused/max(len(starts),1):.0f}%)")
    for dur, R in rows.items():
        if not R: print(f"  {dur}s: no armed cuts"); continue
        D = pd.DataFrame(R)
        print(f"  {dur}s blackouts armed: {len(D)} | held {D.held.median():.2f} m/s vs true {D.true.median():.2f} m/s (over-read {100*(D.held/D.true-1).median():+.0f}%)")
        for col, lab in (("road", "road-bound on route from the other walk"), ("free_held_plus", "free, app sign convention, held speed"),
                         ("free_held", "free, best sign (hindsight), held speed"), ("free_last", "free, app sign, last-fix speed")):
            print(f"    {lab:44s} median {D[col].median():6.1f}%  under 10%: {100*np.mean(D[col] < 10):3.0f}%")
        print(f"    gyro bias learned on {100*np.mean(D.still >= 150):.0f}% of cuts")
        N = D[D.stopped < 3]
        if len(N):
            print(f"    windows with <3 s stopped (a continuous lap): {len(N)} | road-bound median {N.road.median():.1f}% "
                  f"(under 10%: {100*np.mean(N.road < 10):.0f}%) | free last-fix median {N.free_last.median():.1f}% | "
                  f"held over-read {100*(N.held/N.true-1).median():+.0f}%")
