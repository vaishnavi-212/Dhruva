"""Dashboard files for the 11 Sept evening demo rides.

Per ride, two files:
  _whole  the entire ride treated as a GNSS blackout (laptop, end to end)
  _cut    GPS used until the moment the phone's blackout switch was flipped, then
          dead reckoning to the moment Finish Run was pressed -- the laptop's answer
          to exactly the stretch the phone card scored.

The cut and finish times are reconstructed from the phone card: the stretch of the
recorded GPS whose length equals "Distance without GPS" over the duration implied
by "Mean speed" (residual 0.2-0.3 m on both rides).

Each file uses the model whose drift is the 11-model median for that file. A
self-calibrated number (speed scale fitted on the GPS before the cut, lambda 0.5)
is printed for comparison but not written: it is not validated on the benchmark.
"""
from __future__ import annotations
import glob, json, os, sys
import numpy as np, pandas as pd, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dhruva.osm_road import OsmNetwork, road_from_osm, to_xy
from dhruva.mapmatch import RoadPolyline
from dhruva.gpsclean import clean_for_truth
from dhruva.uncertainty import radius_for, K_DEFAULT
from dhruva import landmarks as lm
from evaluate_system import speed_from_imu, _extend
from contracts.run_schema import SCHEMA_VERSION, validate

MATCH_M, MERGE_S, LAMBDA = 15.0, 0.6, 0.5
RUNS = [
    ("rides/app_DhruvaRun_2026-09-11_19-37-43", "demo_11sep_1937", 55.0, 172.0,
     "phone card at this cut: 9.0% PASS, held 16 km/h"),
    ("rides/app_DhruvaRun_2026-09-11_19-41-28", "demo_11sep_1941", 48.0, 137.0,
     "phone card at this cut: 25.9% FAIL, held 15 km/h (rider sped up to 19.6)"),
]
# Also write straight into the app repo's dashboard when DHRUVA_APP_DIR points at it.
OUTS = ["handover/dashboard_runs_evening"] + (
    [os.path.join(os.environ["DHRUVA_APP_DIR"], "dashboard", "runs")] if os.environ.get("DHRUVA_APP_DIR") else [])


def main():
    S = pd.read_csv("data/landmarks/campus_landmarks.csv")
    lat0, lon0 = float(S.lat.iloc[0]), float(S.lon.iloc[0])
    SX = to_xy(S.lat.to_numpy(float), S.lon.to_numpy(float), lat0, lon0)
    net = OsmNetwork("data/osm/wide_ways_v2.json", lat0, lon0)
    ckpts = sorted(glob.glob("checkpoints/loro/*.pt"))
    CK = [(os.path.basename(c)[:-3], torch.load(c, map_location="cpu", weights_only=False)) for c in ckpts]
    print(f"{'file':26s} {'denied':>6} {'drift':>7}  {'11-model median [range]':>24} {'pass':>5}   self-cal from pre-cut GPS")
    for d, rid, cut_s, fin_s, card in RUNS:
        L = pd.read_csv(os.path.join(d, "Location.csv")); lt = L.seconds_elapsed.to_numpy(float)
        gps = to_xy(L.latitude.to_numpy(float), L.longitude.to_numpy(float), lat0, lon0)
        gps, lt = clean_for_truth(gps, lt)
        road = RoadPolyline(_extend(np.asarray(road_from_osm(net, gps)[0], float), 3000.0), 2.0)

        per = []
        for name, ck in CK:
            t, v = speed_from_imu(d, ck)
            keep = (t >= lt[0]) & (t <= lt[-1]); t, v = t[keep], np.clip(v[keep], 0, None)
            dt = float(np.median(np.diff(t)))
            truth = np.column_stack([np.interp(t, lt, gps[:, 0]), np.interp(t, lt, gps[:, 1])])
            cum = np.r_[0, np.cumsum(np.linalg.norm(np.diff(truth, axis=0), axis=1))]
            out = {}
            for variant in ("whole", "cut"):
                if variant == "whole":
                    i0, i1 = 0, len(t)
                else:
                    i0 = int(np.searchsorted(t, cut_s)); i1 = min(int(np.searchsorted(t, fin_s)) + 1, len(t))
                tt, vv, tr, cm = t[:i1], v[:i1], truth[:i1], cum[:i1]
                pred = tr.copy()
                since = np.cumsum(vv[i0:]) * dt
                pred[i0:] = road.at(road.project(tr[i0]) + since)[0]
                err = np.linalg.norm(pred - tr, axis=1); dist = float(cm[-1] - cm[i0])
                out[variant] = dict(t=tt, truth=tr, pred=pred, since=np.r_[np.zeros(i0), since],
                                    i0=i0, dist=dist, err=err, drift=100 * err[-1] / dist, dt=dt)
                if variant == "cut":
                    est = np.clip(cm[i0] / max(np.sum(vv[:i0]) * dt, 1e-6), 0.5, 2.0); sc = 1 + LAMBDA * (est - 1)
                    pc = road.at(road.project(tr[i0]) + np.cumsum(vv[i0:] * sc) * dt)[0]
                    out["selfcal_cut"] = 100 * np.linalg.norm(pc[-1] - tr[-1]) / dist
            per.append((name, out))

        # landmarks: model-independent, merged only when <0.6 s apart
        A = pd.read_csv(os.path.join(d, "Accelerometer.csv")); ta = A.seconds_elapsed.to_numpy(float)
        ev = sorted([e for e in lm.detect(A[["x", "y", "z"]].to_numpy(float), ta, 1.0 / np.median(np.diff(ta)))
                     if lt[0] <= e.t <= lt[-1]], key=lambda e: e.t)
        merged = []
        for e in ev:
            if merged and e.t - merged[-1].t < MERGE_S:
                if e.amp > merged[-1].amp: merged[-1] = e
            else: merged.append(e)
        tg = np.arange(lt[0], lt[-1], 0.01)
        trg = np.column_stack([np.interp(tg, lt, gps[:, 0]), np.interp(tg, lt, gps[:, 1])])

        for variant in ("whole", "cut"):
            drifts = np.array([o[variant]["drift"] for _, o in per]); med = float(np.median(drifts))
            name, o = min(per, key=lambda p: abs(p[1][variant]["drift"] - med)); r = o[variant]
            t, truth, pred, i0 = r["t"], r["truth"], r["pred"], r["i0"]
            sigma = np.where(r["since"] > 0, radius_for(r["since"], 0.90, K_DEFAULT) / 2.146, 2.0)
            marks, found = [], set()
            for e in merged:
                if e.t > t[-1]: continue
                p = np.array([np.interp(e.t, tg, trg[:, 0]), np.interp(e.t, tg, trg[:, 1])])
                dd = np.linalg.norm(SX - p, axis=1); j = int(np.argmin(dd)); hit = bool(dd[j] < MATCH_M)
                if hit: found.add(int(S.id.iloc[j]))
                marks.append({"t": round(float(e.t), 2), "snr": round(float(e.snr), 1), "kind": "detected", "matched": hit,
                              "map_id": f"survey:{int(S.id.iloc[j])}" if hit else None, "pos": [round(float(p[0]), 2), round(float(p[1]), 2)]})
            passed = [int(S.id.iloc[i]) for i in range(len(S)) if np.min(np.linalg.norm(truth - SX[i], axis=1)) < MATCH_M]
            for i in range(len(S)):
                marks.append({"t": 0.0, "kind": "surveyed", "matched": False, "map_id": f"survey:{int(S.id.iloc[i])}",
                              "pos": [round(float(SX[i, 0]), 2), round(float(SX[i, 1]), 2)]})
            dt = r["dt"]; step = max(int(round((1 / dt) / 10)), 1); idx = np.arange(0, len(t), step)
            spd = np.r_[0.0, np.linalg.norm(np.diff(truth, axis=0), axis=1) / dt]
            note = (("whole ride as GNSS blackout" if variant == "whole" else f"GPS until {t[i0]:.0f} s, then dead reckoning; {card}")
                    + f"; model {name} = 11-model median {med:.1f}% (range {drifts.min():.1f}-{drifts.max():.1f}%, {int((drifts < 10).sum())}/11 under 10%)")
            run = {
                "schema_version": SCHEMA_VERSION, "run_id": f"{rid}_{variant}",
                "source": {"dataset": "Dhruva field data", "file": os.path.basename(d), "hz": round(1 / dt, 1)},
                "origin": {"lat": lat0, "lon": lon0, "note": "local ENU origin; xy are metres from here"},
                "vehicle": {"class": "two_wheeler", "confidence": 0.97},
                "mount": {"estimated": True, "pitch_deg": 0.0, "roll_deg": 0.0, "yaw_deg": 0.0},
                "blackout": {"start_s": round(float(t[i0]), 1), "end_s": round(float(t[-1]), 1), "distance_m": round(r["dist"], 1)},
                "samples": [{"t": round(float(t[i]), 3), "truth": [round(float(truth[i, 0]), 2), round(float(truth[i, 1]), 2)],
                             "pred": [round(float(pred[i, 0]), 2), round(float(pred[i, 1]), 2)], "sigma_m": round(float(sigma[i]), 2),
                             "terrain": "urban" if i >= i0 else "unknown", "speed_mps": round(float(spd[i]), 2)} for i in idx],
                "landmarks": marks, "guidance": [],
                "metrics": {"final_drift_m": round(float(r["err"][-1]), 2), "drift_pct": round(float(r["drift"]), 2),
                            "ate_m": round(float(np.sqrt(np.mean(r["err"][i0:] ** 2))), 2), "passes_isro": bool(r["drift"] < 10)},
                "landmark_summary": {"surveyed_passed": len(passed), "found": len(found & set(passed)), "detections": sum(1 for m in marks if m["kind"] == "detected"),
                                     "at_surveyed": sum(1 for m in marks if m["kind"] == "detected" and m["matched"])},
                "note": note,
            }
            ok = validate(run)
            for od in OUTS:
                os.makedirs(od, exist_ok=True)
                with open(os.path.join(od, run["run_id"] + ".json"), "w") as f: json.dump(run, f)
            sc = np.array([o["selfcal_cut"] for _, o in per])
            extra = f"   median {np.median(sc):5.1f}%  {int((sc < 10).sum())}/11" if variant == "cut" else ""
            print(f"{run['run_id']:26s} {r['dist']:5.0f}m {r['drift']:6.2f}%  {med:9.1f}% [{drifts.min():4.1f}-{drifts.max():4.1f}] {int((drifts < 10).sum()):>3}/11{extra}"
                  f"   found {run['landmark_summary']['found']}/{run['landmark_summary']['surveyed_passed']} {'valid' if ok else 'INVALID'}")


if __name__ == "__main__":
    main()
