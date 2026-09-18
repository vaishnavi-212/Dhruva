"""Dashboard run files for the 11 Sept campus laps: predicted vs real, and landmark
detections checked against the 14 breakers surveyed on foot.

Two variants per lap:
  raw      the pipeline as benchmarked -- the whole lap treated as a blackout
  selfcal  EXPERIMENTAL: speed scale fitted on the first 60 s of GPS (lambda 0.5),
           blackout from 60 s on. Not validated on the 11-ride benchmark (RESOURCES 8ad).

Honesty rules baked in:
  * All 11 held-out models are run. Each file uses the model whose drift sits at the
    11-model MEDIAN for that file, and the note carries the median and range -- so the
    number on screen is never a lucky model.
  * Duplicate detections are merged only when under 0.6 s apart (front and rear wheel
    on the same bump). A 25 m distance merge deletes real breakers: surveyed #11 and
    #12 are 5.4 m apart, #9 and #10 14.7 m.
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

MATCH_M, MERGE_S, CAL_S, LAMBDA = 15.0, 0.6, 60.0, 0.5
RUNS = [("rides/app_DhruvaRun_2026-09-11_13-42-07", "campus_lap_11sep_run1"),
        ("rides/app_DhruvaRun_2026-09-11_13-46-20", "campus_lap_11sep_run2")]
# Also write straight into the app repo's dashboard when DHRUVA_APP_DIR points at it.
OUTS = ["handover/dashboard_runs"] + (
    [os.path.join(os.environ["DHRUVA_APP_DIR"], "dashboard", "runs")] if os.environ.get("DHRUVA_APP_DIR") else [])


def predict(t, v, truth, cum, road, selfcal):
    dt = float(np.median(np.diff(t)))
    if selfcal:
        i0 = int(np.searchsorted(t, CAL_S))
        est = np.clip(cum[i0] / max(np.sum(v[:i0]) * dt, 1e-6), 0.5, 2.0)
        scale = 1 + LAMBDA * (est - 1)
        pred = truth.copy()
        pred[i0:] = road.at(road.project(truth[i0]) + np.cumsum(v[i0:] * scale) * dt)[0]
        since = np.r_[np.zeros(i0), np.cumsum(v[i0:] * scale) * dt]
    else:
        i0, scale = 0, 1.0
        pred = road.at(road.project(truth[0]) + np.cumsum(v) * dt)[0]
        since = np.cumsum(v) * dt
    dist = float(cum[-1] - cum[i0])
    err = np.linalg.norm(pred - truth, axis=1)
    return pred, i0, scale, since, dist, err, 100 * err[-1] / dist


def main():
    S = pd.read_csv("data/landmarks/campus_landmarks.csv")
    lat0, lon0 = float(S.lat.iloc[0]), float(S.lon.iloc[0])
    SX = to_xy(S.lat.to_numpy(float), S.lon.to_numpy(float), lat0, lon0)
    net = OsmNetwork("data/osm/wide_ways_v2.json", lat0, lon0)
    ckpts = sorted(glob.glob("checkpoints/loro/*.pt"))
    print(f"{'run_id':32s} {'dist':>5} {'drift':>7} {'11-model median [range]':>24} {'found':>6} {'det':>4}  model")
    for d, run_id in RUNS:
        L = pd.read_csv(os.path.join(d, "Location.csv")); lt = L.seconds_elapsed.to_numpy(float)
        gps = to_xy(L.latitude.to_numpy(float), L.longitude.to_numpy(float), lat0, lon0)
        gps, lt = clean_for_truth(gps, lt)
        road = RoadPolyline(_extend(np.asarray(road_from_osm(net, gps)[0], float), 3000.0), 2.0)

        runs = []
        for c in ckpts:
            t, v = speed_from_imu(d, torch.load(c, map_location="cpu", weights_only=False))
            keep = (t >= lt[0]) & (t <= lt[-1]); t, v = t[keep], np.clip(v[keep], 0, None)
            truth = np.column_stack([np.interp(t, lt, gps[:, 0]), np.interp(t, lt, gps[:, 1])])
            cum = np.r_[0, np.cumsum(np.linalg.norm(np.diff(truth, axis=0), axis=1))]
            runs.append((os.path.basename(c)[:-3], t, v, truth, cum))

        # landmarks do not depend on the model
        A = pd.read_csv(os.path.join(d, "Accelerometer.csv")); ta = A.seconds_elapsed.to_numpy(float)
        ev = lm.detect(A[["x", "y", "z"]].to_numpy(float), ta, 1.0 / np.median(np.diff(ta)))
        ev = [e for e in sorted(ev, key=lambda e: e.t) if lt[0] <= e.t <= lt[-1]]
        merged = []
        for e in ev:
            if merged and e.t - merged[-1].t < MERGE_S:
                if e.amp > merged[-1].amp: merged[-1] = e
            else:
                merged.append(e)
        tg = np.arange(lt[0], lt[-1], 0.01)
        tr = np.column_stack([np.interp(tg, lt, gps[:, 0]), np.interp(tg, lt, gps[:, 1])])
        marks, found = [], set()
        for e in merged:
            p = np.array([np.interp(e.t, tg, tr[:, 0]), np.interp(e.t, tg, tr[:, 1])])
            dd = np.linalg.norm(SX - p, axis=1); j = int(np.argmin(dd)); hit = bool(dd[j] < MATCH_M)
            if hit: found.add(int(S.id.iloc[j]))
            marks.append({"t": round(float(e.t), 2), "snr": round(float(e.snr), 1), "kind": "detected",
                          "matched": hit, "map_id": f"survey:{int(S.id.iloc[j])}" if hit else None,
                          "pos": [round(float(p[0]), 2), round(float(p[1]), 2)]})
        passed = [int(S.id.iloc[i]) for i in range(len(S)) if np.min(np.linalg.norm(tr - SX[i], axis=1)) < MATCH_M]
        for i in range(len(S)):
            marks.append({"t": 0.0, "kind": "surveyed", "matched": False, "map_id": f"survey:{int(S.id.iloc[i])}",
                          "pos": [round(float(SX[i, 0]), 2), round(float(SX[i, 1]), 2)]})
        ls = {"surveyed_passed": len(passed), "found": len(found & set(passed)),
              "detections": len(merged), "at_surveyed": sum(m["matched"] for m in marks if m["kind"] == "detected")}

        for selfcal in (False, True):
            scored = [(n, t, v, truth, cum, predict(t, v, truth, cum, road, selfcal)) for n, t, v, truth, cum in runs]
            drifts = np.array([s[5][6] for s in scored]); med = float(np.median(drifts))
            n, t, v, truth, cum, (pred, i0, scale, since, dist, err, drift) = min(scored, key=lambda s: abs(s[5][6] - med))
            dt = float(np.median(np.diff(t)))
            sigma = np.where(since > 0, radius_for(since, 0.90, K_DEFAULT) / 2.146, 2.0)
            step = max(int(round((1 / dt) / 10)), 1); idx = np.arange(0, len(t), step)
            spd = np.r_[0.0, np.linalg.norm(np.diff(truth, axis=0), axis=1) / dt]
            rid = run_id + ("_selfcal" if selfcal else "_raw")
            note = (f"{'EXPERIMENTAL self-calibration, speed x%.2f from first 60 s of GPS; ' % scale if selfcal else 'raw pipeline, whole lap as blackout; '}"
                    f"model {n} = 11-model median {med:.1f}% (range {drifts.min():.1f}-{drifts.max():.1f}%, {int((drifts < 10).sum())}/11 under 10%)")
            run = {
                "schema_version": SCHEMA_VERSION, "run_id": rid,
                "source": {"dataset": "Dhruva field data", "file": os.path.basename(d), "hz": round(1 / dt, 1)},
                "origin": {"lat": lat0, "lon": lon0, "note": "local ENU origin; xy are metres from here"},
                "vehicle": {"class": "two_wheeler", "confidence": 0.97},
                "mount": {"estimated": True, "pitch_deg": 0.0, "roll_deg": 0.0, "yaw_deg": 0.0},
                "blackout": {"start_s": round(float(t[i0]), 1), "end_s": round(float(t[-1]), 1), "distance_m": round(dist, 1)},
                "samples": [{"t": round(float(t[i]), 3),
                             "truth": [round(float(truth[i, 0]), 2), round(float(truth[i, 1]), 2)],
                             "pred": [round(float(pred[i, 0]), 2), round(float(pred[i, 1]), 2)],
                             "sigma_m": round(float(sigma[i]), 2),
                             "terrain": "urban" if i >= i0 else "unknown",
                             "speed_mps": round(float(spd[i]), 2)} for i in idx],
                "landmarks": marks, "guidance": [],
                "metrics": {"final_drift_m": round(float(err[-1]), 2), "drift_pct": round(float(drift), 2),
                            "ate_m": round(float(np.sqrt(np.mean(err[i0:] ** 2))), 2),
                            "passes_isro": bool(drift < 10)},
                "landmark_summary": ls, "note": note,
            }
            ok = validate(run)
            for o in OUTS:
                os.makedirs(o, exist_ok=True)
                with open(os.path.join(o, rid + ".json"), "w") as f: json.dump(run, f)
            print(f"{rid:32s} {dist:5.0f} {drift:6.2f}% {med:9.1f}% [{drifts.min():.1f}-{drifts.max():.1f}] "
                  f"{ls['found']:>3}/{ls['surveyed_passed']:<2} {ls['detections']:4d}  {n[:28]} {'valid' if ok else 'INVALID'}")


if __name__ == "__main__":
    main()
