"""CONTRACT B — the run file.  FROZEN. Do not change without telling everyone.

One JSON file describes one GNSS blackout: what really happened, what we
predicted, which landmarks fired, and what the app said out loud. The web
dashboard and the Android app both render from this, so they can be built
before any model exists.

Generate real samples straight from IO-VNBD:

    python contracts/run_schema.py data/S-S1.csv --n 3
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np

SCHEMA_VERSION = "1.0"

SCHEMA = {
  "schema_version": "str",
  "run_id":         "str, unique",
  "source":   {"dataset": "str", "file": "str", "hz": "float"},
  "origin":   {"lat": "float", "lon": "float", "note": "local ENU origin; xy are metres from here"},
  "vehicle":  {"class": "car|two_wheeler|auto_rickshaw|bus|pedestrian", "confidence": "0..1"},
  "mount":    {"estimated": "bool", "pitch_deg": "float", "roll_deg": "float", "yaw_deg": "float"},
  "blackout": {"start_s": "float", "end_s": "float", "distance_m": "float"},
  "samples":  [{"t": "float, seconds from blackout start",
                "truth": "[x, y] metres, GNSS ground truth (null in a real blackout)",
                "pred":  "[x, y] metres, our estimate",
                "sigma_m": "float, 1-sigma radius",
                "terrain": "urban|ghat|highway|unknown",
                "speed_mps": "float"}],
  "landmarks": [{"t": "float", "type": "bump|hump|table|rumble_strip|hairpin",
                 "matched": "bool, did it match a map feature",
                 "map_id": "str|null, e.g. osm:node/123456",
                 "pos": "[x, y] metres",
                 "correction_m": "float, how far the fix moved us",
                 "contributed": "bool, was this written back to the map"}],
  "guidance": [{"t": "float", "text": "str, what the voice said",
                "confidence": "tight|medium|loose"}],
  "metrics":  {"final_drift_m": "float", "drift_pct": "float", "ate_m": "float",
               "passes_isro": "bool, drift_pct < 10"},
}

REQUIRED_TOP = ["schema_version", "run_id", "source", "origin", "vehicle",
                "mount", "blackout", "samples", "landmarks", "guidance", "metrics"]


def validate(run: dict) -> bool:
    problems = [f"missing key: {k}" for k in REQUIRED_TOP if k not in run]
    if "samples" in run:
        if not run["samples"]:
            problems.append("samples is empty")
        else:
            s = run["samples"][0]
            for k in ("t", "pred", "sigma_m", "terrain", "speed_mps"):
                if k not in s: problems.append(f"samples[0] missing '{k}'")
    if "metrics" in run:
        for k in ("final_drift_m", "drift_pct", "ate_m", "passes_isro"):
            if k not in run["metrics"]: problems.append(f"metrics missing '{k}'")
    if problems:
        print("[contract B] FAIL")
        for p in problems: print("  -", p)
        return False
    print(f"[contract B] PASS — {run['run_id']}: {len(run['samples'])} samples, "
          f"{len(run['landmarks'])} landmarks, drift {run['metrics']['drift_pct']:.2f}%")
    return True


def build(trace, outage, xy, sigma, run_id, landmarks=None, guidance=None) -> dict:
    sl = outage.slice()
    truth = trace.xy[sl]; t = trace.t[sl] - trace.t[outage.i0]
    err = np.linalg.norm(np.asarray(xy) - truth, axis=1)
    dist = float(np.linalg.norm(np.diff(truth, axis=0), axis=1).sum())
    spd = np.concatenate([[0.0], np.linalg.norm(np.diff(truth, axis=0), axis=1) / (1/trace.hz)])

    return {
      "schema_version": SCHEMA_VERSION,
      "run_id": run_id,
      "source": {"dataset": "IO-VNBD", "file": trace.name, "hz": trace.hz},
      "origin": {"lat": float(trace.lat[0]), "lon": float(trace.lon[0]),
                 "note": "local ENU origin; xy are metres from here"},
      "vehicle": {"class": "car", "confidence": 1.0},
      "mount": {"estimated": False, "pitch_deg": 0.0, "roll_deg": 0.0, "yaw_deg": 0.0},
      "blackout": {"start_s": float(trace.t[outage.i0]),
                   "end_s": float(trace.t[outage.i1]), "distance_m": dist},
      "samples": [
        {"t": round(float(t[i]), 3),
         "truth": [round(float(truth[i,0]),2), round(float(truth[i,1]),2)],
         "pred":  [round(float(xy[i][0]),2),  round(float(xy[i][1]),2)],
         "sigma_m": round(float(sigma[i]),2) if sigma is not None else None,
         "terrain": "unknown",
         "speed_mps": round(float(spd[i]),2)}
        for i in range(len(t))],
      "landmarks": landmarks or [],
      "guidance": guidance or [],
      "metrics": {"final_drift_m": round(float(err[-1]),2),
                  "drift_pct": round(float(err[-1]/dist*100),2) if dist > 1 else None,
                  "ate_m": round(float(np.sqrt(np.mean(err**2))),2),
                  "passes_isro": bool(dist > 1 and err[-1]/dist*100 < 10)},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv"); ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--outdir", default="contracts/samples")
    a = ap.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from harness import dataset, outage as om, evaluate
    from contracts.model_interface import fake_model

    tr = dataset.load(a.csv)
    outs = om.moving_only(tr, om.by_distance(tr, targets_m=(200, 500, 1000)))
    if not outs: sys.exit("no usable blackouts in this trace")

    os.makedirs(a.outdir, exist_ok=True)
    picks = [outs[i] for i in np.linspace(0, len(outs)-1, min(a.n, len(outs))).astype(int)]
    for k, o in enumerate(picks):
        sl = o.slice()
        init = evaluate._init_state(tr, o.i0)
        xy, sigma = fake_model(tr.acc[sl], tr.gyro[sl], 1/tr.hz, init)
        run = build(tr, o, xy, sigma, f"{os.path.splitext(tr.name)[0]}_blackout_{k+1:03d}")
        validate(run)
        p = os.path.join(a.outdir, run["run_id"] + ".json")
        json.dump(run, open(p, "w"), indent=1)
        print(f"           -> {p}  ({os.path.getsize(p)/1024:.0f} KB)")

    json.dump(SCHEMA, open(os.path.join(a.outdir, "_SCHEMA.json"), "w"), indent=2)
    print(f"\n[contract B] schema reference -> {a.outdir}/_SCHEMA.json")


if __name__ == "__main__":
    main()
