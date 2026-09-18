"""Run predictors over simulated blackouts and emit the submission artifacts."""
from __future__ import annotations
import json, os
from pathlib import Path
import numpy as np
from . import metrics, outage as outage_mod, plots
from .dataset import Trace


def _init_state(trace: Trace, i0: int) -> dict:
    """Vehicle state at the instant GNSS drops — the only thing carried in."""
    k = max(i0 - 1, 0)
    dt = 1.0 / trace.hz
    v = (trace.xy[i0] - trace.xy[k]) / dt if i0 > 0 else np.zeros(2)
    if i0 > 5:  # smooth over ~0.5 s
        v = (trace.xy[i0] - trace.xy[i0 - 5]) / (5 * dt)
    heading = float(np.arctan2(v[1], v[0])) if np.linalg.norm(v) > 0.1 else 0.0
    return {"xy": trace.xy[i0].copy(), "v": v, "heading": heading}


def run(trace: Trace,
        predictors: dict,
        outages: list,
        outdir: str = "results",
        make_plots: bool = True) -> dict:
    Path(outdir).mkdir(parents=True, exist_ok=True)
    dt = 1.0 / trace.hz
    results: dict[str, list[metrics.Result]] = {k: [] for k in predictors}

    for o in outages:
        sl = o.slice()
        acc, gyro = trace.acc[sl], trace.gyro[sl]
        truth, t = trace.xy[sl], trace.t[sl]
        init = _init_state(trace, o.i0)

        for name, fn in predictors.items():
            xy, sigma = fn(acc, gyro, dt, init)
            results[name].append(metrics.score(xy, truth, t, sigma))

    print(f"\n=== {trace.name} — {len(outages)} blackouts ===")
    for name, rs in results.items():
        metrics.print_summary(name, rs)

    summary = {name: metrics.summarise(rs) for name, rs in results.items()}
    with open(os.path.join(outdir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    if make_plots and outages:
        # pick a mid-length blackout for the hero position plot
        dists = [o.target for o in outages]
        pick = outages[int(np.argsort(dists)[len(dists) // 2])]
        sl = pick.slice()
        init = _init_state(trace, pick.i0)
        preds = {}
        for name, fn in predictors.items():
            xy, _ = fn(trace.acc[sl], trace.gyro[sl], dt, init)
            preds[name] = xy
        plots.position_plot(trace, pick, preds,
                            os.path.join(outdir, "position_plot.png"),
                            title=f"{trace.name} — {pick.target:.0f} m GNSS blackout")
        plots.drift_curve(results, os.path.join(outdir, "drift_curve.png"))
        print(f"\n  plots -> {outdir}/position_plot.png, {outdir}/drift_curve.png")

    return summary
