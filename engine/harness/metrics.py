"""Metrics, built around ISRO's stated benchmark for SIH26168.

The headline number ISRO asks for:

    positional drift < 10% of the total distance travelled

with two worked gates quoted in the problem statement:
    * < 5 m drift over a 50 m GNSS-denied stretch, in under 1 minute
    * < 100 m drift over a 1 km GNSS-denied stretch at 60 kmph

Everything else here is supporting evidence.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import numpy as np

ISRO_DRIFT_LIMIT_PCT = 10.0


@dataclass
class Result:
    """Scores for one simulated blackout."""
    distance_m: float          # ground-truth distance travelled during outage
    duration_s: float
    mean_speed_kmph: float
    final_drift_m: float       # error at the moment GNSS returns
    drift_pct: float           # ISRO headline: final_drift / distance * 100
    ate_m: float               # RMSE across the whole blackout
    max_err_m: float
    cep50_m: float
    cep95_m: float
    passes_isro: bool
    gate_50m: bool | None      # < 5 m over 50 m in < 60 s
    gate_1km: bool | None      # < 100 m over 1 km at ~60 kmph
    coverage_90: float | None  # fraction of truth inside the 90% circle

    def as_dict(self) -> dict:
        return asdict(self)

    def line(self) -> str:
        flag = "PASS" if self.passes_isro else "FAIL"
        return (f"{self.distance_m:7.0f} m  {self.duration_s:6.1f} s  "
                f"{self.mean_speed_kmph:5.1f} kmph | "
                f"drift {self.final_drift_m:7.1f} m = {self.drift_pct:5.2f}%  "
                f"ATE {self.ate_m:6.1f} m  [{flag}]")


def _cep(err: np.ndarray, q: float) -> float:
    return float(np.percentile(err, q))


def score(pred_xy: np.ndarray,
          true_xy: np.ndarray,
          t: np.ndarray,
          sigma_m: np.ndarray | None = None) -> Result:
    """Score one blackout.

    pred_xy : (N,2) predicted local ENU positions during the outage
    true_xy : (N,2) ground truth for the same samples
    t       : (N,) seconds
    sigma_m : (N,) optional 1-sigma radius, for calibration checking
    """
    pred_xy = np.asarray(pred_xy, float)
    true_xy = np.asarray(true_xy, float)
    if pred_xy.shape != true_xy.shape:
        raise ValueError(f"shape mismatch {pred_xy.shape} vs {true_xy.shape}")

    err = np.linalg.norm(pred_xy - true_xy, axis=1)
    step = np.linalg.norm(np.diff(true_xy, axis=0), axis=1)
    distance = float(step.sum())
    duration = float(t[-1] - t[0])
    speed_kmph = (distance / duration * 3.6) if duration > 0 else 0.0

    final_drift = float(err[-1])
    drift_pct = (final_drift / distance * 100.0) if distance > 1e-6 else float("inf")

    gate_50m = None
    if 40 <= distance <= 60 and duration < 60:
        gate_50m = final_drift < 5.0
    gate_1km = None
    if 900 <= distance <= 1100:
        gate_1km = final_drift < 100.0

    coverage = None
    if sigma_m is not None:
        # 90% circle for a 2-D Gaussian is ~2.1460 sigma (Rayleigh)
        coverage = float(np.mean(err <= 2.1460 * np.asarray(sigma_m, float)))

    return Result(
        distance_m=distance,
        duration_s=duration,
        mean_speed_kmph=speed_kmph,
        final_drift_m=final_drift,
        drift_pct=drift_pct,
        ate_m=float(np.sqrt(np.mean(err ** 2))),
        max_err_m=float(err.max()),
        cep50_m=_cep(err, 50),
        cep95_m=_cep(err, 95),
        passes_isro=drift_pct < ISRO_DRIFT_LIMIT_PCT,
        gate_50m=gate_50m,
        gate_1km=gate_1km,
        coverage_90=coverage,
    )


def summarise(results: list[Result]) -> dict:
    """Aggregate over many blackouts — this is what goes in the report."""
    if not results:
        return {}
    pct = np.array([r.drift_pct for r in results])
    ate = np.array([r.ate_m for r in results])
    return {
        "n_outages":        len(results),
        "drift_pct_mean":   float(pct.mean()),
        "drift_pct_median": float(np.median(pct)),
        "drift_pct_p90":    float(np.percentile(pct, 90)),
        "drift_pct_worst":  float(pct.max()),
        "ate_mean_m":       float(ate.mean()),
        "pass_rate":        float(np.mean([r.passes_isro for r in results])),
        "gate_50m_pass":    _gate_rate(results, "gate_50m"),
        "gate_1km_pass":    _gate_rate(results, "gate_1km"),
    }


def _gate_rate(results: list[Result], attr: str):
    vals = [getattr(r, attr) for r in results if getattr(r, attr) is not None]
    return float(np.mean(vals)) if vals else None


def print_summary(name: str, results: list[Result]) -> None:
    s = summarise(results)
    if not s:
        print(f"  {name}: no results")
        return
    print(f"\n  {name}")
    print(f"    outages          : {s['n_outages']}")
    print(f"    drift % mean/med : {s['drift_pct_mean']:.2f} / {s['drift_pct_median']:.2f}")
    print(f"    drift % p90/worst: {s['drift_pct_p90']:.2f} / {s['drift_pct_worst']:.2f}")
    print(f"    ATE mean         : {s['ate_mean_m']:.1f} m")
    print(f"    ISRO pass rate   : {s['pass_rate']*100:.0f}%  (target: drift < 10% of distance)")
    for gate, label in (("gate_50m_pass", "50 m gate  (<5 m)"),
                        ("gate_1km_pass", "1 km gate (<100 m)")):
        if s.get(gate) is not None:
            print(f"    {label}: {s[gate]*100:.0f}%")
