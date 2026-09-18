"""Vehicle-class gating, including the pedestrian case.

Contribution ① classifies two-wheeler vs car from vibration (97%, RESOURCES 8b).
It has no pedestrian class, and a walking person lands inside the two-wheeler
region — vertical-accel std 1.260 and gyro RMS 0.917 against bikes at 1.45-2.78
and 0.48-0.87 (8u). On a phone that is always on, walking is the most common
motion there is, so the system must recognise it and stand down.

Two vibration-based detectors were tried and BOTH FAILED (RESOURCES 8w):

  * step-band spectral power (1.2-2.8 Hz): walking 65.8%, vehicles up to 73.6%.
    Scooter vibration on a rough campus road occupies the same band.
  * rhythmic periodicity (autocorrelation at step lags): walking 0.319 against
    a worst vehicle of 0.274 — only 1.17x, nowhere near a usable threshold.

What separates cleanly is **speed**, which we already have:

  walking          median 0.58 m/s,  8% of the time above 2.5 m/s
  slowest vehicle  median 4.14 m/s, 72% of the time above 2.5 m/s

A 7.2x gap on median speed with no overlap. The lesson is that this was never a
vibration problem, and the exotic features were a detour.
"""
from __future__ import annotations
import numpy as np

WALK_MAX_MPS = 2.5          # brisk walking tops out well below this
MIN_MOVING_FRACTION = 0.35  # a vehicle spends most of a trip above WALK_MAX_MPS
MIN_SECONDS = 45.0          # long enough that traffic does not look like walking


def is_pedestrian(speed_mps, dt: float) -> tuple[bool, float]:
    """Decide whether this is a person on foot. Returns (verdict, confidence).

    Evaluate while GNSS is healthy — normally the first minute of a session,
    the same window the mount and vehicle classifiers use. The verdict is then
    carried into a blackout rather than re-derived inside one.

    A vehicle crawling in traffic can momentarily look like walking, which is
    why the test needs `MIN_SECONDS` of sustained low speed. It is also why the
    consequence of a wrong call is small: a vehicle moving at walking pace is,
    for navigation purposes, doing much the same thing.
    """
    v = np.asarray(speed_mps, float)
    v = v[np.isfinite(v)]
    if len(v) * dt < MIN_SECONDS:
        return False, 0.0                     # too short to judge; assume vehicle
    moving = float(np.mean(v > WALK_MAX_MPS))
    med = float(np.median(v))
    pedestrian = moving < MIN_MOVING_FRACTION and med < WALK_MAX_MPS
    # confidence: how far the moving-fraction sits from the decision boundary
    conf = min(abs(moving - MIN_MOVING_FRACTION) / MIN_MOVING_FRACTION, 1.0)
    return bool(pedestrian), float(conf)


def classify(speed_mps, dt: float, vert_accel_std: float, gyro_rms: float) -> dict:
    """Full gate: pedestrian first, then vehicle class (8b thresholds)."""
    ped, conf = is_pedestrian(speed_mps, dt)
    if ped:
        return {"class": "pedestrian", "confidence": round(conf, 2),
                "navigate": False,
                "reason": "sustained speed below walking threshold"}
    cls = "two_wheeler" if vert_accel_std > 1.2 else "car"
    return {"class": cls, "confidence": 0.97, "navigate": True,
            "reason": "vertical vibration"}
