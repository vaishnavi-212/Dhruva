"""CONTRACT A — the model interface.  FROZEN. Do not change without telling everyone.

Every positioning model in Dhruva has this signature. The app, the dashboard and
the harness all call it the same way, so a fake model and the real TFLite model
are interchangeable.

    predict(acc, gyro, dt, init) -> (xy, sigma)

  acc   : (N,3) float, m/s^2, sensor frame, gravity already removed
  gyro  : (N,3) float, rad/s, sensor frame
  dt    : float, seconds between samples (1/hz)
  init  : dict, the vehicle state at the instant GNSS drops
            'xy'      : (2,) local ENU metres, position at t=0
            'v'       : (2,) local ENU m/s, velocity at t=0
            'heading' : float, radians, 0 = east, CCW positive

  returns
    xy    : (N,2) predicted positions, metres, same frame as init['xy']
    sigma : (N,) 1-sigma uncertainty radius in metres, or None

Rules
  * Never look at GNSS. The whole point is that it isn't there.
  * xy[0] must equal init['xy'].
  * Return exactly N rows. No resampling.
  * Pure function. No global state, no file writes.
"""
from __future__ import annotations
import numpy as np

SCHEMA_VERSION = "1.0"


def fake_model(acc, gyro, dt, init):
    """A deliberately WRONG model with the RIGHT shape.

    Build the app and the dashboard against this. It integrates heading from the
    gyro and assumes constant speed, so it drifts badly -- which is fine and in
    fact useful: if your UI looks sensible with this, it will look better with
    the real model.
    """
    n = len(acc)
    heading = init.get("heading", 0.0) + np.cumsum(gyro[:, 2]) * dt
    speed = float(np.linalg.norm(init["v"]))
    step = speed * dt
    xy = np.empty((n, 2), float)
    xy[0] = init["xy"]
    for i in range(1, n):
        xy[i] = xy[i-1] + step * np.array([np.cos(heading[i]), np.sin(heading[i])])
    sigma = 1.0 + 0.08 * (np.arange(n) * dt) ** 1.5   # grows, plausibly
    return xy, sigma


def validate(fn, hz: float = 10.0, n: int = 300) -> bool:
    """Check any model honours the contract. Run this before you ship a model."""
    rng = np.random.default_rng(0)
    acc = rng.normal(0, 1.5, (n, 3))
    gyro = rng.normal(0, 0.2, (n, 3))
    init = {"xy": np.array([12.0, -4.0]), "v": np.array([9.0, 1.0]), "heading": 0.11}
    dt = 1.0 / hz

    out = fn(acc, gyro, dt, init)
    problems = []
    if not (isinstance(out, tuple) and len(out) == 2):
        return _fail(["must return a 2-tuple (xy, sigma)"])
    xy, sigma = out
    xy = np.asarray(xy)
    if xy.shape != (n, 2):
        problems.append(f"xy shape {xy.shape}, expected ({n}, 2)")
    if xy.shape == (n, 2) and not np.allclose(xy[0], init["xy"], atol=1e-6):
        problems.append(f"xy[0] is {xy[0]}, must equal init['xy'] {init['xy']}")
    if not np.all(np.isfinite(xy)):
        problems.append("xy contains NaN or inf")
    if sigma is not None:
        sigma = np.asarray(sigma)
        if sigma.shape != (n,):
            problems.append(f"sigma shape {sigma.shape}, expected ({n},)")
        elif np.any(sigma < 0):
            problems.append("sigma has negative values")
    return _fail(problems) if problems else _ok()


def _ok():
    print("[contract A] PASS — model honours the interface")
    return True


def _fail(problems):
    print("[contract A] FAIL")
    for p in problems:
        print("  -", p)
    return False


if __name__ == "__main__":
    validate(fake_model)
