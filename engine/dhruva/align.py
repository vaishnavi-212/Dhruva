"""Contribution ② — In-Vehicle Alignment & Calibration Engine.

The PS requires a module that "automatically determines the phone's pitch, roll
and yaw relative to the vehicle's driving direction, whether the phone is
dashboard-mounted or in a mobile holder."

IO-VNBD's own ORIENTATION columns are unusable (see RESOURCES.md §5a), so we
estimate the mounting ourselves. Two stages, because they have very different
difficulty:

  TILT  (pitch & roll) -- easy and robust. Over a few seconds the mean
        accelerometer vector IS gravity, because vehicle acceleration averages
        out. That gives us "which way is down" without any other sensor.

  YAW   (which horizontal direction the vehicle drives in) -- the hard one.
        Gravity says nothing about it. We recover it while GNSS is healthy by
        finding the rotation that best aligns the phone's horizontal
        acceleration with the vehicle's forward acceleration from GPS, then
        carry that angle into the blackout.

Everything here is validated against physics, never assumed.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

G = 9.80665


def _smooth(x: np.ndarray, n: int) -> np.ndarray:
    if x.ndim == 1:
        return pd.Series(x).rolling(n, center=True, min_periods=1).mean().to_numpy()
    return np.column_stack([_smooth(x[:, i], n) for i in range(x.shape[1])])


def estimate_tilt(acc: np.ndarray, hz: float, window_s: float = 8.0) -> np.ndarray:
    """(N,3) accelerometer -> (N,3,3) rotations that put gravity on world -Z.

    Only fixes pitch and roll. Yaw is left arbitrary and handled separately.
    """
    n = max(int(window_s * hz) | 1, 3)
    down = _smooth(acc, n)
    down /= np.maximum(np.linalg.norm(down, axis=1, keepdims=True), 1e-9)

    # Build an orthonormal basis whose third row maps `down` onto +Z.
    ref = np.tile(np.array([1.0, 0.0, 0.0]), (len(acc), 1))
    bad = np.abs(np.einsum("ij,ij->i", ref, down)) > 0.9
    ref[bad] = np.array([0.0, 1.0, 0.0])

    east = np.cross(ref, down)
    east /= np.maximum(np.linalg.norm(east, axis=1, keepdims=True), 1e-9)
    north = np.cross(down, east)

    R = np.stack([east, north, down], axis=1)      # rows = new basis
    return R


def apply(R: np.ndarray, v: np.ndarray) -> np.ndarray:
    return np.einsum("nij,nj->ni", R, v)


def yaw_from_gps(acc_level: np.ndarray,
                 gps_speed: np.ndarray,
                 t: np.ndarray,
                 hz: float,
                 fix_idx: np.ndarray | None = None,
                 moving_mps: float = 3.0,
                 smooth_s: float = 9.0) -> tuple[float, float]:
    """Find the yaw angle that aligns level-frame horizontal accel with GPS.

    Returns (yaw_radians, correlation). Correlation is the quality signal:
    below ~0.3 the estimate should not be trusted.
    """
    n = max(int(smooth_s * hz) | 1, 3)

    # GPS is a ~9 s staircase: differentiating it at 10 Hz produces spikes.
    # Compute forward acceleration at the real fixes, then interpolate back.
    if fix_idx is not None and len(fix_idx) > 5:
        tf, sf = t[fix_idx], gps_speed[fix_idx]
        a_fix = np.gradient(sf, tf)
        a_fwd = np.interp(t, tf, a_fix)
    else:
        a_fwd = np.gradient(_smooth(gps_speed, n), 1.0 / hz)
    a_fwd = _smooth(a_fwd, n)

    # match the phone's bandwidth to GPS's (~0.11 Hz) or the correlation is noise
    ax = _smooth(acc_level[:, 0], n)
    ay = _smooth(acc_level[:, 1], n)

    m = (gps_speed > moving_mps) & np.isfinite(a_fwd)
    if m.sum() < 200:
        return 0.0, 0.0

    # correlation of (ax cos t + ay sin t) with a_fwd is maximised in closed form
    best_yaw, best_r = 0.0, 0.0
    for yaw in np.linspace(0, 2 * np.pi, 721, endpoint=False):
        proj = ax[m] * np.cos(yaw) + ay[m] * np.sin(yaw)
        sd = proj.std()
        if sd < 1e-6:
            continue
        r = float(np.corrcoef(proj, a_fwd[m])[0, 1])
        if r > best_r:
            best_yaw, best_r = float(yaw), r
    return best_yaw, best_r


def yaw_rotation(yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])


def check_tilt(R: np.ndarray, acc: np.ndarray, verbose: bool = True) -> dict:
    """Physics check: after tilt correction, gravity must sit on Z alone."""
    lv = apply(R, acc)
    mag = np.linalg.norm(lv, axis=1)
    vf = np.abs(lv[:, 2]) / np.maximum(mag, 1e-9)
    out = {
        "vertical_fraction_mean": float(np.mean(vf)),
        "vertical_fraction_p10": float(np.percentile(vf, 10)),
        "mean_z_ms2": float(np.mean(lv[:, 2])),
        "horizontal_rms_ms2": float(np.sqrt(np.mean(lv[:, :2] ** 2))),
    }
    if verbose:
        ok = out["vertical_fraction_mean"] > 0.99
        print("[align] tilt check")
        print(f"  vertical fraction : {out['vertical_fraction_mean']:.4f}  (want >0.99)")
        print(f"  p10               : {out['vertical_fraction_p10']:.4f}")
        print(f"  mean Z            : {out['mean_z_ms2']:+.3f} m/s²  (want ~{G:.2f})")
        print(f"  horizontal RMS    : {out['horizontal_rms_ms2']:.3f} m/s²  (motion + noise)")
        print(f"  -> {'PASS' if ok else 'FAIL'}")
    return out


def calibrate(trace, verbose: bool = True) -> dict:
    """Full alignment for one trace. Returns tilt rotations, yaw, and quality."""
    # Tilt needs the RAW accelerometer -- gravity is the signal here, so a
    # gravity-removed channel carries no information about "down".
    acc_for_tilt = trace.acc_raw if trace.acc_raw is not None else trace.acc
    R = estimate_tilt(acc_for_tilt, trace.hz)
    tilt = check_tilt(R, acc_for_tilt, verbose=verbose)
    level = apply(R, trace.acc)          # level-frame LINEAR acceleration

    yaw, r = (0.0, 0.0)
    if trace.speed is not None:
        yaw, r = yaw_from_gps(level, trace.speed, trace.t, trace.hz,
                              fix_idx=getattr(trace, "fix_idx", None))
        if verbose:
            print(f"[align] yaw vs GPS forward accel: {np.rad2deg(yaw):6.1f}°  r = {r:+.3f}"
                  f"   -> {'usable' if r > 0.3 else 'WEAK, do not trust'}")

    Ryaw = yaw_rotation(yaw)
    vehicle = np.einsum("ij,nj->ni", Ryaw, level)
    return {"R_tilt": R, "yaw": yaw, "yaw_corr": r,
            "acc_level": level, "acc_vehicle": vehicle, "tilt_check": tilt}


def yaw_from_magnetometer(mag_level: np.ndarray, gps_course: np.ndarray,
                          moving: np.ndarray) -> tuple[float, float]:
    """Mount yaw offset from the magnetometer. Returns (yaw_rad, concentration R).

    A rigidly mounted phone has a CONSTANT yaw offset between its own frame and
    the vehicle's, so the offset is (GPS course) - (magnetic heading in the level
    frame), averaged circularly over the ride.

    Measured (RESOURCES.md 8k): the *instantaneous* estimate is very noisy —
    about 78 deg of angular spread, well below any per-sample trust threshold.
    But the circular MEAN is highly repeatable for a rigid mount: the same phone
    on the same handlebar gives 0.7 deg and 3.4 deg standard deviation across
    repeat runs. In a trouser pocket the offset scatters over 61 deg, essentially
    uniform, because the mount is not rigid.

    So `R` (circular concentration) is not the right gate — a single run never
    reaches it. What makes this usable is that the mount type is itself
    detectable: contribution ② classifies handlebar vs pocket at 91%, so the
    system knows whether to trust an averaged yaw at all.

    This replaces `yaw_from_gps`, which correlates level-frame acceleration
    against GPS speed: median correlation 0.203 on two-wheelers and only 0.371 on
    IO-VNBD cars, against a 0.3 trust threshold — too weak either way.
    """
    m = np.asarray(moving, bool)
    if m.sum() < 30:
        return 0.0, 0.0
    head = np.arctan2(mag_level[m, 1], mag_level[m, 0])
    d = np.asarray(gps_course, float)[m] - head
    z = np.mean(np.exp(1j * d))
    return float(np.angle(z)), float(np.abs(z))
