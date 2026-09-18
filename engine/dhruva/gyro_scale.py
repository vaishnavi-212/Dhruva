"""Per-run gyro scale estimation.

DIAGNOSTIC ONLY -- DO NOT APPLY IN THE PIPELINE.

Measured (RESOURCES.md 7s) on the six 30-Aug bike laps: there is NO evidence for a
per-device gyro scale factor. Orthogonal regression puts every run's 95% CI across 1.0,
and the one run with enough SNR to constrain k gives 1.00 [0.85, 1.18]. The apparent
0.20-0.61 "scale" seen with ordinary least squares is attenuation bias from a very noisy
GPS heading reference, not a sensor property -- the two OLS directions bracket 1.0 from
both sides (0.05-0.97 and 1.03-18.8).

Applying an estimated k here would inject noise to correct an effect that is not there.
Kept because the fit quality r is a useful per-run signal of how trustworthy the GPS
heading reference is.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class GyroScale:
    k: float                 # multiply level-frame gyro-Z by this to get world yaw rate
    r: float                 # fit quality
    n_points: int
    usable: bool

    def apply(self, wz):
        return np.asarray(wz, float) * self.k


def _unwrap_course(xy, t, min_step_m=2.0):
    """Heading over ground from the GPS track, keeping only real movement."""
    d = np.diff(xy, axis=0)
    seg = np.linalg.norm(d, axis=1)
    ok = seg >= min_step_m
    if ok.sum() < 5:
        return None, None
    hd = np.unwrap(np.arctan2(d[ok, 1], d[ok, 0]))
    tm = (0.5 * (t[:-1] + t[1:]))[ok]
    return tm, hd


def estimate(wz, t_imu, xy_gps, t_gps, min_turn_deg=60.0, min_r=0.6) -> GyroScale:
    """Fit world_heading_change ~ k * gyro_heading_change over the run."""
    tm, hd = _unwrap_course(xy_gps, t_gps)
    if tm is None:
        return GyroScale(1.0, 0.0, 0, False)

    # cumulative gyro heading, sampled at the GPS heading times
    cum = np.concatenate([[0.0], np.cumsum(np.asarray(wz, float)[:-1] * np.diff(t_imu))])
    g = np.interp(tm, t_imu, cum)

    # work in increments so a constant offset cannot bias the fit
    dg = np.diff(g)
    dh = np.diff(hd)
    m = np.isfinite(dg) & np.isfinite(dh) & (np.abs(dh) < np.deg2rad(60))
    if m.sum() < 10:
        return GyroScale(1.0, 0.0, int(m.sum()), False)
    dg, dh = dg[m], dh[m]

    # least squares through the origin, then one robust pass dropping outliers
    k = float(np.sum(dg * dh) / max(np.sum(dg * dg), 1e-12))
    res = dh - k * dg
    keep = np.abs(res) < 2.5 * np.std(res)
    if keep.sum() > 10:
        dg, dh = dg[keep], dh[keep]
        k = float(np.sum(dg * dh) / max(np.sum(dg * dg), 1e-12))

    r = float(np.corrcoef(dg, dh)[0, 1]) if len(dg) > 2 else 0.0
    total_turn = float(np.degrees(np.sum(np.abs(dh))))
    usable = (r >= min_r) and (total_turn >= min_turn_deg) and (0.1 < k < 5.0)
    return GyroScale(k if usable else 1.0, r, len(dg), usable)
