"""GPS ground-truth cleaning.

Measured 31 Aug (RESOURCES.md 7x): the 31-Aug rides were all recorded with the
phone in a trouser pocket, which blocks sky view and degrades the fix. Raw
GPS-derived path length was inflated by up to **100.8%** on one run -- a 825 m lap
reported as 1657 m -- purely from position spikes reaching 277 m/s.

Ground truth built from unfiltered GPS is therefore not ground truth. Every
distance, drift and speed-label computation must clean the fix sequence first.
The 30-Aug handlebar runs are barely affected (0.3-1.5%), which is why this went
unnoticed until pocket data arrived.
"""
from __future__ import annotations
import numpy as np

V_MAX_MPS = 15.0        # 54 km/h -- hard ceiling for a campus two-wheeler


def clean_track(xy: np.ndarray, t: np.ndarray, v_max: float = V_MAX_MPS):
    """Drop fixes that imply an impossible speed. Returns (xy, t, keep_mask).

    Iterative: removing a spike changes the implied speed of its neighbours, so
    the pass repeats until stable.
    """
    xy = np.asarray(xy, float)
    t = np.asarray(t, float)
    keep = np.ones(len(t), bool)
    for _ in range(10):
        idx = np.flatnonzero(keep)
        if len(idx) < 3:
            break
        d = np.linalg.norm(np.diff(xy[idx], axis=0), axis=1)
        dt = np.maximum(np.diff(t[idx]), 1e-3)
        bad = np.flatnonzero(d / dt > v_max)
        if len(bad) == 0:
            break
        # drop the LATER endpoint of each offending step
        keep[idx[bad + 1]] = False
    return xy[keep], t[keep], keep


def smooth_track(xy: np.ndarray, k: int = 5) -> np.ndarray:
    """Median filter along the fix sequence.

    Dropping spikes alone is not enough: it removes the worst fixes but leaves
    sub-threshold jitter, which still inflates path length because every wiggle
    adds. A median filter suppresses the jitter without deleting real travel.
    Validated against geometry -- the 31-Aug "fast" lap reads 1657 m raw,
    1204 m spike-dropped, and 955 m smoothed, against a known lap of ~910 m,
    with 88% of its cleaned fixes lying on that same lap.
    """
    import pandas as _pd
    xy = np.asarray(xy, float)
    if len(xy) < k:
        return xy
    return np.column_stack([
        _pd.Series(xy[:, i]).rolling(k, center=True, min_periods=1).median().to_numpy()
        for i in (0, 1)])


def clean_distance(xy: np.ndarray, t: np.ndarray, v_max: float = V_MAX_MPS) -> float:
    """Path length after removing impossible steps AND jitter."""
    cxy, _, _ = clean_track(xy, t, v_max)
    if len(cxy) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(smooth_track(cxy), axis=0), axis=1).sum())


def clean_for_truth(xy: np.ndarray, t: np.ndarray, v_max: float = V_MAX_MPS):
    """(xy, t) ready to be used as ground truth: spikes dropped, jitter smoothed."""
    cxy, ct, _ = clean_track(xy, t, v_max)
    return smooth_track(cxy), ct
