"""Contribution ⑤ — an honest confidence radius.

The EKF's own covariance CANNOT be used for this. Measured (RESOURCES.md 8e):
it is overconfident by three to four orders of magnitude (ANEES ~19,000, coverage
1-5% against nominal 50-95%), and no amount of tuning fixes it — every setting
that improves calibration degrades accuracy, the best trade reaching only 50%
coverage at nominal 90% while drift went from 3.7% to 19.6%.

The cause is structural, not a tuning error. Every aiding source we use during a
blackout — road heading, road position, learned speed — has a *temporally
correlated* error, and a Kalman filter assumes white noise. Applying a road
constraint at 20 Hz treats one geometric fact as twenty independent facts per
second, so the covariance collapses while the true error grows. Adding a
speed-scale state to absorb the systematic part was tried and did not fix it.

So the reported confidence is derived empirically instead, from measured error
against distance travelled since the last GNSS fix:

    sigma(d) = k * d

`k` is fitted once from logged runs. On our 11 runs it is remarkably stable
(0.182-0.208), which is why a one-parameter model generalises. Leave-one-run-out
coverage: **86% at nominal 90%**, against 4% from the filter covariance.

The filter keeps its own covariance for FUSION — that is what the Kalman gain
needs. This model is only for what the user is shown.
"""
from __future__ import annotations
import numpy as np

K_DEFAULT = 0.19          # fitted on the 11 logged runs, leave-one-run-out


def sigma_at(distance_since_fix_m, k: float = K_DEFAULT, floor_m: float = 2.0):
    """1-sigma position radius after travelling this far without GNSS."""
    return np.maximum(k * np.asarray(distance_since_fix_m, float), floor_m)


def radius_for(distance_since_fix_m, p: float = 0.90, k: float = K_DEFAULT,
               floor_m: float = 2.0):
    """Radius of the circle that should contain the true position with prob p.

    Rayleigh quantile for a 2-D Gaussian: r = sigma * sqrt(-2 ln(1-p)).
    """
    return np.sqrt(-2.0 * np.log(1.0 - p)) * sigma_at(distance_since_fix_m, k, floor_m)


def fit_k(distances, errors, min_distance_m: float = 20.0) -> float:
    """Refit k from logged (distance-since-fix, error) pairs.

    Uses the Rayleigh relation sigma = sqrt(mean(e^2)/2) so k is a 1-sigma scale,
    not a mean-error scale.
    """
    d = np.asarray(distances, float)
    e = np.asarray(errors, float)
    m = d > min_distance_m
    if m.sum() < 10:
        return K_DEFAULT
    return float(np.sqrt(np.mean((e[m] / d[m]) ** 2) / 2.0))
