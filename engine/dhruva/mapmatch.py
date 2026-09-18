"""Map matching — the fix for the dominant error source.

Measured (RESOURCES.md 7b): heading, not speed, dominates. With a perfect speed
model, gyro-integrated heading still gives ~58% drift; with perfect heading and
a 28%-wrong speed model, 14.8%.

The insight that makes this tractable: on a road without branches -- a tunnel, a
ghat, a campus loop -- position is ONE-DIMENSIONAL. You do not need to know
where you are in the plane, only how far along the road you are. Heading then
comes from the road's own geometry instead of from an integrating gyro, which
removes the dominant error entirely.

This is the "binds the calculated position to known road networks and geometric
paths during a dropout" component the PS requires, plus the non-holonomic
constraint (a vehicle cannot leave the road sideways).
"""
from __future__ import annotations
import numpy as np


class RoadPolyline:
    """A road centreline, queried by arc length."""

    def __init__(self, xy: np.ndarray, resample_m: float = 2.0):
        xy = np.asarray(xy, float)
        seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        s = np.concatenate([[0.0], np.cumsum(seg)])
        keep = np.concatenate([[True], seg > 1e-6])
        xy, s = xy[keep], s[keep]
        # uniform resample so arc-length lookups are cheap and stable
        self.s = np.arange(0.0, s[-1], resample_m)
        self.xy = np.column_stack([np.interp(self.s, s, xy[:, 0]),
                                   np.interp(self.s, s, xy[:, 1])])
        d = np.gradient(self.xy, axis=0)
        self.heading = np.arctan2(d[:, 1], d[:, 0])
        self.length = float(self.s[-1])

    def at(self, arc):
        """Position and heading at arc length(s), clamped to the road."""
        a = np.clip(arc, 0.0, self.length)
        x = np.interp(a, self.s, self.xy[:, 0])
        y = np.interp(a, self.s, self.xy[:, 1])
        h = np.interp(a, self.s, np.unwrap(self.heading))
        return np.column_stack([x, y]), h

    def project(self, p):
        """Arc length of the nearest point on the road to p."""
        d = np.linalg.norm(self.xy - np.asarray(p, float), axis=1)
        return float(self.s[int(np.argmin(d))])


def constrained_track(road: RoadPolyline, speed: np.ndarray, dt: float,
                      start_xy, direction: int = +1) -> np.ndarray:
    """Propagate ALONG the road using speed only. Heading comes from the road.

    This is the whole point: the gyro never touches the heading, so gyro drift
    -- the dominant error -- cannot accumulate.
    """
    s0 = road.project(start_xy)
    arc = s0 + direction * np.cumsum(np.asarray(speed, float)) * dt
    pos, _ = road.at(arc)
    return pos


def snap(road: RoadPolyline, track: np.ndarray) -> np.ndarray:
    """Non-holonomic snap: nearest point on the road for each estimate.

    Removes cross-track error but leaves along-track error untouched -- which is
    exactly why landmarks are still needed.
    """
    out = np.empty_like(track)
    for i, p in enumerate(track):
        out[i] = road.at(road.project(p))[0][0]
    return out
