"""Contribution 7 — terrain-adaptive landmarks: curvature from road geometry.

Speed breakers are TAGGED data: OSM has none on our campus, and none on most
Indian roads. Road GEOMETRY is different -- every OSM way carries its shape, so
curvature is available everywhere, on every road, with nobody having surveyed
anything.

That matters because a ghat is defined by hairpins, not speed breakers. Where
the tagged alphabet has nothing to offer, the road's own shape does.

Curvature k = dtheta/ds (radians per metre); radius r = 1/k. A hairpin is
r ~ 8-20 m, a normal bend r ~ 50-200 m, a straight r -> infinity.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


def resample(xy: np.ndarray, step_m: float = 2.0):
    """Uniform arc-length resampling — curvature is meaningless on uneven spacing."""
    xy = np.asarray(xy, float)
    seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    keep = np.concatenate([[True], seg > 1e-9])
    xy, s = xy[keep], s[keep]
    if len(s) < 3:
        return s, xy
    grid = np.arange(0.0, s[-1], step_m)
    return grid, np.column_stack([np.interp(grid, s, xy[:, 0]),
                                  np.interp(grid, s, xy[:, 1])])


def curvature(xy: np.ndarray, step_m: float = 2.0, smooth_m: float = 12.0):
    """(arc_length, signed curvature 1/m). Positive = left turn."""
    s, p = resample(xy, step_m)
    if len(s) < 5:
        return s, np.zeros(len(s))
    n = max(int(smooth_m / step_m) | 1, 3)
    k = np.ones(n) / n
    px = np.convolve(p[:, 0], k, mode="same")
    py = np.convolve(p[:, 1], k, mode="same")
    dx, dy = np.gradient(px, step_m), np.gradient(py, step_m)
    ddx, ddy = np.gradient(dx, step_m), np.gradient(dy, step_m)
    denom = np.power(dx * dx + dy * dy, 1.5)
    kap = np.divide(dx * ddy - dy * ddx, np.maximum(denom, 1e-12))
    edge = n
    kap[:edge] = kap[edge]
    kap[-edge:] = kap[-edge - 1]
    return s, kap


@dataclass
class Turn:
    s: float            # arc length at the apex, metres
    radius_m: float
    direction: int      # +1 left, -1 right
    angle_deg: float    # total heading change through the turn
    length_m: float

    @property
    def kind(self):
        if self.radius_m < 20: return "hairpin"
        if self.radius_m < 60: return "sharp"
        return "bend"


def find_turns(s, kap, min_angle_deg=25.0, min_radius=4.0, max_radius=250.0):
    """Contiguous stretches of consistent curvature -> discrete turn landmarks."""
    thr = 1.0 / max_radius
    sign = np.where(np.abs(kap) < thr, 0, np.sign(kap)).astype(int)
    turns, i = [], 0
    while i < len(sign):
        if sign[i] == 0:
            i += 1; continue
        j = i
        while j < len(sign) and sign[j] == sign[i]:
            j += 1
        seg = slice(i, j)
        ds = s[j - 1] - s[i]
        ang = float(np.degrees(abs(np.trapezoid(kap[seg], s[seg]))))
        if ang >= min_angle_deg and ds > 3.0:
            kmax = float(np.max(np.abs(kap[seg])))
            r = 1.0 / max(kmax, 1e-9)
            if min_radius <= r <= max_radius:
                apex = s[i + int(np.argmax(np.abs(kap[seg])))]
                turns.append(Turn(s=float(apex), radius_m=r, direction=int(sign[i]),
                                  angle_deg=ang, length_m=float(ds)))
        i = j
    return turns


def profile_from_polyline(xy, **kw):
    s, kap = curvature(xy, **{k: v for k, v in kw.items() if k in ("step_m", "smooth_m")})
    return s, kap, find_turns(s, kap,
                              **{k: v for k, v in kw.items()
                                 if k in ("min_angle_deg", "min_radius", "max_radius")})
