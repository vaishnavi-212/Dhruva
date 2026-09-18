"""Simulate GNSS blackouts.

ISRO frames the benchmark by DISTANCE travelled, not by time, so the primary
generator here cuts windows of a target distance. A duration-based generator is
kept for the drift-vs-time curve used in the pitch deck.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .dataset import Trace


@dataclass
class Outage:
    i0: int
    i1: int
    target: float
    kind: str          # "distance" | "duration"

    def slice(self):
        return slice(self.i0, self.i1 + 1)


def by_distance(trace: Trace,
                targets_m=(50, 200, 500, 1000, 2000, 5000),
                stride_m: float = 250.0,
                max_per_target: int = 40) -> list[Outage]:
    """Sliding blackout windows of roughly `target` metres of travel."""
    d = trace.distance
    out: list[Outage] = []
    for target in targets_m:
        if d[-1] < target * 1.1:
            continue
        starts = np.arange(0.0, d[-1] - target, stride_m)
        if len(starts) > max_per_target:
            starts = starts[np.linspace(0, len(starts) - 1, max_per_target).astype(int)]
        for s in starts:
            i0 = int(np.searchsorted(d, s))
            i1 = int(np.searchsorted(d, s + target))
            if i1 - i0 < 5 or i1 >= len(d):
                continue
            out.append(Outage(i0, i1, float(target), "distance"))
    return out


def by_duration(trace: Trace,
                targets_s=(30, 60, 90, 180, 300, 600),
                stride_s: float = 20.0,
                max_per_target: int = 40) -> list[Outage]:
    """Blackout windows of fixed wall-clock length (for the drift-vs-time chart)."""
    t = trace.t
    out: list[Outage] = []
    for target in targets_s:
        if t[-1] < target * 1.1:
            continue
        starts = np.arange(0.0, t[-1] - target, stride_s)
        if len(starts) > max_per_target:
            starts = starts[np.linspace(0, len(starts) - 1, max_per_target).astype(int)]
        for s in starts:
            i0 = int(np.searchsorted(t, s))
            i1 = int(np.searchsorted(t, s + target))
            if i1 - i0 < 5 or i1 >= len(t):
                continue
            out.append(Outage(i0, i1, float(target), "duration"))
    return out


def snap_to_fixes(trace, outages: list[Outage]) -> list[Outage]:
    """Move blackout boundaries onto real GPS fixes.

    Ground truth between fixes is interpolated, so measuring final drift at an
    interpolated point measures our error against a guess. Snapping means the
    headline number is scored against a genuine satellite fix.
    """
    fix = getattr(trace, "fix_idx", None)
    if fix is None or len(fix) < 3:
        return outages
    out, seen = [], set()
    for o in outages:
        i0 = int(fix[np.searchsorted(fix, o.i0, "left").clip(0, len(fix) - 1)])
        i1 = int(fix[np.searchsorted(fix, o.i1, "right").clip(0, len(fix) - 1)])
        if i1 - i0 < 5 or (i0, i1) in seen:
            continue
        seen.add((i0, i1))
        out.append(Outage(i0, i1, o.target, o.kind))
    return out


def moving_only(trace: Trace, outages: list[Outage], min_kmph: float = 5.0) -> list[Outage]:
    """Drop windows where the vehicle was basically parked."""
    keep = []
    for o in outages:
        sl = o.slice()
        dist = np.linalg.norm(np.diff(trace.xy[sl], axis=0), axis=1).sum()
        dur = trace.t[o.i1] - trace.t[o.i0]
        if dur > 0 and (dist / dur) * 3.6 >= min_kmph:
            keep.append(o)
    return keep
