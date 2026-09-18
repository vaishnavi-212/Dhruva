"""Sequential landmark matching by dynamic programming.

Measured (RESOURCES.md 7d): greedy nearest-landmark matching gives a 33% pass
rate because a drifted estimate snaps to the WRONG landmark and the error then
cascades -- every later match is wrong too. Enforcing that a vehicle passes
landmarks in order, and only forwards, restores 92%.

This is sequence alignment, not nearest-neighbour lookup. Detections and map
landmarks are two ordered sequences; we find the monotonic alignment minimising
position discrepancy plus type mismatch, allowing gaps on both sides for missed
detections and false positives.

Two entry points:
    align()        offline Needleman-Wunsch style DP -- uses the whole sequence,
                   best accuracy, for replay and evaluation
    OnlineMatcher  causal, one detection at a time -- what actually runs on the
                   phone, where the future is unknown
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

GAP = -1  # "matched to nothing"


@dataclass
class MatchConfig:
    gate_m: float = 80.0        # never match beyond this position discrepancy
    sigma_m: float = 20.0       # scale for position cost
    type_penalty: float = 1.5   # cost of matching different landmark types
    skip_landmark: float = 1.0  # cost of passing a landmark without detecting it
    skip_detection: float = 2.0 # cost of a detection matching nothing (false positive)


def _pair_cost(ds, ls, dt_, lt_, cfg):
    d = abs(ds - ls)
    if d > cfg.gate_m:
        return np.inf
    c = (d / cfg.sigma_m) ** 2
    if dt_ and lt_ and dt_ != lt_:
        c += cfg.type_penalty
    return c


def align(det_s, map_s, det_types=None, map_types=None, cfg=MatchConfig()):
    """Monotonic alignment of detections to map landmarks.

    det_s : (N,) estimated along-road position of each detection, metres
    map_s : (M,) known along-road position of each map landmark, metres
    returns list of length N: index into map_s, or GAP (-1)
    """
    det_s = np.asarray(det_s, float)
    map_s = np.asarray(map_s, float)
    n, m = len(det_s), len(map_s)
    if n == 0 or m == 0:
        return [GAP] * n
    det_types = det_types or [None] * n
    map_types = map_types or [None] * m

    INF = float("inf")
    D = np.full((n + 1, m + 1), INF)
    P = np.zeros((n + 1, m + 1), dtype=np.int8)   # 0=match 1=skip-det 2=skip-map
    D[0, 0] = 0.0
    for j in range(1, m + 1):
        D[0, j] = D[0, j - 1] + cfg.skip_landmark
        P[0, j] = 2
    for i in range(1, n + 1):
        D[i, 0] = D[i - 1, 0] + cfg.skip_detection
        P[i, 0] = 1

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            c = _pair_cost(det_s[i - 1], map_s[j - 1],
                           det_types[i - 1], map_types[j - 1], cfg)
            best, arg = D[i - 1][j] + cfg.skip_detection, 1
            if D[i][j - 1] + cfg.skip_landmark < best:
                best, arg = D[i][j - 1] + cfg.skip_landmark, 2
            if c < INF and D[i - 1][j - 1] + c < best:
                best, arg = D[i - 1][j - 1] + c, 0
            D[i][j], P[i][j] = best, arg

    out = [GAP] * n
    i, j = n, m
    while i > 0 and j > 0:
        if P[i][j] == 0:
            out[i - 1] = j - 1; i -= 1; j -= 1
        elif P[i][j] == 1:
            i -= 1
        else:
            j -= 1
    return out


class OnlineMatcher:
    """Causal matcher — what runs on the phone. Never looks ahead."""

    def __init__(self, map_s, map_types=None, cfg=MatchConfig()):
        self.map_s = np.asarray(map_s, float)
        self.map_types = map_types or [None] * len(self.map_s)
        self.cfg = cfg
        self.last = -1                 # last landmark index consumed

    def match(self, det_s, det_type=None):
        """Return map index for this detection, or GAP. Monotonic by construction."""
        best, arg = np.inf, GAP
        for j in range(self.last + 1, len(self.map_s)):
            if self.map_s[j] - det_s > self.cfg.gate_m:
                break                  # map is ordered; nothing further can be closer
            c = _pair_cost(det_s, self.map_s[j], det_type, self.map_types[j], self.cfg)
            if c < best:
                best, arg = c, j
        if arg != GAP:
            self.last = arg
        return arg
