"""Contribution 4 — the self-healing landmark map.

Evidenced premise: OpenStreetMap has **156 highway ways / 1262 geometry points**
on this campus and **ZERO traffic_calming nodes**. The roads are mapped; the
speed breakers are not. Every landmark our system needs is missing from the
public map.

Design principle -- CONTRIBUTE WHEN YOU CAN SEE, BENEFIT WHEN YOU CANNOT:

  * A detection made while GNSS is HEALTHY can be geolocated, so it is a valid
    map contribution.
  * A detection made during a BLACKOUT cannot be geolocated (that is the whole
    problem), so it is used for correction only, never written back.

That asymmetry is what keeps the map from poisoning itself, and it is why the
system improves with fleet use rather than drifting.

Promotion is deliberately conservative: an observation becomes a published
landmark only after being seen on several independent passes, and a landmark
that stops being detected decays back out.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import json
import numpy as np

R = 6378137.0


@dataclass
class Observation:
    xy: np.ndarray
    kind: str
    run: str
    amp: float = 0.0


@dataclass
class Candidate:
    xy: np.ndarray
    kinds: list = field(default_factory=list)
    runs: set = field(default_factory=set)
    n: int = 0
    misses: int = 0

    @property
    def kind(self):
        return max(set(self.kinds), key=self.kinds.count) if self.kinds else ""

    @property
    def confidence(self):
        """Grows with independent passes, saturating."""
        return 1.0 - 0.5 ** len(self.runs)


class SelfHealingMap:
    def __init__(self, cluster_m=12.0, promote_runs=2, decay_limit=3):
        self.cluster_m = cluster_m
        self.promote_runs = promote_runs
        self.decay_limit = decay_limit
        self.cands: list[Candidate] = []

    # ---------- ingest ----------
    def observe(self, obs: list[Observation]):
        for o in obs:
            best, bd = None, 1e9
            for c in self.cands:
                d = float(np.linalg.norm(c.xy - o.xy))
                if d < bd:
                    best, bd = c, d
            if best is not None and bd <= self.cluster_m:
                w = best.n / (best.n + 1.0)
                best.xy = best.xy * w + o.xy * (1 - w)      # running mean
                best.kinds.append(o.kind); best.runs.add(o.run); best.n += 1
            else:
                self.cands.append(Candidate(xy=o.xy.copy(), kinds=[o.kind],
                                            runs={o.run}, n=1))

    def decay(self, run: str, traversed_xy=None, corridor_m: float = 25.0):
        """Age out landmarks the vehicle passed but did not detect.

        A landmark is only penalised if this run actually went past it -- driving
        a different road must not delete landmarks elsewhere. `traversed_xy` is
        the route taken; anything within `corridor_m` of it was passed.
        """
        removed = 0
        keep = []
        for c in self.cands:
            passed = True
            if traversed_xy is not None and len(traversed_xy):
                d = np.min(np.linalg.norm(np.asarray(traversed_xy) - c.xy, axis=1))
                passed = d <= corridor_m
            if passed and run not in c.runs:
                c.misses += 1
            if c.misses < self.decay_limit:
                keep.append(c)
            else:
                removed += 1
        self.cands = keep
        return removed

    # ---------- publish ----------
    @property
    def published(self) -> list[Candidate]:
        return [c for c in self.cands if len(c.runs) >= self.promote_runs]

    def positions(self) -> np.ndarray:
        p = self.published
        return np.array([c.xy for c in p]) if p else np.zeros((0, 2))

    def to_osm_json(self, lat0, lon0, path):
        """Export as OSM-style nodes so it can be contributed upstream."""
        feats = []
        for i, c in enumerate(self.published, 1):
            lat = lat0 + np.rad2deg(c.xy[1] / R)
            lon = lon0 + np.rad2deg(c.xy[0] / (R * np.cos(np.deg2rad(lat0))))
            feats.append({"type": "node", "id": -i, "lat": float(lat), "lon": float(lon),
                          "tags": {"traffic_calming": c.kind or "bump",
                                   "source": "Dhruva:crowdsourced",
                                   "observations": str(c.n),
                                   "confidence": f"{c.confidence:.2f}"}})
        json.dump({"version": 0.6, "elements": feats}, open(path, "w"), indent=1)
        return len(feats)


# Physical widths measured on the campus survey (paces x 0.75 m):
#   bump 1 pace = 0.75 m | hump 2 paces = 1.50 m | table 3 paces = 2.25 m
# Validated: surveyed widths predict event duration to 2% (0.210 s vs 0.206 s).
WIDTH_BINS = [(1.15, "bump"), (1.95, "hump"), (99.0, "table")]


def classify_by_width(duration_s: float, speed_mps: float) -> str:
    """Landmark type from width = duration x speed.

    MEASURED NOT TO WORK (RESOURCES.md 7o): across 63 surveyed crossings the
    inferred width is INVERTED -- bumps measure wider than humps despite being
    physically half the size -- because suspension response dominates the signal.
    Cohen's d is 0.19-0.45 on every feature tried.

    Kept for reference and for a future high-speed test where the duration gap
    may widen. DO NOT use it to populate the map: publish landmarks untyped.
    """
    if not (np.isfinite(duration_s) and np.isfinite(speed_mps)) or speed_mps <= 0.3:
        return ""
    w = duration_s * speed_mps
    for lim, name in WIDTH_BINS:
        if w < lim:
            return name
    return "table"


def observations_from_ride(detections_t, ride_t, ride_xy, run_name,
                           gnss_available=None, kinds=None):
    """Geolocate detections using GPS — only where GNSS is actually available."""
    out = []
    skipped = 0
    for i, td in enumerate(detections_t):
        if gnss_available is not None and not gnss_available(td):
            skipped += 1
            continue                      # blackout: cannot geolocate, do not contribute
        j = int(np.argmin(np.abs(ride_t - td)))
        out.append(Observation(xy=ride_xy[j].copy(),
                               kind=(kinds[i] if kinds else ""), run=run_name))
    if skipped:
        print(f"  [selfheal] {run_name}: {skipped} detections during blackout "
              f"NOT contributed (cannot be geolocated)")
    return out
