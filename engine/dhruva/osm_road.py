"""Build the road centreline from OpenStreetMap instead of the ride's own GPS.

Until now every drift number used a road polyline derived from the ride's own
GPS track, which is circular: the "map" came from the answer. In deployment the
map is OSM. This module closes that gap.

Measured on the campus bbox: OSM has **156 highway ways / 1262 geometry points**
and **zero traffic_calming nodes** -- the roads are mapped, the speed breakers are
not. That is precisely the gap contribution 4 (self-healing map) fills, and it is
now an evidenced claim rather than an assumption.
"""
from __future__ import annotations
import json
import numpy as np

R = 6378137.0
DRIVABLE = {"residential", "service", "primary", "secondary", "tertiary",
            "unclassified", "living_street", "busway", "road", "track"}


def to_xy(lat, lon, lat0, lon0):
    return np.column_stack([
        (np.deg2rad(np.asarray(lon)) - np.deg2rad(lon0)) * np.cos(np.deg2rad(lat0)) * R,
        (np.deg2rad(np.asarray(lat)) - np.deg2rad(lat0)) * R])


class OsmNetwork:
    """Drivable OSM ways as a set of segments, queryable by nearest point."""

    def __init__(self, path, lat0, lon0, drivable_only=True):
        d = json.load(open(path))
        self.lat0, self.lon0 = lat0, lon0
        self.segments = []          # list of (P, Q) endpoint pairs in local xy
        self.polylines = []
        for w in d["elements"]:
            if w["type"] != "way" or "geometry" not in w:
                continue
            hw = w.get("tags", {}).get("highway")
            if drivable_only and hw not in DRIVABLE:
                continue
            g = w["geometry"]
            xy = to_xy([p["lat"] for p in g], [p["lon"] for p in g], lat0, lon0)
            if len(xy) < 2:
                continue
            self.polylines.append(xy)
            for i in range(len(xy) - 1):
                self.segments.append((xy[i], xy[i + 1]))
        self.A = np.array([s[0] for s in self.segments])
        self.B = np.array([s[1] for s in self.segments])
        self.AB = self.B - self.A
        self.len2 = np.maximum((self.AB ** 2).sum(1), 1e-12)

    def snap(self, p):
        """Nearest point on the network to p -> (point, distance)."""
        t = np.clip(((p - self.A) * self.AB).sum(1) / self.len2, 0.0, 1.0)
        proj = self.A + t[:, None] * self.AB
        d = np.linalg.norm(proj - p, axis=1)
        j = int(np.argmin(d))
        return proj[j], float(d[j])

    def snap_track(self, track):
        """Snap a whole GPS track; returns snapped points and distances."""
        out = np.empty_like(track)
        dist = np.empty(len(track))
        for i, p in enumerate(track):
            out[i], dist[i] = self.snap(p)
        return out, dist


def road_from_osm(network: OsmNetwork, track, smooth_m: float = 6.0):
    """OSM-derived centreline following the route the vehicle actually took.

    Snapping alone is jittery where parallel ways exist, so the snapped path is
    lightly smoothed. The SHAPE comes from OSM; only the traversal order comes
    from the track.
    """
    snapped, d = network.snap_track(track)
    n = max(int(smooth_m), 3)
    k = np.ones(n) / n
    # Edge-pad before convolving. `mode="same"` zero-pads, which drags the first
    # and last n//2 points toward the coordinate ORIGIN -- measured on the 30-Aug
    # laps (RESOURCES.md 7t) as a ~108 m excursion at t<5 s that inflated
    # along-track path length by 25%, above ISRO's 10% drift limit, on 3 of 6
    # runs. Restoring only sm[:2]/sm[-2:] did not cover a 6-wide window.
    pad = n // 2
    sm = np.column_stack([
        np.convolve(np.concatenate([np.full(pad, snapped[0, i]),
                                    snapped[:, i],
                                    np.full(pad, snapped[-1, i])]), k, mode="valid")[:len(snapped)]
        for i in (0, 1)])
    return sm, d
