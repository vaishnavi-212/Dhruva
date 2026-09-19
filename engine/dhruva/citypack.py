"""Offline city pack: roads you can route on and places you can search for, no network.

Built once on a laptop from OpenStreetMap (scripts/build_city_pack.py), shipped inside the app.
This module is also the REFERENCE the phone's Kotlin router and search are tested against, so it is
kept plain: flat arrays, Dijkstra, no libraries.

Pack format "dhruva-city/1":
  lat, lon        road node coordinates, degrees x 1e7, ints
  ways            [[node indices...], class, oneway, private]   oneway: 0 both ways, 1 along, -1 against
  places          [{name, alt, kind, lat, lon, arrive: [node indices]}]

Cost is travel TIME at a typical two-wheeler speed per road class, so the route prefers main roads
the way a person would, not the shortest lane-by-lane path. Private roads cost 5x (used only if
there is no other way in). "arrive" = road nodes within 40 m of a place's outline (60 m of a point),
worked out here so the phone does no geometry: for a lake, the route ends at the nearest reachable
shore road, not a road on the far side of the water.
"""
from __future__ import annotations
import heapq, math, re

R_EARTH = 6371008.8
CLASSES = ["trunk", "primary", "secondary", "tertiary", "unclassified", "residential", "service", "living_street", "road"]
KMH = [40.0, 35.0, 30.0, 25.0, 20.0, 20.0, 15.0, 10.0, 20.0]         # typical two-wheeler speed per class
LINK = {"motorway": "trunk", "motorway_link": "trunk", "trunk_link": "trunk", "primary_link": "primary",
        "secondary_link": "secondary", "tertiary_link": "tertiary"}
PRIVATE_FACTOR = 5.0
ARRIVE_POINT_M, ARRIVE_AREA_M = 60.0, 40.0


def metres(lat1, lon1, lat2, lon2):
    """Equirectangular distance; exact enough inside a city and identical in Kotlin."""
    k = math.cos(math.radians((lat1 + lat2) / 2))
    return R_EARTH * math.hypot(math.radians(lon2 - lon1) * k, math.radians(lat2 - lat1))


# ------------------------------------------------------------------ routing
class City:
    def __init__(self, pack: dict):
        self.lat = [v / 1e7 for v in pack["lat"]]
        self.lon = [v / 1e7 for v in pack["lon"]]
        self.ways = pack["ways"]
        self.places = pack["places"]
        n = len(self.lat)
        self.adj: list[list[tuple[int, float]]] = [[] for _ in range(n)]
        self.segs = []                                         # (a, b, way index) for snapping
        for wi, (nodes, cls, oneway, private) in enumerate(self.ways):
            f = 3.6 / KMH[cls] * (PRIVATE_FACTOR if private else 1.0)     # seconds per metre
            for a, b in zip(nodes[:-1], nodes[1:]):
                c = metres(self.lat[a], self.lon[a], self.lat[b], self.lon[b]) * f
                if oneway != -1: self.adj[a].append((b, c))
                if oneway != 1: self.adj[b].append((a, c))
                self.segs.append((a, b, wi))

    def snap(self, lat, lon):
        """Nearest point on any road: (a, b, t along a->b in 0..1, distance m, way index)."""
        best = None
        k = math.cos(math.radians(lat))
        for a, b, wi in self.segs:
            ax, ay = (self.lon[a] - lon) * k, self.lat[a] - lat
            bx, by = (self.lon[b] - lon) * k, self.lat[b] - lat
            dx, dy = bx - ax, by - ay
            L2 = dx * dx + dy * dy
            t = 0.0 if L2 == 0 else min(1.0, max(0.0, -(ax * dx + ay * dy) / L2))
            d2 = (ax + t * dx) ** 2 + (ay + t * dy) ** 2
            if best is None or d2 < best[0]:
                best = (d2, a, b, t, wi)
        d2, a, b, t, wi = best
        return a, b, t, math.radians(math.sqrt(d2)) * R_EARTH, wi

    def route(self, lat, lon, targets):
        """Fastest path from a position to ANY of the target nodes.
        Returns (polyline [(lat, lon)...], length_m, seconds), or None if unreachable."""
        targets = set(targets)
        a, b, t, _, wi = self.snap(lat, lon)
        _, cls, oneway, private = self.ways[wi]
        f = 3.6 / KMH[cls] * (PRIVATE_FACTOR if private else 1.0)
        L = metres(self.lat[a], self.lon[a], self.lat[b], self.lon[b])
        plat, plon = self.lat[a] + t * (self.lat[b] - self.lat[a]), self.lon[a] + t * (self.lon[b] - self.lon[a])
        dist, prev = {}, {}
        pq = []
        if oneway != 1: dist[a] = t * L * f; heapq.heappush(pq, (dist[a], a))          # travel b->a direction
        if oneway != -1 and (1 - t) * L * f < dist.get(b, math.inf):
            dist[b] = (1 - t) * L * f; heapq.heappush(pq, (dist[b], b))
        done = set()
        while pq:
            d, u = heapq.heappop(pq)
            if u in done: continue
            done.add(u)
            if u in targets:
                path = [u]
                while path[-1] in prev: path.append(prev[path[-1]])
                path.reverse()
                pts = [(plat, plon)] + [(self.lat[i], self.lon[i]) for i in path]
                length = sum(metres(*p, *q) for p, q in zip(pts[:-1], pts[1:]))
                return pts, length, d
            for v, c in self.adj[u]:
                nd = d + c
                if nd < dist.get(v, math.inf):
                    dist[v] = nd; prev[v] = u; heapq.heappush(pq, (nd, v))
        return None


# ------------------------------------------------------------------ search
ALIASES = {"kere": "lake", "lake": "kere", "stn": "station", "rly": "railway", "univ": "university", "hosp": "hospital"}


def tokens(s: str) -> list[str]:
    return [w for w in re.sub(r"[^a-z0-9 ]", " ", s.lower()).split() if w]


def _edit1(a: str, b: str) -> bool:
    """True if a and b differ by at most one insert, delete or substitution."""
    if abs(len(a) - len(b)) > 1: return False
    if len(a) > len(b): a, b = b, a
    i = j = diff = 0
    while i < len(a) and j < len(b):
        if a[i] == b[j]: i += 1; j += 1; continue
        diff += 1
        if diff > 1: return False
        if len(a) == len(b): i += 1
        j += 1
    return diff + (len(b) - j) + (len(a) - i) <= 1


def token_match(q: str, w: str, last: bool) -> int:
    """2 exact (or alias), 1 prefix of the word being typed / one typo, 0 no match."""
    if q == w or ALIASES.get(q) == w: return 2
    if last and len(q) >= 2 and w.startswith(q): return 1
    if len(q) >= 4 and _edit1(q, w): return 1
    return 0


def search(places, query: str, lat=None, lon=None, limit=8):
    """Every typed word must match a word of the name; best matches first, then nearest first."""
    qt = tokens(query)
    if not qt: return []
    out = []
    for i, p in enumerate(places):
        words = tokens(p["name"] + " " + p.get("alt", ""))
        score = 0
        for k, q in enumerate(qt):
            m = max((token_match(q, w, k == len(qt) - 1) for w in words), default=0)
            if m == 0: break
            score += m
        else:
            if any(len(tokens(n)) == len(qt) for n in (p["name"], p.get("alt", "")) if n):
                score += 1                                   # the whole name was typed, not just part of it
            d = metres(lat, lon, p["lat"], p["lon"]) if lat is not None else 0.0
            out.append((-score, d, i))
    out.sort()
    return [(places[i], d) for _, d, i in out[:limit]]
