"""Road network as a GRAPH — the structure requirement 3 needs.

`OsmNetwork` treats the map as a bag of segments: good for snapping, useless for
asking "which way did we go at that junction?". The 1-D `RoadPolyline` is worse
in this respect — it presupposes the answer, because the route was built from the
run's own GPS.

This builds a real graph: OSM ways split at every shared node, so a junction is a
vertex with degree > 2 and the choices at it are enumerable.
"""
from __future__ import annotations
import json
import numpy as np
from dhruva.osm_road import to_xy, DRIVABLE


class RoadGraph:
    def __init__(self, path, lat0, lon0, drivable_only=True):
        d = json.load(open(path))
        ways = []
        node_use = {}
        for w in d["elements"]:
            if w.get("type") != "way" or "geometry" not in w:
                continue
            if drivable_only and w.get("tags", {}).get("highway") not in DRIVABLE:
                continue
            nodes = w.get("nodes") or []
            g = w["geometry"]
            if len(g) < 2 or len(nodes) != len(g):
                continue
            xy = to_xy([p["lat"] for p in g], [p["lon"] for p in g], lat0, lon0)
            ways.append((nodes, xy))
            for n in nodes:
                node_use[n] = node_use.get(n, 0) + 1

        # split each way at nodes shared with another way -> graph edges
        self.edges = []        # (u_node, v_node, polyline xy)
        for nodes, xy in ways:
            cut = [0]
            for i in range(1, len(nodes) - 1):
                if node_use.get(nodes[i], 0) > 1:
                    cut.append(i)
            cut.append(len(nodes) - 1)
            for a, b in zip(cut[:-1], cut[1:]):
                if b - a < 1:
                    continue
                seg = xy[a:b + 1]
                if len(seg) >= 2 and np.linalg.norm(seg[-1] - seg[0]) > 1e-6:
                    self.edges.append((nodes[a], nodes[b], seg))

        self.node_xy = {}
        for u, v, seg in self.edges:
            self.node_xy.setdefault(u, seg[0])
            self.node_xy.setdefault(v, seg[-1])
        # adjacency: node -> list of (edge_index, forward?)
        self.adj = {}
        for i, (u, v, _) in enumerate(self.edges):
            self.adj.setdefault(u, []).append((i, True))
            self.adj.setdefault(v, []).append((i, False))
        self.lengths = np.array([
            float(np.linalg.norm(np.diff(s, axis=0), axis=1).sum()) for _, _, s in self.edges])

    # ---------- geometry helpers ----------
    def edge_points(self, i, step=5.0):
        """Resample edge i every `step` metres -> (arc, xy, heading)."""
        seg = self.edges[i][2]
        d = np.linalg.norm(np.diff(seg, axis=0), axis=1)
        s = np.concatenate([[0.0], np.cumsum(d)])
        if s[-1] < 1e-6:
            return np.zeros(1), seg[:1], np.zeros(1)
        a = np.arange(0.0, s[-1], step)
        if len(a) == 0:
            a = np.array([0.0])
        xy = np.column_stack([np.interp(a, s, seg[:, 0]), np.interp(a, s, seg[:, 1])])
        g = np.gradient(xy, axis=0) if len(xy) > 1 else np.array([[1.0, 0.0]])
        return a, xy, np.arctan2(g[:, 1], g[:, 0])

    def junction_degree(self, node):
        return len(self.adj.get(node, []))

    def stats(self):
        deg = np.array([len(v) for v in self.adj.values()])
        return dict(edges=len(self.edges), nodes=len(self.node_xy),
                    junctions=int((deg > 2).sum()), total_km=float(self.lengths.sum() / 1000))
