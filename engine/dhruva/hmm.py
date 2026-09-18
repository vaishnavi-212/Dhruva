"""ISRO requirement 3 — HMM map matching over a BRANCHING road graph.

The 1-D tracker that produces our benchmark presupposes the route: `road_from_osm`
is handed the run's own GPS to decide which ways were used and in what order. On
a real GNSS-denied drive nothing supplies that. At a junction the vehicle can go
several ways and the estimator must decide which.

This is the decoder that decides. Hidden state is a position on the road GRAPH —
(edge, offset along it). Observations are IMU-derived, because GNSS is gone:

    distance travelled in the step   ->  constrains WHERE you can now be
    heading CHANGE over the step     ->  constrains WHICH way you turned

Heading *change* is used rather than absolute heading because integrated gyro
heading drifts, while the change over a few seconds does not. This is
contribution ⑦'s turn matching generalised from one polyline to a graph.

Decoding is a beam-search Viterbi: exact within the beam, and linear in path
length rather than exponential in the number of junctions passed.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class HmmConfig:
    step_m: float = 10.0         # distance between observations
    beam: int = 400              # hypotheses kept per step
    sigma_turn_deg: float = 22.0 # how much heading-change disagreement to tolerate
    sigma_dist_m: float = 6.0    # slack on distance travelled
    max_branch: int = 4          # successors considered at a junction
    w_junction: float = 8.0      # route-plausibility prior: a driver follows roads
                                 # rather than zigzagging through side streets.
                                 # Measured (8m): median cross-track 79.4 -> 59.0 m.


class _Hyp:
    __slots__ = ("edge", "fwd", "off", "cost", "prev", "node")

    def __init__(self, edge, fwd, off, cost, prev, node):
        self.edge, self.fwd, self.off = edge, fwd, off
        self.cost, self.prev, self.node = cost, prev, node


def _heading_at(graph, e, fwd, off):
    seg = graph.edges[e][2]
    d = np.linalg.norm(np.diff(seg, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(d)])
    o = float(np.clip(off, 0.0, max(s[-1], 1e-6)))
    i = int(np.clip(np.searchsorted(s, o) - 1, 0, len(d) - 1))
    v = seg[i + 1] - seg[i]
    h = np.arctan2(v[1], v[0])
    return h if fwd else h + np.pi


def _xy_at(graph, e, fwd, off):
    seg = graph.edges[e][2]
    d = np.linalg.norm(np.diff(seg, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(d)])
    o = float(np.clip(off, 0.0, max(s[-1], 1e-6)))
    return np.array([np.interp(o, s, seg[:, 0]), np.interp(o, s, seg[:, 1])])


def _advance(graph, h, dist, cfg, out):
    """Move hypothesis h forward by `dist`, branching at junctions."""
    L = graph.lengths[h.edge]
    off = h.off + dist if h.fwd else h.off - dist
    if 0.0 <= off <= L:
        out.append(_Hyp(h.edge, h.fwd, off, h.cost, h, h.node))
        return
    # ran off the end -> we are at a node, take every outgoing edge
    over = off - L if off > L else -off
    node = graph.edges[h.edge][1] if h.fwd else graph.edges[h.edge][0]
    succ = graph.adj.get(node, [])[: cfg.max_branch * 2]
    for (e2, starts_here) in succ:
        if e2 == h.edge:
            continue
        L2 = graph.lengths[e2]
        if L2 < 1e-6:
            continue
        if starts_here:
            out.append(_Hyp(e2, True, min(over, L2), h.cost, h, node))
        else:
            out.append(_Hyp(e2, False, max(L2 - over, 0.0), h.cost, h, node))


def decode(graph, start_xy, step_dists, step_turns, cfg: HmmConfig = HmmConfig(),
           start_radius: float = 40.0):
    """Viterbi over the road graph.

    step_dists : (K,) metres travelled in each step, from the speed estimate
    step_turns : (K,) heading change in each step, radians, from the gyro
    Returns (path_xy (K+1,2), edge_sequence, junctions_passed).
    """
    start_xy = np.asarray(start_xy, float)
    # seed: every edge point within start_radius of the last known fix
    hyps = []
    for e in range(len(graph.edges)):
        a, xy, _ = graph.edge_points(e, cfg.step_m)
        d = np.linalg.norm(xy - start_xy, axis=1)
        j = int(np.argmin(d))
        if d[j] > start_radius:
            continue
        for fwd in (True, False):
            hyps.append(_Hyp(e, fwd, float(a[j]), float(d[j]) / 10.0, None, None))
    if not hyps:
        return None, [], 0
    hyps = sorted(hyps, key=lambda h: h.cost)[: cfg.beam]

    st = np.radians(cfg.sigma_turn_deg)
    for k in range(len(step_dists)):
        nxt = []
        for h in hyps:
            h0 = _heading_at(graph, h.edge, h.fwd, h.off)
            cand = []
            _advance(graph, h, float(step_dists[k]), cfg, cand)
            for c in cand:
                h1 = _heading_at(graph, c.edge, c.fwd, c.off)
                dturn = np.arctan2(np.sin(h1 - h0), np.cos(h1 - h0))
                r = np.arctan2(np.sin(dturn - step_turns[k]), np.cos(dturn - step_turns[k]))
                cost = h.cost + (r / st) ** 2
                if c.edge != h.edge:
                    cost += cfg.w_junction      # crossing into a new edge is not free
                c.cost = cost
                nxt.append(c)
        if not nxt:
            break
        nxt.sort(key=lambda h: h.cost)
        hyps = nxt[: cfg.beam]

    best = hyps[0]
    path, edges, junc = [], [], 0
    h = best
    seen_nodes = []
    while h is not None:
        path.append(_xy_at(graph, h.edge, h.fwd, h.off))
        edges.append(h.edge)
        if h.node is not None:
            seen_nodes.append(h.node)
        h = h.prev
    path.reverse(); edges.reverse()
    junc = sum(1 for n in set(seen_nodes) if graph.junction_degree(n) > 2)
    return np.array(path), edges, junc
