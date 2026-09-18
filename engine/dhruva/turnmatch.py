"""Curvature-sequence matching — contribution ⑦ / requirement 3's geometric half.

Landmark aiding needs surveyed point features. On a ghat road there are none, but
the road's own SHAPE is a signature: a sequence of turns with particular angles
and spacings is as identifying as a sequence of speed breakers.

The map here is the LEARNED map, not OSM. Measured (RESOURCES.md 7z): turn
fidelity is 0.615 from OSM but 0.810 from a map learned over 8 traversals, and it
improves with every pass — 0.424 at 5 passes. OSM is fixed; the learned map is not.

Turn events are found the same way on both sides so they are comparable:
a sustained same-sign heading rate integrating to at least `min_angle`.
"""
from __future__ import annotations
import numpy as np


def turns_from_heading(s: np.ndarray, dpsi_ds: np.ndarray,
                       min_angle: float = 25.0, max_radius: float = 60.0):
    """Sustained same-sign curvature runs -> (arc_at_peak, signed_angle_deg)."""
    thr = 1.0 / max_radius
    sign = np.where(np.abs(dpsi_ds) < thr, 0, np.sign(dpsi_ds)).astype(int)
    out = []
    i = 0
    while i < len(sign):
        if sign[i] == 0:
            i += 1
            continue
        j = i
        while j < len(sign) and sign[j] == sign[i]:
            j += 1
        ang = np.degrees(np.trapezoid(dpsi_ds[i:j], s[i:j]))
        if abs(ang) >= min_angle:
            k = i + int(np.argmax(np.abs(dpsi_ds[i:j])))
            out.append((float(s[k]), float(ang)))
        i = j
    return out


def map_turns(road, min_angle: float = 25.0, max_radius: float = 60.0):
    """Turns of a RoadPolyline, by arc length."""
    h = np.unwrap(road.heading)
    ds = np.gradient(road.s)
    return turns_from_heading(road.s, np.gradient(h) / np.maximum(ds, 1e-9),
                              min_angle, max_radius)


def vehicle_turns(t, wz, arc, min_angle: float = 25.0, min_rate: float = 0.12):
    """Turns the vehicle actually made, from level-frame yaw rate.

    Indexed by dead-reckoned arc length so they are comparable with map turns.
    """
    sign = np.where(np.abs(wz) < min_rate, 0, np.sign(wz)).astype(int)
    out = []
    i = 0
    while i < len(sign):
        if sign[i] == 0:
            i += 1
            continue
        j = i
        while j < len(sign) and sign[j] == sign[i]:
            j += 1
        ang = np.degrees(np.trapezoid(wz[i:j], t[i:j]))
        if abs(ang) >= min_angle:
            k = i + int(np.argmax(np.abs(wz[i:j])))
            out.append((int(k), float(arc[k]), float(ang)))
        i = j
    return out


def correct(arc, t, wz, road, growth: float = 0.07, map_sigma: float = 8.0,
            sigma0: float = 5.0, gate_sigmas: float = 3.0,
            angle_tol_deg: float = 45.0):
    """Gated monotonic along-track correction from turn geometry.

    Same statistical gate as the landmark path: uncertainty grows with distance
    travelled, a turn is accepted only if the discrepancy is consistent with it,
    and the correction is Kalman-weighted. Additionally the turn ANGLE must agree
    -- matching a 30 deg bend to a 120 deg hairpin is rejected outright.
    """
    arc = np.asarray(arc, float).copy()
    mt = map_turns(road)
    if not mt:
        return arc, {"accepted": 0, "rejected": 0}
    ms = np.array([m[0] for m in mt])
    ma = np.array([m[1] for m in mt])
    vt = vehicle_turns(t, wz, arc)
    var = sigma0 ** 2
    r2 = map_sigma ** 2
    last_arc = arc[0]
    nxt = 0
    acc = rej = 0
    for i, a_est, ang in vt:
        var += (growth * max(arc[i] - last_arc, 0.0)) ** 2
        last_arc = arc[i]
        sig = np.sqrt(var)
        while nxt < len(ms) and ms[nxt] < arc[i] - gate_sigmas * sig - 3 * map_sigma:
            nxt += 1
        cand = [j for j in range(nxt, min(nxt + 4, len(ms)))
                if abs(ma[j] - ang) <= angle_tol_deg and np.sign(ma[j]) == np.sign(ang)]
        if not cand:
            rej += 1
            continue
        j = min(cand, key=lambda q: abs(ms[q] - arc[i]))
        innov = ms[j] - arc[i]
        if innov ** 2 <= (gate_sigmas ** 2) * (var + r2):
            K = var / (var + r2)
            arc[i:] += K * innov
            var *= (1.0 - K)
            last_arc = arc[i]
            nxt = j + 1
            acc += 1
        else:
            rej += 1
    return arc, {"accepted": acc, "rejected": rej}


def merge_turn_pieces(vt, min_sep_m: float = 15.0):
    """Merge same-sign turns detected close together — one turn seen in pieces.

    Exactly the failure the landmark detector had (RESOURCES.md 7w): a single
    physical feature fires more than once, and each piece is then matched to a
    different map entry. On the ghat route the gyro produced 35 turns against 28
    real ones; merging same-sign detections within 15 m of travelled distance
    brings that to 30 and roughly halves the localisation error.

    The angles are SUMMED, not averaged — the pieces are parts of one turn, so
    their contributions add.
    """
    if not vt:
        return vt
    out = [list(vt[0])]
    for i, a, ang in vt[1:]:
        if a - out[-1][1] < min_sep_m and np.sign(ang) == np.sign(out[-1][2]):
            out[-1][2] += ang
        else:
            out.append([i, a, ang])
    return [tuple(x) for x in out]


def localise(vt, map_s, map_ang, n_turns: int = 6,
             ang_tol: float = 45.0, gap_tol: float = 40.0):
    """Find where on the road a sequence of turns puts us. Needs no prior.

    Measured on the 2305 m ghat route (RESOURCES.md 8f): accuracy depends
    strongly on how many turns are in the window --

        3 turns  44% within 50 m, 206 m median error
        5 turns  50% within 50 m,  65 m
        6 turns  64% within 50 m,   8 m
        8 turns 100% within 50 m,   2 m   (only 4 windows -- weak evidence)

    Below about six turns the road shape is simply not distinctive enough.
    Returns (map_index, cost) or (None, inf).
    """
    map_s = np.asarray(map_s, float); map_ang = np.asarray(map_ang, float)
    N = int(n_turns)
    if len(vt) < N or len(map_s) < N:
        return None, np.inf
    seq = vt[:N]
    angs = np.array([s[2] for s in seq])
    gaps = np.diff([s[1] for s in seq])
    best, arg = np.inf, None
    for j in range(len(map_s) - N + 1):
        mang = map_ang[j:j + N]
        if np.any(np.sign(mang) != np.sign(angs)):
            continue
        if np.any(np.abs(mang - angs) > ang_tol):
            continue
        c = (np.sum((mang - angs) ** 2) / ang_tol ** 2
             + np.sum((np.diff(map_s[j:j + N]) - gaps) ** 2) / gap_tol ** 2)
        if c < best:
            best, arg = c, j
    return arg, best
