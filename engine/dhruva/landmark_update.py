"""Uncertainty-gated along-track landmark correction.

STATUS (31-Aug audit, RESOURCES.md 7v): this module makes the landmark layer SAFE
but not yet BENEFICIAL. Ship with landmarks OFF until detector precision improves.
`correct()` is the supported entry point; `correct_scaled()` is a recorded failure.

The as-built layer applied a HARD along-track reset on every match inside an 80 m
window. Measured (RESOURCES.md 7u): the detector fires 18-30 times against 14
surveyed landmarks, so 30-115% of detections are false positives, and one bad
reset is catastrophic -- median drift went from 4.5% (landmarks OFF) to 30.3%.
Tightening the window by hand recovered it, but those thresholds were chosen on
the same runs being scored, so the gain was not real.

This replaces the threshold with a statistical test the vehicle can actually
evaluate online.

Idea: along-track position error grows in proportion to distance travelled since
the last correction, because it is dominated by the speed model's scale error.
So carry a variance, propagate it with distance, and accept a landmark only if
the discrepancy is consistent with that variance (a chi-square gate). Then apply
a Kalman-weighted correction rather than a hard snap, so a marginal match nudges
the estimate instead of teleporting it.

`growth` is not a free knob: it is the speed model's measured relative distance
error (RESOURCES 7u LORO -> ~7%). `survey_sigma` is the surveyed landmark
position accuracy (campus survey: 3-4 m).
"""
from __future__ import annotations
import numpy as np

GAP = -1


class AlongTrackFilter:
    """1-D position-error filter along the road."""

    def __init__(self, growth: float = 0.07, survey_sigma: float = 4.0,
                 sigma0: float = 5.0, gate_sigmas: float = 3.0):
        self.growth = float(growth)
        self.r2 = float(survey_sigma) ** 2
        self.var = float(sigma0) ** 2
        self.gate2 = float(gate_sigmas) ** 2
        self.last_s = None

    @property
    def sigma(self) -> float:
        return float(np.sqrt(self.var))

    def propagate(self, s: float) -> None:
        """Grow uncertainty for the distance travelled since the last call."""
        if self.last_s is not None:
            ds = abs(s - self.last_s)
            self.var += (self.growth * ds) ** 2
        self.last_s = s

    def accepts(self, innovation: float) -> bool:
        """Chi-square gate: is this discrepancy consistent with our uncertainty?"""
        return innovation ** 2 <= self.gate2 * (self.var + self.r2)

    def update(self, innovation: float) -> float:
        """Kalman-weighted correction. Returns the shift to apply to arc length."""
        K = self.var / (self.var + self.r2)
        self.var *= (1.0 - K)
        return K * innovation


def correct(arc: np.ndarray, t: np.ndarray, events, map_s: np.ndarray,
            growth: float = 0.07, survey_sigma: float = 4.0,
            sigma0: float = 5.0, gate_sigmas: float = 3.0):
    """Apply gated landmark corrections to an along-track arc-length series.

    Monotonic: a landmark once passed is never revisited. Returns (arc, stats).
    """
    arc = np.asarray(arc, float).copy()
    map_s = np.asarray(map_s, float)
    f = AlongTrackFilter(growth, survey_sigma, sigma0, gate_sigmas)
    nxt = 0
    n_acc = n_rej = 0
    for e in events:
        i = int(np.searchsorted(t, e.t))
        if i >= len(arc):
            break
        f.propagate(arc[i])
        # nearest not-yet-consumed landmark ahead of or at our estimate
        while nxt < len(map_s) and map_s[nxt] < arc[i] - gate_sigmas * f.sigma - 3 * survey_sigma:
            nxt += 1
        if nxt >= len(map_s):
            break
        j = nxt + int(np.argmin(np.abs(map_s[nxt:nxt + 3] - arc[i])))
        innov = map_s[j] - arc[i]
        if f.accepts(innov):
            shift = f.update(innov)
            arc[i:] += shift
            f.last_s = arc[i]
            nxt = j + 1
            n_acc += 1
        else:
            n_rej += 1
    return arc, {"accepted": n_acc, "rejected": n_rej}


class ScaleAidedFilter:
    """MEASURED NOT TO WORK -- kept as a recorded negative. Do not enable.

    Result (RESOURCES.md 7v): median drift 13.8% vs 4.5% with landmarks off, and
    association DROPPED to 26% from 42%. Early wrong matches corrupt the scale
    estimate, which then drags every later prediction. The bootstrap assumption --
    that early matches are reliable because uncertainty is small -- does not hold
    when the detector's false-positive rate is this high.

    Joint along-track position AND speed-scale estimation.

    Measured (RESOURCES.md 7u/7v): position-only landmark aiding associates only
    42% of detections correctly. The reason is structural -- surveyed landmarks on
    the campus lap sit ~60 m apart, and the speed model's ~7% distance error
    reaches ~30 m by mid-lap, half the spacing, so the nearest-landmark choice
    becomes a coin flip.

    The fix uses the fact that the error is dominated by a SCALE error, not a
    random walk: the model consistently over- or under-reads distance for a given
    rider and phone. Early landmarks are matched while uncertainty is still small
    (the first is 20 m in), and each confident match refines a scale estimate that
    is applied to all FUTURE distance -- which keeps uncertainty small enough for
    the later matches to stay reliable.

    arc(t) = s0 + k * raw_distance(t) + b
    """

    def __init__(self, growth: float = 0.07, survey_sigma: float = 4.0,
                 sigma0: float = 5.0, gate_sigmas: float = 3.0,
                 min_pairs_for_scale: int = 2, max_scale_dev: float = 0.35):
        self.growth = float(growth)
        self.r2 = float(survey_sigma) ** 2
        self.sigma0 = float(sigma0)
        self.gate2 = float(gate_sigmas) ** 2
        self.min_pairs = int(min_pairs_for_scale)
        self.max_dev = float(max_scale_dev)
        self.k = 1.0
        self.b = 0.0
        self.pairs: list[tuple[float, float]] = []   # (raw_distance, map_arc)

    def sigma_at(self, raw_d: float) -> float:
        """Uncertainty grows with distance since the last accepted landmark."""
        d0 = self.pairs[-1][0] if self.pairs else 0.0
        return float(np.hypot(self.sigma0, self.growth * max(raw_d - d0, 0.0)))

    def predict(self, raw_d: float, s0: float) -> float:
        return s0 + self.k * raw_d + self.b

    def accepts(self, innovation: float, raw_d: float) -> bool:
        s = self.sigma_at(raw_d)
        return innovation ** 2 <= self.gate2 * (s * s + self.r2)

    def add(self, raw_d: float, map_arc: float, s0: float) -> None:
        """Accept a match and refit scale + offset over all accepted pairs."""
        self.pairs.append((raw_d, map_arc))
        if len(self.pairs) < self.min_pairs:
            # not enough to identify a scale -- shift only
            self.b = map_arc - (s0 + self.k * raw_d)
            return
        D = np.array([p[0] for p in self.pairs])
        Ms = np.array([p[1] for p in self.pairs]) - s0
        A = np.column_stack([D, np.ones_like(D)])
        k, b = np.linalg.lstsq(A, Ms, rcond=None)[0]
        # a scale far from 1 means the fit is being driven by bad associations
        if abs(k - 1.0) <= self.max_dev:
            self.k, self.b = float(k), float(b)
        else:
            self.b = map_arc - (s0 + self.k * raw_d)


def correct_scaled(raw_dist: np.ndarray, t: np.ndarray, events, map_s: np.ndarray,
                   s0: float, **kw):
    """Scale-aided monotonic landmark correction. Returns (arc, stats)."""
    raw_dist = np.asarray(raw_dist, float)
    map_s = np.asarray(map_s, float)
    f = ScaleAidedFilter(**kw)
    nxt = 0
    n_acc = n_rej = 0
    marks = []
    for e in events:
        i = int(np.searchsorted(t, e.t))
        if i >= len(raw_dist):
            break
        est = f.predict(raw_dist[i], s0)
        sig = f.sigma_at(raw_dist[i])
        while nxt < len(map_s) and map_s[nxt] < est - 3.0 * sig - 3 * np.sqrt(f.r2):
            nxt += 1
        if nxt >= len(map_s):
            break
        j = nxt + int(np.argmin(np.abs(map_s[nxt:nxt + 3] - est)))
        if f.accepts(map_s[j] - est, raw_dist[i]):
            f.add(raw_dist[i], map_s[j], s0)
            nxt = j + 1
            n_acc += 1
            marks.append((i, j))
        else:
            n_rej += 1
    arc = s0 + f.k * raw_dist + f.b
    return arc, {"accepted": n_acc, "rejected": n_rej, "k": f.k, "marks": marks}


def merge_double_fires(events, t, raw_dist, min_sep_m: float = 25.0):
    """Collapse repeat detections of the SAME physical landmark.

    Measured (RESOURCES.md 7w): across the six 30-Aug laps the detector produced
    145 detections at only **12 distinct locations**, 11 of which repeat in all
    six runs. So the detector's precision was never the problem -- it fires
    roughly twice per landmark per pass, and those repeats were being counted as
    false positives and matched to separate map entries.

    Merging must be done in TRAVELLED DISTANCE, not time: the detector's own
    `min_gap_s` is a time gap, so it merges differently at 3 m/s than at 8 m/s.

    `min_sep_m` is set from the map, not from the drift result: the distinct
    repeatable locations are ~50 m apart, so half that separates real landmarks
    from repeats of one.
    """
    if not events:
        return events
    import numpy as _np
    pos = lambda e: float(_np.interp(e.t, t, raw_dist))
    out = [events[0]]
    for e in events[1:]:
        if pos(e) - pos(out[-1]) < min_sep_m:
            if e.snr > out[-1].snr:
                out[-1] = e
        else:
            out.append(e)
    return out
