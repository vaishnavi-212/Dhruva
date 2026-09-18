"""Contribution ③ — typed landmark detection from phone IMU.

Design follows what the ride data actually showed:

  * Work on |a| (acceleration magnitude), not a single axis. It is
    orientation-independent, so it survives the phone being in a pocket.
  * Speed breakers are TRANSIENTS. Subtract a rolling median to kill the
    slow-varying ride, leaving only events.
  * Threshold adaptively (MAD), never on a fixed number -- engine noise
    differs per vehicle, per mount, per lap.

Every event carries features so it can later be TYPED (bump / hump / table /
rumble), which is what makes sequence matching against the map unique.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd


@dataclass
class Event:
    t: float            # seconds, peak of the event
    amp: float          # peak envelope amplitude, m/s^2
    snr: float          # amp / noise floor
    duration_s: float   # width at half maximum
    energy: float       # integral of the envelope over the event
    dom_freq_hz: float  # dominant frequency inside the event
    kind: str = ""      # bump | hump | table | rumble  (filled once labelled)

    def as_dict(self):
        return asdict(self)


def _roll(x, n, fn="median"):
    s = pd.Series(x).rolling(n, center=True, min_periods=1)
    return (s.median() if fn == "median" else s.max()).to_numpy()


def preprocess(acc_xyz: np.ndarray, hz: float,
               detrend_s: float = 0.7, env_s: float = 0.15):
    """(N,3) accelerometer -> (magnitude, high-passed, envelope)."""
    mag = np.linalg.norm(acc_xyz, axis=1)
    mag = mag - np.median(mag)                       # remove gravity offset
    hp = mag - _roll(mag, max(int(detrend_s * hz) | 1, 3), "median")
    env = _roll(np.abs(hp), max(int(env_s * hz) | 1, 3), "max")
    return mag, hp, env


def noise_floor(hp: np.ndarray) -> float:
    """Robust sigma. MAD is immune to the very spikes we are hunting."""
    return float(np.median(np.abs(hp - np.median(hp))) * 1.4826)


def detect(acc_xyz: np.ndarray, t: np.ndarray, hz: float,
           k: float = 5.0, min_gap_s: float = 1.5,
           min_duration_s: float = 0.05) -> list[Event]:
    """Find landmark events. `k` is the threshold in robust sigmas."""
    mag, hp, env = preprocess(acc_xyz, hz)
    sigma = noise_floor(hp)
    thresh = k * sigma

    above = env > thresh
    if not above.any():
        return []

    # contiguous runs above threshold
    edges = np.diff(above.astype(int))
    starts = np.where(edges == 1)[0] + 1
    ends = np.where(edges == -1)[0] + 1
    if above[0]:
        starts = np.r_[0, starts]
    if above[-1]:
        ends = np.r_[ends, len(above)]

    events: list[Event] = []
    for s, e in zip(starts, ends):
        if (e - s) / hz < min_duration_s:
            continue
        seg = env[s:e]
        pk = s + int(np.argmax(seg))
        amp = float(env[pk])

        # width at half maximum, expanded outside the threshold crossing
        half = amp / 2.0
        l = pk
        while l > 0 and env[l] > half:
            l -= 1
        r = pk
        while r < len(env) - 1 and env[r] > half:
            r += 1
        dur = float((r - l) / hz)

        w = hp[max(l - int(0.1 * hz), 0): min(r + int(0.1 * hz), len(hp))]
        dom = _dominant_freq(w, hz)

        events.append(Event(t=float(t[pk]), amp=amp, snr=amp / max(sigma, 1e-9),
                            duration_s=dur, energy=float(np.sum(seg) / hz),
                            dom_freq_hz=dom))

    # merge events that are too close to be separate road features
    events.sort(key=lambda ev: ev.t)
    merged: list[Event] = []
    for ev in events:
        if merged and ev.t - merged[-1].t < min_gap_s:
            if ev.amp > merged[-1].amp:
                merged[-1] = ev
        else:
            merged.append(ev)
    return merged


def _dominant_freq(w: np.ndarray, hz: float) -> float:
    if len(w) < 8:
        return float("nan")
    w = w - w.mean()
    sp = np.abs(np.fft.rfft(w * np.hanning(len(w))))
    fr = np.fft.rfftfreq(len(w), 1.0 / hz)
    band = (fr > 1.0) & (fr < hz / 2)
    if not band.any():
        return float("nan")
    return float(fr[band][np.argmax(sp[band])])


def to_frame(events: list[Event]) -> pd.DataFrame:
    return pd.DataFrame([e.as_dict() for e in events]) if events else pd.DataFrame()


def load_ride(folder: str):
    """Sensor Logger export -> (t, acc_xyz, hz)."""
    import os
    a = pd.read_csv(os.path.join(folder, "Accelerometer.csv"))
    t = a["seconds_elapsed"].to_numpy(float)
    acc = a[["x", "y", "z"]].to_numpy(float)
    hz = 1.0 / float(np.median(np.diff(t)))
    return t, acc, hz
