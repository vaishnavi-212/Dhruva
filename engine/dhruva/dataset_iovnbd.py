"""Windowed IO-VNBD dataset: level-frame IMU -> vehicle speed.

Speed is a SCALAR, so this target is invariant to how the phone is yawed in the
vehicle. That matters because IO-VNBD's ORIENTATION channel is unusable
(RESOURCES.md §5a) and yaw cannot be recovered from its 0.11 Hz GPS.

Heading is handled separately, by integrating level-frame gyro Z.
"""
from __future__ import annotations
import numpy as np
import torch
from torch.utils.data import Dataset

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from harness import dataset as ds
from dhruva import align


def build_features(trace, invariant: bool = True):
    """Level-frame IMU -> features + cumulative distance target.

    invariant=True builds YAW-INVARIANT channels. This matters more than it
    sounds: tilt correction fixes pitch and roll, but yaw cannot be recovered
    from IO-VNBD's 0.11 Hz GPS, so the level-frame X and Y axes point in a
    DIFFERENT arbitrary direction in every file. A network fed (ax, ay) sees the
    same physical motion encoded differently per recording and can only memorise
    it -- which is exactly the failure we measured (best epoch always early,
    then validation diverges while training loss keeps falling).

    Magnitudes and the vertical/yaw axes carry no yaw dependence at all, so
    every file presents motion in the same frame.
    """
    R = align.estimate_tilt(trace.acc_raw if trace.acc_raw is not None else trace.acc,
                            trace.hz)
    acc_l = align.apply(R, trace.acc)
    gyr_l = align.apply(R, trace.gyro)

    if invariant:
        a_h = np.linalg.norm(acc_l[:, :2], axis=1)      # horizontal magnitude
        a_v = acc_l[:, 2]                               # vertical (yaw-free)
        w_h = np.linalg.norm(gyr_l[:, :2], axis=1)      # roll/pitch rate magnitude
        w_z = gyr_l[:, 2]                               # yaw rate (yaw-free)
        a_m = np.linalg.norm(acc_l, axis=1)             # total accel magnitude
        jerk = np.r_[0.0, np.abs(np.diff(a_m)) * trace.hz]   # roughness
        feats = np.column_stack([a_h, a_v, w_h, w_z, a_m, jerk]).astype(np.float32)
    else:
        feats = np.concatenate([gyr_l, acc_l], axis=1).astype(np.float32)

    # TARGET: cumulative distance from the GPS TRACK, not the GPS SPEED column.
    # The SPEED column disagrees with position-derived speed by a median of
    # 6.57 m/s -- a floor no model can beat. GPS position is good to ~3 m, so
    # distance over a long window carries ~1% label error instead of ~50%.
    cum = np.concatenate([[0.0], np.cumsum(
        np.linalg.norm(np.diff(trace.xy, axis=0), axis=1))]).astype(np.float32)
    return feats, cum


class WindowSet(Dataset):
    """Windows of level-frame IMU -> DISTANCE TRAVELLED over the window (metres).

    Longer windows give a better label signal-to-noise ratio, because GPS
    position error (~3 m) is fixed while distance grows with window length.
    """

    def __init__(self, files, window=300, step=8, hz=10.0, stats=None,
                 verbose=True, invariant=True):
        self.window, self.X, self.y = window, [], []
        skipped = []
        for f in files:
            try:
                tr = ds.load(f, hz=hz, verbose=False)
                feats, spd = build_features(tr, invariant=invariant)
            except Exception as e:
                skipped.append((os.path.basename(f), str(e).split(" -- ")[-1]))
                continue
            n = len(feats)
            for i in range(0, n - window, step):
                d = spd[i + window - 1] - spd[i]          # metres over the window
                if not np.isfinite(d) or d < 0:
                    continue
                self.X.append(feats[i:i + window])
                self.y.append(np.float32(d))
            if verbose:
                print(f"  {os.path.basename(f):16s} {n:7d} samples -> {len(self.X):7d} windows")
        if skipped:
            print(f"  skipped {len(skipped)} unusable file(s): "
                  f"{', '.join(n for n, _ in skipped[:8])}"
                  f"{' ...' if len(skipped) > 8 else ''}")
        if not self.X:
            raise RuntimeError("no usable windows built from any file")
        self.X = np.stack(self.X).transpose(0, 2, 1)       # (N, C, T)
        self.y = np.array(self.y, dtype=np.float32)[:, None]

        if stats is None:
            self.mean = self.X.mean(axis=(0, 2), keepdims=True)
            self.std = self.X.std(axis=(0, 2), keepdims=True) + 1e-6
        else:
            self.mean, self.std = stats
        self.X = (self.X - self.mean) / self.std

    @property
    def stats(self):
        return self.mean, self.std

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return torch.from_numpy(self.X[i]), torch.from_numpy(self.y[i])
