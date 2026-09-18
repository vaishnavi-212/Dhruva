"""ISRO requirement 7 — edge-deployable engine.

The rest of the codebase assumes a phone: Sensor Logger CSV exports, pandas,
whole arrays in memory, a 10 Hz Android IMU. An edge box has none of that. It
streams samples from an external IMU at 200 Hz, has no filesystem convention,
and cannot hold a ride in RAM.

This is the adapter. One class, sample-at-a-time, fixed memory, no pandas, no
file format, no lookahead. The same estimator the phone uses runs behind it.

Three things the adapter must reconcile:

  * RATE.       The speed model is trained at 10 Hz on a 300-sample (30 s)
                window. An external IMU at 200 Hz is 20x that. The engine keeps
                a ring buffer at the native rate and decimates into the model's
                rate, so the model sees exactly what it was trained on
                regardless of what the hardware produces.
  * CAUSALITY.  Everything here is forward-only. No centred filters, no future
                samples -- the offline pipeline's `rolling(center=True)` would
                silently look ahead and cannot be used online.
  * MEMORY.     Bounded. The ring buffers are the only state that scales, and
                they are fixed at construction.

    python -m dhruva.edge --imu run.csv --hz 200
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

GRAVITY = 9.80665


@dataclass
class EdgeConfig:
    imu_hz: float = 200.0          # native rate of the attached IMU
    model_hz: float = 10.0         # rate the speed model was trained at
    model_window: int = 300        # samples at model_hz
    tilt_window_s: float = 20.0    # gravity averaging for the level frame
    infer_every_s: float = 1.0     # how often to re-run the speed model


class _Ring:
    """Fixed-size circular buffer. Bounded memory, O(1) append."""

    def __init__(self, n, width):
        self.buf = np.zeros((int(n), int(width)), dtype=np.float32)
        self.n = int(n)
        self.i = 0
        self.filled = 0

    def push(self, v):
        self.buf[self.i] = v
        self.i = (self.i + 1) % self.n
        self.filled = min(self.filled + 1, self.n)

    def latest(self, k):
        """Most recent k samples, oldest first. Returns fewer if not yet filled."""
        k = min(int(k), self.filled)
        if k == 0:
            return np.zeros((0, self.buf.shape[1]), np.float32)
        idx = (self.i - k + np.arange(k)) % self.n
        return self.buf[idx]


class EdgeEngine:
    """Streaming dead-reckoning engine for an external IMU.

    Feed it one sample at a time with `push_imu`; give it a fix with `push_gnss`
    whenever one is available. `state` is valid after the first fix.
    """

    def __init__(self, cfg: EdgeConfig = EdgeConfig(), speed_fn=None,
                 road=None, uncertainty_k: float = 0.19):
        self.cfg = cfg
        self.speed_fn = speed_fn          # callable(features NxC) -> metres per window
        self.road = road                  # optional RoadPolyline
        self.k = float(uncertainty_k)

        self.decim = max(int(round(cfg.imu_hz / cfg.model_hz)), 1)
        self._n = 0
        self.acc = _Ring(cfg.model_window + 8, 3)      # at MODEL rate
        self.gyro = _Ring(cfg.model_window + 8, 3)
        self._grav = np.array([0.0, 0.0, GRAVITY], np.float64)
        self._grav_a = float(np.exp(-1.0 / (cfg.tilt_window_s * cfg.imu_hz)))

        self.t = 0.0
        self.speed = 0.0
        self.arc = None                   # along-road position, once initialised
        self.dist_since_fix = 0.0
        self.mode = "NO_FIX"
        self._last_infer_t = -1e9
        self._last_t = None

    # ---------- input ----------
    def push_gnss(self, pos_xy, speed_mps=None):
        """A GNSS fix. Resets the distance-since-fix that drives confidence."""
        if self.road is not None:
            self.arc = self.road.project(np.asarray(pos_xy, float))
        self.dist_since_fix = 0.0
        if speed_mps is not None:
            self.speed = float(speed_mps)
        self.mode = "GNSS_AIDED"

    def push_imu(self, t, accel_xyz, gyro_xyz):
        """One IMU sample at the native rate. Returns the current state."""
        a = np.asarray(accel_xyz, float)
        g = np.asarray(gyro_xyz, float)
        dt = 0.0 if self._last_t is None else max(t - self._last_t, 0.0)
        self._last_t = t
        self.t = t

        # causal gravity estimate -- exponential, never centred
        self._grav = self._grav_a * self._grav + (1.0 - self._grav_a) * a

        if self._n % self.decim == 0:
            lin = a - self._grav
            self.acc.push(lin.astype(np.float32))
            self.gyro.push(g.astype(np.float32))
        self._n += 1

        if self.mode != "NO_FIX":
            self.dist_since_fix += self.speed * dt
            if self.road is not None and self.arc is not None:
                self.arc += self.speed * dt

        if (self.speed_fn is not None
                and self.acc.filled >= self.cfg.model_window
                and t - self._last_infer_t >= self.cfg.infer_every_s):
            self._infer()
            self._last_infer_t = t
        return self.state

    # ---------- estimator ----------
    def _level_frame(self):
        """Level-frame rotation, matching `dhruva.align.estimate_tilt` EXACTLY.

        This must reproduce the offline basis, not merely *a* basis that puts
        gravity down. The speed model is yaw-DEPENDENT (RESOURCES.md 7k), so a
        rotation with an arbitrary yaw feeds it channels it never saw in
        training. Using a Rodrigues rotation here cost 12% drift against the
        offline pipeline's 1.7% before the convention was matched.

        Offline builds rows [east, north, down] from a fixed reference vector;
        the only difference here is that `down` comes from a causal exponential
        estimate instead of a centred moving average.
        """
        down = self._grav / max(np.linalg.norm(self._grav), 1e-9)
        ref = np.array([1.0, 0.0, 0.0])
        if abs(float(ref @ down)) > 0.9:
            ref = np.array([0.0, 1.0, 0.0])
        east = np.cross(ref, down)
        east /= max(np.linalg.norm(east), 1e-9)
        north = np.cross(down, east)
        return np.stack([east, north, down], axis=0)

    def _infer(self):
        W = self.cfg.model_window
        A = self.acc.latest(W); G = self.gyro.latest(W)
        if len(A) < W:
            return
        R = self._level_frame()
        F = np.concatenate([G @ R.T, A @ R.T], axis=1).astype(np.float32)  # v @ R.T == R @ v per row
        metres = float(self.speed_fn(F))
        self.speed = max(metres / (W / self.cfg.model_hz), 0.0)

    # ---------- output ----------
    @property
    def state(self):
        from dhruva.uncertainty import radius_for
        pos = None
        if self.road is not None and self.arc is not None:
            pos = self.road.at(np.array([self.arc]))[0][0]
        return {
            "t": self.t,
            "pos": pos,
            "speed": self.speed,
            "mode": self.mode,
            "dist_since_fix": self.dist_since_fix,
            "radius_90": float(radius_for(self.dist_since_fix, 0.90, self.k)),
        }
