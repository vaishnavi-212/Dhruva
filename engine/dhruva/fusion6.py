"""Six-state fusion — adds the speed-scale state the 5-state filter was missing.

Contribution ⑤. Measured (RESOURCES.md 8e): the 5-state filter is not merely
mis-tuned, it is structurally unable to describe its own error. Its dominant
error is a SYSTEMATIC per-run speed-scale error -- the learned speed model
over- or under-reads distance by a consistent percentage for a given rider,
phone and mount (measured +0.8% to -25.7% across our 11 runs). A filter whose
process model assumes white noise cannot represent a persistent bias: its
covariance shrinks while the true error grows linearly with distance. That is
exactly the overconfidence we measured -- ANEES in the thousands.

Adding a scale state lets the filter (a) estimate the bias from GNSS while it is
available and carry it into the blackout, and (b) grow position covariance in
proportion to distance travelled, which is how the error actually behaves.

State: [east, north, speed, heading, gyro_bias, speed_scale]
The model speed enters as   z = v / s   so the filter can attribute a persistent
mismatch to the scale rather than fighting it every epoch.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

I6 = np.eye(6)


@dataclass
class Config6:
    q_accel: float = 1.2
    q_yawrate: float = 0.35
    q_bias: float = 1e-5
    q_scale: float = 4e-3        # scale random walk, per sqrt-second
    r_gnss_pos: float = 4.0
    r_gnss_speed: float = 1.0
    r_model_speed: float = 3.5
    r_landmark: float = 4.0
    # a road constraint is ONE geometric fact, not a fresh measurement every
    # epoch. Applying it at full rate collapses the covariance (8e).
    road_update_every: int = 20


@dataclass
class State6:
    x: np.ndarray = field(default_factory=lambda: np.array([0., 0., 0., 0., 0., 1.]))
    P: np.ndarray = field(default_factory=lambda: np.diag([25., 25., 4., .5, 1e-4, .04]))
    mode: str = "GNSS_AIDED"
    epochs_since_gnss: int = 0

    @property
    def pos(self): return self.x[:2]
    @property
    def speed(self): return float(self.x[2])
    @property
    def heading(self): return float(self.x[3])
    @property
    def scale(self): return float(self.x[5])
    @property
    def pos_sigma(self):
        return float(np.sqrt(max(self.P[0, 0] + self.P[1, 1], 0.0) / 2.0))


class Fusion6:
    def __init__(self, cfg: Config6 = Config6()):
        self.cfg = cfg
        self.s = State6()
        self._k = 0

    def initialise(self, pos, speed, heading):
        self.s = State6()
        self.s.x[:2] = np.asarray(pos, float)
        self.s.x[2] = float(speed)
        self.s.x[3] = float(heading)

    def predict(self, gyro_z, accel_fwd, dt):
        c = self.cfg
        x, y, v, psi, b, sc = self.s.x
        w = gyro_z - b
        self.s.x = np.array([x + v * np.cos(psi) * dt,
                             y + v * np.sin(psi) * dt,
                             max(v + accel_fwd * dt, 0.0),
                             psi + w * dt, b, sc])
        F = I6.copy()
        F[0, 2] = np.cos(psi) * dt; F[0, 3] = -v * np.sin(psi) * dt
        F[1, 2] = np.sin(psi) * dt; F[1, 3] = v * np.cos(psi) * dt
        F[3, 4] = -dt
        Q = np.zeros((6, 6))
        Q[2, 2] = (c.q_accel * dt) ** 2
        Q[3, 3] = (c.q_yawrate * dt) ** 2
        Q[4, 4] = (c.q_bias * dt) ** 2
        Q[5, 5] = (c.q_scale * dt) ** 2
        Q[0, 0] = Q[1, 1] = (0.5 * c.q_accel * dt * dt) ** 2
        self.s.P = F @ self.s.P @ F.T + Q

    def _update(self, H, r, R):
        S = H @ self.s.P @ H.T + R
        K = self.s.P @ H.T @ np.linalg.inv(S)
        self.s.x = self.s.x + K @ r
        A = I6 - K @ H
        self.s.P = A @ self.s.P @ A.T + K @ R @ K.T
        self.s.x[3] = np.arctan2(np.sin(self.s.x[3]), np.cos(self.s.x[3]))
        self.s.x[5] = float(np.clip(self.s.x[5], 0.4, 2.5))

    def update_gnss_position(self, pos, sigma=None):
        R = np.eye(2) * (sigma or self.cfg.r_gnss_pos) ** 2
        H = np.zeros((2, 6)); H[0, 0] = H[1, 1] = 1.0
        self._update(H, np.asarray(pos, float) - self.s.x[:2], R)
        self.s.mode = "GNSS_AIDED"; self.s.epochs_since_gnss = 0

    def update_model_speed(self, v_model, sigma=None):
        """z = v / scale, so a persistent mismatch is attributed to the scale."""
        v, sc = self.s.x[2], self.s.x[5]
        H = np.zeros((1, 6)); H[0, 2] = 1.0 / sc; H[0, 5] = -v / (sc * sc)
        r = np.array([v_model - v / sc])
        self._update(H, r, np.array([[(sigma or self.cfg.r_model_speed) ** 2]]))

    def update_road_heading(self, heading, sigma_rad=0.05):
        H = np.zeros((1, 6)); H[0, 3] = 1.0
        d = np.arctan2(np.sin(heading - self.s.x[3]), np.cos(heading - self.s.x[3]))
        self._update(H, np.array([d]), np.array([[sigma_rad ** 2]]))

    def update_road_position(self, road_pos, road_heading, sigma_cross=2.5):
        n = np.array([-np.sin(road_heading), np.cos(road_heading)])
        H = np.zeros((1, 6)); H[0, 0], H[0, 1] = n[0], n[1]
        resid = float(n @ (np.asarray(road_pos, float) - self.s.x[:2]))
        self._update(H, np.array([resid]), np.array([[sigma_cross ** 2]]))

    def update_landmark(self, pos, sigma=None):
        R = np.eye(2) * (sigma or self.cfg.r_landmark) ** 2
        H = np.zeros((2, 6)); H[0, 0] = H[1, 1] = 1.0
        self._update(H, np.asarray(pos, float) - self.s.x[:2], R)

    def step(self, dt, gyro_z, accel_fwd, gnss_pos=None, model_speed=None,
             road_heading=None, road_pos=None, landmark_pos=None):
        self.predict(gyro_z, accel_fwd, dt)
        if gnss_pos is not None:
            self.update_gnss_position(gnss_pos)
        else:
            self.s.epochs_since_gnss += 1
            self.s.mode = "DEAD_RECKONING"
            if landmark_pos is not None:
                self.update_landmark(landmark_pos)
        if model_speed is not None:
            self.update_model_speed(model_speed)
        # road geometry is one fact; apply it at a reduced rate
        if self._k % max(self.cfg.road_update_every, 1) == 0:
            if road_heading is not None:
                self.update_road_heading(road_heading)
            if road_pos is not None:
                self.update_road_position(road_pos, road_heading if road_heading
                                          is not None else self.s.heading)
        self._k += 1
        return self.s
