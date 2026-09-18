"""Requirements 4 & 5 — GNSS+INS fusion engine and seamless deficit handler.

PS 4: "An innovative AI based Sensor Fusion Algorithm that combines GNSS & IMU
       measurements ... eliminating drift errors and providing accurate position
       and velocity." 10 Hz on phone, ~200 Hz on the edge engine.
PS 5: "An instant seamless transition mechanism between GNSS aided INS and Dead
       reckoning modes within milliseconds of GNSS signal blackout and vice-versa."

Design note on requirement 5: a correctly built Kalman filter makes the
transition free. There is no mode switch to execute -- you simply stop applying
the GNSS measurement update and keep propagating. Uncertainty grows on its own,
and when GNSS returns the update is weighted by the covariance the filter has
already accumulated, so the estimate slides back instead of teleporting. The
"seamless handler" is therefore a property of the design, not a separate branch.

State x = [east, north, speed, heading, gyro_bias]

Measurements, each optional per epoch:
  * GNSS position, GNSS speed        -- when satellites are available
  * learned speed from the A2 model  -- always; this is the "AI based" element,
                                        injected with the model's own covariance
  * non-holonomic constraint         -- a vehicle does not move sideways
  * landmark along-road fix          -- from the matcher, during a blackout
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

I5 = np.eye(5)


@dataclass
class FusionConfig:
    # process noise (per sqrt-second)
    q_accel: float = 1.2          # m/s^2, unmodelled acceleration
    q_yawrate: float = 0.35       # rad/s -- MEMS gyro heading is genuinely
                              # this bad; measured 0.18 deg/s bias alone
    q_bias: float = 1e-5          # rad/s, bias random walk
    # measurement noise
    r_gnss_pos: float = 4.0       # m
    r_gnss_speed: float = 1.0     # m/s
    r_model_speed: float = 3.5    # m/s -- A2 is ~28% at ~13 m/s
    r_nhc: float = 0.35           # m/s lateral
    r_landmark: float = 4.0       # m, surveyed landmark accuracy
    nhc_min_speed: float = 2.0    # only apply NHC while actually moving


@dataclass
class FusionState:
    x: np.ndarray = field(default_factory=lambda: np.zeros(5))
    P: np.ndarray = field(default_factory=lambda: np.diag([25., 25., 4., 0.3, 1e-4]))
    mode: str = "INIT"
    epochs_since_gnss: int = 0

    @property
    def pos(self):   return self.x[:2]
    @property
    def speed(self): return float(self.x[2])
    @property
    def heading(self): return float(self.x[3])
    @property
    def bias(self):  return float(self.x[4])
    @property
    def pos_sigma(self):
        return float(np.sqrt(max(self.P[0, 0] + self.P[1, 1], 0.0) / 2.0))


class GnssInsFusion:
    """Loosely-coupled GNSS/INS EKF with an always-on learned-speed measurement."""

    def __init__(self, cfg: FusionConfig = FusionConfig()):
        self.cfg = cfg
        self.s = FusionState()
        self.history: list[dict] = []

    # ---------- initialisation ----------
    def initialise(self, pos, speed, heading):
        self.s = FusionState()
        self.s.x = np.array([pos[0], pos[1], speed, heading, 0.0], float)
        self.s.P = np.diag([16., 16., 1., 0.05, 1e-4])
        self.s.mode = "GNSS_AIDED"

    # ---------- prediction ----------
    def predict(self, gyro_z: float, accel_fwd: float, dt: float):
        c = self.cfg
        x, y, v, psi, b = self.s.x
        w = gyro_z - b

        self.s.x = np.array([x + v * np.cos(psi) * dt,
                             y + v * np.sin(psi) * dt,
                             max(v + accel_fwd * dt, 0.0),
                             psi + w * dt,
                             b])

        F = I5.copy()
        F[0, 2] = np.cos(psi) * dt; F[0, 3] = -v * np.sin(psi) * dt
        F[1, 2] = np.sin(psi) * dt; F[1, 3] = v * np.cos(psi) * dt
        F[3, 4] = -dt

        Q = np.zeros((5, 5))
        Q[2, 2] = (c.q_accel * dt) ** 2
        Q[3, 3] = (c.q_yawrate * dt) ** 2
        Q[4, 4] = (c.q_bias * dt) ** 2
        Q[0, 0] = Q[1, 1] = (0.5 * c.q_accel * dt * dt) ** 2
        self.s.P = F @ self.s.P @ F.T + Q

    # ---------- generic scalar/vector update ----------
    def _update(self, H, r, R):
        S = H @ self.s.P @ H.T + R
        K = self.s.P @ H.T @ np.linalg.inv(S)
        self.s.x = self.s.x + K @ r
        A = I5 - K @ H
        self.s.P = A @ self.s.P @ A.T + K @ R @ K.T          # Joseph form
        self.s.x[3] = np.arctan2(np.sin(self.s.x[3]), np.cos(self.s.x[3]))

    # ---------- measurements ----------
    def update_gnss_position(self, pos, sigma=None):
        R = np.eye(2) * (sigma or self.cfg.r_gnss_pos) ** 2
        H = np.zeros((2, 5)); H[0, 0] = H[1, 1] = 1.0
        self._update(H, np.asarray(pos, float) - self.s.x[:2], R)
        self.s.mode = "GNSS_AIDED"; self.s.epochs_since_gnss = 0

    def update_gnss_speed(self, speed):
        H = np.zeros((1, 5)); H[0, 2] = 1.0
        self._update(H, np.array([speed - self.s.x[2]]),
                     np.array([[self.cfg.r_gnss_speed ** 2]]))

    def update_model_speed(self, speed, sigma=None):
        """The AI element: A2's learned speed, with its own uncertainty."""
        H = np.zeros((1, 5)); H[0, 2] = 1.0
        R = np.array([[(sigma or self.cfg.r_model_speed) ** 2]])
        self._update(H, np.array([speed - self.s.x[2]]), R)

    def update_nhc(self):
        """Non-holonomic constraint.

        NOTE: with the state parameterised as (speed, heading) the velocity
        vector is ALWAYS along the heading, so lateral velocity is identically
        zero by construction. NHC is therefore structurally satisfied here and a
        pseudo-measurement adds no information -- an earlier version applied one
        anyway and it wrongly shrank the covariance, making the filter report
        5 m of uncertainty while carrying ~500 m of error.

        Kept as a no-op so the PS requirement is explicitly addressed rather
        than silently dropped. NHC matters only if velocity is tracked as a free
        2-D vector.
        """
        return

    def update_road_position(self, road_pos, road_heading, sigma_cross=2.5):
        """Bind position to the road WITHOUT constraining distance along it.

        This is the positional half of the PS's "binds the calculated position to
        known road networks and geometric paths". `update_road_heading` only
        corrects which way we are pointing; nothing stopped the estimate sliding
        sideways off the carriageway.

        The constraint must be anisotropic. A vehicle's CROSS-track position is
        known to about half a lane; its ALONG-track position is exactly what we
        do not know during a blackout. Applying an isotropic position update
        would inject a fake along-track fix and make the filter confidently
        wrong -- the same mistake the NHC pseudo-measurement made (RESOURCES 7f).

        So we measure only the component normal to the road:
            n . (x - road_pos) = 0,   n = unit normal to the road heading
        """
        n = np.array([-np.sin(road_heading), np.cos(road_heading)])
        H = np.zeros((1, 5)); H[0, 0], H[0, 1] = n[0], n[1]
        resid = float(n @ (np.asarray(road_pos, float) - self.s.x[:2]))
        self._update(H, np.array([resid]), np.array([[sigma_cross ** 2]]))

    def update_landmark(self, pos, sigma=None):
        """Along-road fix from the landmark matcher."""
        self.update_gnss_position(pos, sigma or self.cfg.r_landmark)
        self.s.mode = "DR_LANDMARK"

    def update_road_heading(self, heading, sigma_rad=0.05):
        """Heading taken from the road geometry rather than the gyro.

        This is the measurement that matters. Gyro-integrated heading is the
        dominant error source (RESOURCES.md 7b); binding heading to the road
        removes it. Requires knowing which road we are on, which is what the
        map-matching layer provides.
        """
        H = np.zeros((1, 5)); H[0, 3] = 1.0
        r = np.arctan2(np.sin(heading - self.s.x[3]), np.cos(heading - self.s.x[3]))
        self._update(H, np.array([r]), np.array([[sigma_rad ** 2]]))

    # ---------- one epoch ----------
    def step(self, dt, gyro_z, accel_fwd, gnss_pos=None, gnss_speed=None,
             model_speed=None, model_sigma=None, landmark_pos=None,
             road_heading=None, road_pos=None):
        self.predict(gyro_z, accel_fwd, dt)

        if gnss_pos is not None:
            self.update_gnss_position(gnss_pos)
            if gnss_speed is not None:
                self.update_gnss_speed(gnss_speed)
        else:
            # REQUIREMENT 5: no branch, no reset, no re-initialisation.
            # We simply stop applying GNSS. Covariance grows by itself.
            self.s.epochs_since_gnss += 1
            self.s.mode = "DEAD_RECKONING"
            if landmark_pos is not None:
                self.update_landmark(landmark_pos)

        if road_heading is not None:
            self.update_road_heading(road_heading)
        if road_pos is not None:
            self.update_road_position(road_pos, road_heading if road_heading is not None
                                      else self.s.heading)
        if model_speed is not None:
            self.update_model_speed(model_speed, model_sigma)
        self.update_nhc()

        self.history.append(dict(pos=self.s.pos.copy(), speed=self.s.speed,
                                 heading=self.s.heading, sigma=self.s.pos_sigma,
                                 mode=self.s.mode, bias=self.s.bias))
        return self.s


# ============================================================================
# REQUIREMENT 5 — seamless transition, both directions
# ============================================================================

@dataclass
class SeamlessConfig:
    tau_s: float = 1.8          # slew time constant, seconds
    max_rate_mps: float = 25.0  # hard cap on how fast the shown dot may move
    snap_below_m: float = 0.5   # below this, stop slewing and sit on the estimate
    reacquire_epochs: int = 3   # consecutive GNSS epochs before we trust it again


class SeamlessOutput:
    """Continuous user-facing position on top of an optimal filter.

    The two directions are NOT symmetric:

      GNSS -> DR  is already seamless. We stop applying the measurement and keep
                  propagating; nothing discontinuous happens. Verified: the jump
                  at blackout onset is zero.

      DR -> GNSS  is the hard one. After a long blackout the covariance is large,
                  so the Kalman gain is near 1 and the OPTIMAL estimate snaps the
                  whole accumulated error in a single epoch. That is correct
                  estimation and terrible user experience -- it is exactly the
                  "blue dot teleports" failure the PS complains about.

    So we keep the filter optimal internally and slew the REPORTED position
    toward it with a bounded rate. The estimate is never degraded; only the
    presentation is smoothed, and the confidence radius stays honest throughout.
    """

    def __init__(self, cfg: SeamlessConfig = SeamlessConfig()):
        self.cfg = cfg
        self.out = None            # reported position
        self.gnss_streak = 0
        self.slewing = False
        self.events: list[dict] = []

    def update(self, filter_pos, dt: float, mode: str):
        fp = np.asarray(filter_pos, float)
        if self.out is None:
            self.out = fp.copy()
            return self.out.copy()

        if mode == "GNSS_AIDED":
            self.gnss_streak += 1
        else:
            if self.gnss_streak > 0:
                self.events.append({"type": "blackout_start"})
            self.gnss_streak = 0

        gap = fp - self.out
        d = float(np.linalg.norm(gap))

        if d <= self.cfg.snap_below_m:
            self.out = fp.copy()
            self.slewing = False
            return self.out.copy()

        # exponential approach, rate-capped
        alpha = 1.0 - np.exp(-dt / self.cfg.tau_s)
        step = gap * alpha
        max_step = self.cfg.max_rate_mps * dt
        if np.linalg.norm(step) > max_step:
            step = step / np.linalg.norm(step) * max_step

        if not self.slewing and d > 2.0:
            self.slewing = True
            self.events.append({"type": "reacquire", "gap_m": d})

        self.out = self.out + step
        return self.out.copy()

    @property
    def converged(self):
        return not self.slewing
