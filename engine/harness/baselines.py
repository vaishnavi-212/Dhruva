"""Baselines to beat.

Every model you build plugs in with the same signature, so the harness can
score anything without changes:

    predict(acc, gyro, dt, init) -> (xy, sigma | None)

    acc  : (N,3) m/s^2  gyro : (N,3) rad/s  dt : float seconds
    init : dict with 'xy' (2,), 'v' (2,) velocity at blackout start, 'heading' rad
    xy   : (N,2) predicted positions, metres, same frame as init['xy']
    sigma: (N,) optional 1-sigma radius in metres, or None
"""
from __future__ import annotations
import numpy as np


def frozen(acc, gyro, dt, init):
    """What a phone does when it gives up: the dot stops. This is the floor."""
    n = len(acc)
    return np.repeat(init["xy"][None, :], n, axis=0), None


def constant_velocity(acc, gyro, dt, init):
    """Carry the last known velocity forward. Roughly what naive DR does."""
    n = len(acc)
    t = np.arange(n) * dt
    return init["xy"][None, :] + np.outer(t, init["v"]), None


def naive_integration(acc, gyro, dt, init):
    """Double-integrate acceleration with gyro heading. Drifts catastrophically.

    This is the 'before' picture. Keep it in every chart -- the gap between this
    and your model IS the contribution.
    """
    n = len(acc)
    heading = init.get("heading", 0.0)
    # integrate yaw rate (z axis) for heading
    yaw = heading + np.cumsum(gyro[:, 2]) * dt
    # forward acceleration = x axis in vehicle frame (assumes calibrated mount)
    a_fwd = acc[:, 0]
    speed = np.linalg.norm(init["v"]) + np.cumsum(a_fwd) * dt
    speed = np.clip(speed, 0, None)
    vx, vy = speed * np.cos(yaw), speed * np.sin(yaw)
    xy = init["xy"][None, :] + np.column_stack([np.cumsum(vx), np.cumsum(vy)]) * dt
    # crude uncertainty: grows with the square of elapsed time
    sigma = 0.05 * (np.arange(n) * dt) ** 2 + 1.0
    return xy, sigma


def oracle_speed(acc, gyro, dt, init, true_speed=None):
    """Upper bound: perfect speed, gyro heading only.

    Tells you how much of your error is speed estimation vs heading. If this is
    already bad, fix heading first.
    """
    n = len(acc)
    yaw = init.get("heading", 0.0) + np.cumsum(gyro[:, 2]) * dt
    s = true_speed if true_speed is not None else np.full(n, np.linalg.norm(init["v"]))
    vx, vy = s * np.cos(yaw), s * np.sin(yaw)
    xy = init["xy"][None, :] + np.column_stack([np.cumsum(vx), np.cumsum(vy)]) * dt
    return xy, None


REGISTRY = {
    "frozen":            frozen,
    "constant_velocity": constant_velocity,
    "naive_integration": naive_integration,
}
