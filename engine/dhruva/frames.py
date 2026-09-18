"""Device frame -> world frame, the way RoNIN does it.

RoNIN's whole trick for handling unknown phone mounting is to rotate gyro and
accelerometer into a GLOBAL frame using the device's own orientation estimate,
so the network never has to learn "which way is the phone pointing".

IO-VNBD gives us ORIENTATION (Azimuth, Pitch, Roll) in degrees, Android
convention. Android's convention is easy to get wrong, so `check_rotation()`
verifies it against physics: gravity, once rotated into the world frame, must
point straight down. If it doesn't, the convention is wrong.
"""
from __future__ import annotations
import numpy as np

G = 9.80665


def rotation_device_to_world(azimuth_deg, pitch_deg, roll_deg) -> np.ndarray:
    """(N,) angles in degrees -> (N,3,3) rotation matrices.

    Android: azimuth about world Z, pitch about device X, roll about device Y.
    world_vec = R @ device_vec
    """
    az = np.deg2rad(np.asarray(azimuth_deg, float))
    pt = np.deg2rad(np.asarray(pitch_deg, float))
    rl = np.deg2rad(np.asarray(roll_deg, float))
    n = len(az)

    ca, sa = np.cos(az), np.sin(az)
    cp, sp = np.cos(pt), np.sin(pt)
    cr, sr = np.cos(rl), np.sin(rl)

    Rz = np.zeros((n, 3, 3)); Rz[:, 2, 2] = 1
    Rz[:, 0, 0] = ca; Rz[:, 0, 1] = -sa
    Rz[:, 1, 0] = sa; Rz[:, 1, 1] = ca

    Rx = np.zeros((n, 3, 3)); Rx[:, 0, 0] = 1
    Rx[:, 1, 1] = cp; Rx[:, 1, 2] = -sp
    Rx[:, 2, 1] = sp; Rx[:, 2, 2] = cp

    Ry = np.zeros((n, 3, 3)); Ry[:, 1, 1] = 1
    Ry[:, 0, 0] = cr; Ry[:, 0, 2] = sr
    Ry[:, 2, 0] = -sr; Ry[:, 2, 2] = cr

    return Rz @ Rx @ Ry


def rotate(R: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Apply (N,3,3) rotations to (N,3) vectors."""
    return np.einsum("nij,nj->ni", R, v)


def check_rotation(R: np.ndarray, gravity_device: np.ndarray, verbose=True) -> dict:
    """Physics check: gravity rotated into world must be ~[0, 0, -g] (or +g).

    Returns diagnostics. If `vertical_fraction` is not close to 1.0, the
    convention is wrong and every downstream feature is garbage.
    """
    gw = rotate(R, gravity_device)
    mag = np.linalg.norm(gw, axis=1)
    vertical = np.abs(gw[:, 2]) / np.maximum(mag, 1e-9)
    horiz = np.linalg.norm(gw[:, :2], axis=1)

    out = {
        "gravity_magnitude_mean": float(np.mean(mag)),
        "vertical_fraction_mean": float(np.mean(vertical)),
        "vertical_fraction_p10": float(np.percentile(vertical, 10)),
        "horizontal_leak_mean_ms2": float(np.mean(horiz)),
        "z_sign": "down" if np.mean(gw[:, 2]) < 0 else "up",
    }
    if verbose:
        print("[frames] gravity check")
        print(f"  |g| mean            : {out['gravity_magnitude_mean']:.3f} m/s²  (expect ~9.81)")
        print(f"  vertical fraction   : {out['vertical_fraction_mean']:.4f}  (expect >0.99)")
        print(f"  p10 vertical        : {out['vertical_fraction_p10']:.4f}")
        print(f"  horizontal leak     : {out['horizontal_leak_mean_ms2']:.3f} m/s²  (expect <1)")
        print(f"  world Z points      : {out['z_sign']}")
        verdict = "PASS" if out["vertical_fraction_mean"] > 0.99 else "FAIL — convention is wrong"
        print(f"  -> {verdict}")
    return out
