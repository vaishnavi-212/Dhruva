"""Run the trained A2 distance model on a Sensor Logger ride.

The model was trained on IO-VNBD: CARS, UK/Nigeria/France, native 10 Hz, 6
level-frame channels, 300-sample (30 s) windows, predicting distance travelled.

Applying it to an Indian motorcycle recorded at 99 Hz is a real domain shift on
three axes at once (vehicle, country, sample rate). That IS contribution 6, so
the number this produces is a measurement, not a formality.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import torch

from dhruva import align
from dhruva.model import ResNet1D

TRAIN_HZ = 10.0


def load_imu(folder: str, target_hz: float = TRAIN_HZ):
    """Sensor Logger export -> (t, acc, gyro) resampled to the training rate."""
    a = pd.read_csv(os.path.join(folder, "Accelerometer.csv"))
    g = pd.read_csv(os.path.join(folder, "Gyroscope.csv"))
    ta = a["seconds_elapsed"].to_numpy(float)
    tg = g["seconds_elapsed"].to_numpy(float)
    t0, t1 = max(ta[0], tg[0]), min(ta[-1], tg[-1])
    grid = np.arange(t0, t1, 1.0 / target_hz)

    def rs(t, v):
        return np.column_stack([np.interp(grid, t, v[:, i]) for i in range(v.shape[1])])

    acc = rs(ta, a[["x", "y", "z"]].to_numpy(float))
    gyro = rs(tg, g[["x", "y", "z"]].to_numpy(float))
    return grid - grid[0], acc, gyro


def features(acc, gyro, hz=TRAIN_HZ):
    """Same 6 level-frame channels the model was trained on.

    Training removed gravity via IO-VNBD's gravity channels; Sensor Logger's
    Accelerometer.csv is already gravity-free, so we reconstruct a raw-equivalent
    for the tilt estimate by adding a constant 1 g along the low-passed direction.
    """
    n = max(int(8.0 * hz) | 1, 3)
    smooth = pd.DataFrame(acc).rolling(n, center=True, min_periods=1).mean().to_numpy()
    d = np.linalg.norm(smooth, axis=1, keepdims=True)
    down = np.divide(smooth, np.maximum(d, 1e-6))
    acc_raw = acc + down * 9.80665          # put gravity back for the tilt estimate

    R = align.estimate_tilt(acc_raw, hz)
    return np.concatenate([align.apply(R, gyro), align.apply(R, acc)], axis=1).astype(np.float32)


def predict_speed(folder: str, ckpt="checkpoints/speed_best.pt", device=None):
    """-> (t, speed_mps) at the training rate."""
    c = torch.load(ckpt, map_location="cpu", weights_only=False)
    W = int(c["window"]); mean, std = np.asarray(c["mean"]), np.asarray(c["std"])
    dev = device or ("mps" if torch.backends.mps.is_available() else "cpu")
    net = ResNet1D(in_ch=6, out_dim=1).to(dev); net.load_state_dict(c["model"]); net.eval()

    t, acc, gyro = load_imu(folder)
    F = features(acc, gyro)
    if len(F) < W:
        raise SystemExit(f"{folder}: only {len(F)} samples, need {W}")

    idx = np.arange(0, len(F) - W, 1)
    X = np.stack([F[i:i + W] for i in idx]).transpose(0, 2, 1)
    X = (X - mean) / std
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), 256):
            preds.append(net(torch.tensor(X[i:i + 256], dtype=torch.float32).to(dev)).cpu().numpy())
    dist = np.concatenate(preds).ravel()          # metres per 30 s window

    # window distance -> instantaneous speed, assigned to the window centre
    v = np.clip(dist / (W / TRAIN_HZ), 0, None)
    tc = t[idx + W // 2]
    speed = np.interp(t, tc, v, left=v[0], right=v[-1])
    return t, speed
