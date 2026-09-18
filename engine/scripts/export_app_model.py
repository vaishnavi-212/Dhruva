#!/usr/bin/env python3
"""Export a speed-model checkpoint to ONNX for the phone app, and prove the export is exact.

Writes, next to the checkpoint:
  <name>.onnx        opset 17, input "imu" (batch, 6, 300), output "dist_m" (batch, 1)
  <name>_norm.json   window, rate, channel order, per-channel mean/std -- the app normalises with these

Parity gate: PyTorch and ONNX Runtime must agree to < 1e-3 m per 30 s window on real rides the
app model never saw (the 11 Sept recordings). Measured 1.07e-4 m on 12 Sept.

    python scripts/export_app_model.py checkpoints/app/speed_app.pt
    python scripts/export_app_model.py checkpoints/app/speed_app.pt --frame replica   # parity in that frame
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dhruva.model import ResNet1D

PARITY_RIDES = ["rides/app_DhruvaRun_2026-09-11_19-37-43", "rides/app_DhruvaRun_2026-09-11_19-41-28",
                "rides/app_DhruvaRun_2026-09-11_13-42-07"]
TOLERANCE_M = 1e-3


def rides_features(frame):
    if frame == "offline":
        from dhruva.infer_speed import load_imu, features
        return [features(*load_imu(r)[1:]) for r in PARITY_RIDES]
    import eval_causal_features as EC
    from train_eval_phone_features import frame_features
    return [frame_features(frame, *EC.load(r)[1:]) for r in PARITY_RIDES]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--frame", default="offline", choices=["offline", "replica", "gravity"],
                    help="feature frame the checkpoint was trained in (used for the parity inputs)")
    a = ap.parse_args()
    import onnx, onnxruntime as ort

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    net = ResNet1D(in_ch=6, out_dim=1); net.load_state_dict(ck["model"]); net.eval()
    W = int(ck["window"])
    stem = os.path.splitext(a.ckpt)[0]
    onnx_path, norm_path = stem + ".onnx", stem + "_norm.json"
    torch.onnx.export(net, torch.zeros(1, 6, W), onnx_path, input_names=["imu"], output_names=["dist_m"],
                      dynamic_axes={"imu": {0: "batch"}, "dist_m": {0: "batch"}}, opset_version=17, dynamo=False)
    onnx.checker.check_model(onnx.load(onnx_path))

    mean, std = np.asarray(ck["mean"], dtype=np.float64).ravel(), np.asarray(ck["std"], dtype=np.float64).ravel()
    with open(norm_path, "w") as f:
        json.dump({"window": W, "model_hz": 10.0, "frame": a.frame,
                   "channels": ["gyro_e", "gyro_n", "gyro_d", "acc_e", "acc_n", "acc_d"],
                   "mean": mean.tolist(), "std": std.tolist(), "output": "metres travelled over the window",
                   "source_checkpoint": os.path.basename(a.ckpt),
                   **{k: ck[k] for k in ("trained_on", "epochs", "seed", "frame") if k in ck}}, f, indent=1)

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    worst, n = 0.0, 0
    for F in rides_features(a.frame):
        idx = np.arange(0, len(F) - W, 5)
        X = ((np.stack([F[i:i + W] for i in idx]).transpose(0, 2, 1) - mean[None, :, None]) / std[None, :, None]).astype(np.float32)
        with torch.no_grad():
            p = net(torch.from_numpy(X)).numpy().ravel()
        q = sess.run(None, {"imu": X})[0].ravel()
        worst, n = max(worst, float(np.max(np.abs(p - q)))), n + len(X)
    print(f"{onnx_path}: {os.path.getsize(onnx_path)/1e6:.2f} MB, onnxruntime {ort.__version__}; "
          f"worst |PyTorch - ONNX| over {n} windows = {worst:.2e} m")
    if worst >= TOLERANCE_M:
        raise SystemExit(f"PARITY FAILED: {worst:.2e} m >= {TOLERANCE_M} m")
    print(f"parity OK; normalisation -> {norm_path}")


if __name__ == "__main__":
    main()
