import glob, os, sys, numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))   # repo root
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts")); os.chdir(ROOT)
from evaluate_system import speed_from_imu
from dhruva.infer_speed import load_imu, features
OUT = os.path.join(HERE, "speed_cache.npz")
rides = ["13-42-07", "13-46-20", "19-37-43", "19-41-28"]
ck = sorted(glob.glob("checkpoints/loro/*.pt"))
res = {}
for r in rides:
    d = f"rides/app_DhruvaRun_2026-09-11_{r}"
    t, acc, gyro = load_imu(d); F = features(acc, gyro)
    res[f"{r}|t"] = t; res[f"{r}|wz"] = F[:, 2]
    for c in ck:
        tt, v = speed_from_imu(d, torch.load(c, map_location="cpu", weights_only=False))
        res[f"{r}|{os.path.basename(c)[:-3]}"] = v
        print(r, os.path.basename(c), f"{np.sum(np.clip(v,0,None))*np.median(np.diff(tt)):.0f} m", flush=True)
np.savez(OUT, **res); print("saved", OUT)
