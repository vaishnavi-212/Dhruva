# Dhruva engine

The Python side of Dhruva: the AI speed model and its training, road-bound dead reckoning, the
GNSS+INS fusion filter, the 200 Hz edge engine, the self-healing map and route experiments, and the
ONNX export the Android app runs. Every number we quote comes from a command below, run on data in
this folder.

**Problem statement:** SIH26168 (ISRO), AI-ML based Intelligent Dead Reckoning system for seamless
navigation. Full text: [`../docs/PROBLEM_STATEMENT.md`](../docs/PROBLEM_STATEMENT.md).
**Hardware:** none. A smartphone's own sensors only.

## Setup

```bash
cd engine
python3 -m pip install -r requirements.txt     # Python 3.12; torch 2.7.0, onnxruntime 1.29.0 (same as the app)
```

Run every command from inside `engine/`.

## Problem statement → where it lives

| Required capability | Where | Runs on the phone today? |
|---|---|---|
| In-vehicle alignment & calibration (pitch, roll, yaw) | `dhruva/align.py` | pitch/roll yes (level frame in `ImuFeatures.kt`); yaw laptop |
| AI speed & vibration filter | `dhruva/model.py` (ResNet1D, 997,633 params), `scripts/train_*` | **yes**, `SpeedModel.kt` via ONNX Runtime |
| Map matching & kinematic constraints | `dhruva/mapmatch.py`, `osm_road.py`, `roadgraph.py`, `hmm.py` | road binding yes (`RoadBinder.kt`); HMM laptop |
| GNSS+INS fusion engine | `dhruva/fusion.py`, `scripts/eval_fusion.py` | laptop only |
| Seamless GNSS deficit handler | `dhruva/fusion.py` (no mode branch), `scripts/eval_fusion.py` | manual switch in the app |
| Edge-deployable engine (external IMU, ~200 Hz) | `dhruva/edge.py`, `scripts/eval_edge.py` | — (runs on any Python host) |
| Position plot on IO-VNBD | `scripts/fetch_iovnbd.py`, `run_baseline.py`, `harness/` | — |

## What is here

| Folder | Contents |
|---|---|
| `dhruva/` | the engine: speed model, road binding, fusion filter, edge engine, confidence radius, landmarks and self-healing map, levelling |
| `scripts/` | training, evaluation, export, route and dashboard tools |
| `harness/` | ISRO drift-% scoring used everywhere |
| `contracts/` | the run-file format shared by the app and the dashboard |
| `checkpoints/speed_best.pt` | speed model pre-trained on IO-VNBD (67 smartphone files, 5 held out) |
| `checkpoints/loro/` | 11 leave-one-ride-out models on offline (look-ahead) features: each never saw the ride it is scored on |
| `checkpoints/loro_phone_replica/` | the same 11, retrained on the look-back features a phone can compute |
| `checkpoints/app/` | the model shipped in the app (`speed_app_replica.*`: 11 benchmark rides, phone features) and its ONNX export |
| `rides/` | 15 two-wheeler recordings on KLE Tech campus roads, Hubballi: 11 benchmark rides (30–31 Aug) and 4 on 11 Sept. Files: `Accelerometer` (linear, m/s²), `Gyroscope` (rad/s), `Gravity` (m/s²), `Location` (GPS, 1 Hz) |
| `data/osm/`, `data/landmarks/` | OpenStreetMap road geometry for the area; 14 speed breakers surveyed on foot |
| `results/` | self-healing map and no-map analysis with charts; phone-feature, route-guard and re-route results |

IO-VNBD is not included: `python scripts/fetch_iovnbd.py --list` (dataset: github.com/onyekpeu/IO-VNBD).

## Reproduce the numbers (re-run 12 Sept 2026)

| Claim | Command | Result |
|---|---|---|
| Benchmark drift, 11 rides, held-out models, road-bound | `LANDMARKS=0 python scripts/loro_eval.py` | **6.8% median, 9 of 11 under 10%** |
| Demo rides, GPS cut mid-ride, all 11 models | `python scripts/make_demo_dashboards.py` | 7:40 ride **3.4%**, 7:44 ride **4.5%**, 11 of 11 each |
| Surveyed speed breakers found | `python scripts/make_campus_dashboard.py` | 13 of 14, both laps |
| GNSS+INS fusion through a 60%-of-route blackout | `python scripts/eval_fusion.py` | 9.4% median, 7 of 11; jump 0.92 m at GPS loss, ≤ 2.5 m at return |
| 200 Hz edge engine | `python scripts/eval_edge.py` | 30× real time, 531 KiB, 9.0% median drift |
| No map vs road; self-healing map | `cd results/selfheal/work && python cache_speed.py && python selfheal_eval.py && python turns_eval.py && python curves_eval.py && python cut_eval.py && python make_figures.py` | no map 50% vs road 6.8%; 7:44 ride 33.1% → 3.5% on a road learned from the 7:40 ride |
| What a phone can compute (no look-ahead) | `python scripts/eval_causal_features.py` | offline-trained models on phone inputs: 8.0% median, 6 of 11 |
| Models retrained on phone inputs | `python scripts/train_eval_phone_features.py` | **5.0% median, 10 of 11** (laptop replay of the phone pipeline) |
| Wrong-turn detection on a planned route | `python scripts/eval_route_guard.py` | mounted phone: 14 of 14 caught, 9 of 10 correct rides quiet (thresholds tuned on these rides) |
| Re-binding after a wrong turn | `python scripts/eval_reroute.py` | dot error 30 / 41 / 56 m vs 103 / 155 / 205 m left on the route |

## The model on the phone

A phone cannot look ahead. The offline features use centred windows (up to 15 s into the future), so the
app model is trained on look-back "replica frame" features, exactly what `ImuFeatures.kt` builds.

```bash
python scripts/train_app_model.py --frame replica          # checkpoints/app/speed_app_replica.pt
python scripts/export_app_model.py checkpoints/app/speed_app_replica.pt --frame replica   # .onnx + _norm.json, parity-checked
cp checkpoints/app/speed_app_replica.onnx ../app/src/main/assets/speed_app.onnx
cp checkpoints/app/speed_app_replica_norm.json ../app/src/main/assets/speed_app_norm.json
python scripts/make_golden_features.py --out ../app/src/test/resources/golden
cd .. && ./gradlew :app:testDebugUnitTest                  # Kotlin features must match this engine
```

| Check | Result |
|---|---|
| ONNX Runtime vs PyTorch | within 0.000076 m per 30 s window |
| Kotlin features vs Python | within 0.000000005 |
| Shipped model, 7:40 ride from the real GPS cut (unseen) | **1.6%** (phone app on held GPS speed: 9.0%) — laptop replay |
| Shipped model, 7:44 ride from the real GPS cut (unseen) | **8.9%** (phone app on held GPS speed: 25.9%) — laptop replay |
| Afternoon laps with crawling (unseen) | 21% and 29%: crawling is still over-read |

## How to read the numbers

- **Drift** = distance between the final estimate and the true position ÷ distance travelled without GPS. ISRO's limit is 10%.
- Every accuracy number comes from models that never saw the ride being scored.
- On our campus rides everyone rode at about 5 m/s, so a fixed speed chosen afterwards does about as well as the model.
  The model earns its place when speed changes (IO-VNBD cars 25–87 km/h: 3.5% distance error vs 29% for a fixed speed).
- "Laptop replay" = the phone's exact pipeline run on a recorded ride. On-phone measurements are in progress.
