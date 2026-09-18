"""No map vs road-bound on the 11 benchmark rides, each with its own held-out model,
against how much the route turns. Mirrors loro_eval.py (wide_ways.json, pad 400)."""
import glob, os, sys, json, numpy as np, pandas as pd, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import selfheal_eval as SE          # chdirs to repo, gives free_dr/resample
from dhruva.osm_road import OsmNetwork, road_from_osm, to_xy
from dhruva.mapmatch import RoadPolyline
from dhruva.gpsclean import clean_for_truth
from dhruva.infer_speed import load_imu, features
from evaluate_system import speed_from_imu, _extend

net = OsmNetwork("data/osm/wide_ways.json", SE.lat0, SE.lon0)
runs = sorted(glob.glob("rides/bike/*/")) + sorted(glob.glob("rides/aug31/*/"))
out, curves = [], {}
for r in runs:
    name = os.path.basename(r[:-1])
    ck = torch.load(f"checkpoints/loro/{name}.pt", map_location="cpu", weights_only=False)
    L = pd.read_csv(os.path.join(r, "Location.csv")); lt = L.seconds_elapsed.to_numpy(float)
    gps = to_xy(L.latitude.to_numpy(float), L.longitude.to_numpy(float), SE.lat0, SE.lon0)
    gps, lt = clean_for_truth(gps, lt)
    road = RoadPolyline(_extend(road_from_osm(net, gps)[0], 400.0), 2.0)
    t, v = speed_from_imu(r, ck); v = np.clip(v, 0, None)
    tt, acc, gyro = load_imu(r); wz = features(acc, gyro)[:, 2]
    dt = float(np.median(np.diff(t)))
    truth = np.column_stack([np.interp(t, lt, gps[:, 0]), np.interp(t, lt, gps[:, 1])])
    cum = np.r_[0, np.cumsum(np.linalg.norm(np.diff(truth, axis=0), axis=1))]
    pr = road.at(road.project(gps[0]) + np.cumsum(v) * dt)[0]
    er = np.linalg.norm(pr - truth, axis=1)
    vg = np.r_[0, np.diff(cum)] / dt
    nd = SE.free_dr(truth, cum, vg, t, wz, [v], dt)
    # the free track itself, for the error-vs-distance curve
    rs = SE.resample(gps, 5.0); h = np.unwrap(np.arctan2(*np.diff(rs, axis=0)[:, ::-1].T))
    turn_per_km = float(np.degrees(np.sum(np.abs(np.diff(h)))) / (cum[-1] / 1000))
    rec = dict(ride=name, dist_m=float(cum[-1]), road_drift=float(100 * er[-1] / cum[-1]), none_drift=nd["model"],
               none_oracle=nd["oracle"], turn_deg_per_km=turn_per_km, bias_dps=nd["bias_dps"])
    out.append(rec)
    print(f"{name[:40]:40s} {cum[-1]:5.0f}m turn {turn_per_km:5.0f} deg/km | no map {nd['model']:6.1f}% (perfect speed {nd['oracle']:5.1f}%) | road {rec['road_drift']:5.1f}%", flush=True)
d = np.array([o["road_drift"] for o in out]); n = np.array([o["none_drift"] for o in out])
print(f"median road {np.median(d):.1f}% ({(d<10).sum()}/11) | median no map {np.median(n):.1f}% ({(n<10).sum()}/11)")
json.dump(out, open(os.path.join(HERE, "turns_results.json"), "w"), indent=1)
