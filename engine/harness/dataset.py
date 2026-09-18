"""IO-VNBD loader.

The IO-VNBD release ships several CSV layouts (vehicle CAN stream vs the
Android smartphone stream). Column names are NOT guaranteed to match what we
guess here, so this loader fuzzy-matches and then PRINTS what it resolved.

Run `python -m harness.dataset <file.csv>` once and check the printed mapping
before trusting any numbers. Edit ALIASES if it guesses wrong.
"""
from __future__ import annotations
import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd

# Edit these if the fuzzy match picks the wrong column.
ALIASES = {
    "t":      ["timesincestartms", "secondselapsed", "time", "timestamp", "elapsed"],
    "ax":     ["accelerometerx", "accx", "acc_x", "accelx", "linearaccx"],
    "ay":     ["accelerometery", "accy", "acc_y", "accely", "linearaccy"],
    "az":     ["accelerometerz", "accz", "acc_z", "accelz", "linearaccz"],
    "gx":     ["gyroscopex", "gyrox", "gyro_x", "angratex"],
    "gy":     ["gyroscopey", "gyroy", "gyro_y", "angratey"],
    "gz":     ["gyroscopez", "gyroz", "gyro_z", "angratez"],
    "lat":    ["gpslatitude", "latitude", "gpslat"],
    "lon":    ["gpslongitude", "longitude", "gpslon", "lng"],
    "speed":  ["gpsspeed", "speed", "velocity", "wheelspeed"],
    "grav_x": ["gravityx", "gravx", "grav_x"],
    "grav_y": ["gravityy", "gravy", "grav_y"],
    "grav_z": ["gravityz", "gravz", "grav_z"],
    "mag_x": ["magneticfieldx", "magnetometerx", "magx"],
    "mag_y": ["magneticfieldy", "magnetometery", "magy"],
    "mag_z": ["magneticfieldz", "magnetometerz", "magz"],
}

# A column is never matched to these keys if its name contains one of these.
EXCLUDE = {
    "ax": ["gravity", "magnetic", "orientation"],
    "ay": ["gravity", "magnetic", "orientation"],
    "az": ["gravity", "magnetic", "orientation"],
    "gx": ["gravity", "magnetic"], "gy": ["gravity", "magnetic"],
    "gz": ["gravity", "magnetic"],
    "speed": ["accuracy"],
}

REQUIRED = ["t", "ax", "ay", "az", "gx", "gy", "gz", "lat", "lon"]


def _norm(s: str) -> str:
    return "".join(ch for ch in s.lower() if ch.isalnum())


def resolve_columns(df: pd.DataFrame) -> dict[str, str]:
    norm = {_norm(c): c for c in df.columns}
    out: dict[str, str] = {}
    for key, cands in ALIASES.items():
        bad = EXCLUDE.get(key, [])
        best = None
        for cand in sorted(cands, key=len, reverse=True):
            for nc, orig in norm.items():
                if any(b in nc for b in bad):
                    continue
                if nc == cand or nc.startswith(cand):
                    best = orig; break
            if best: break
        if not best:  # loose substring fallback
            for cand in sorted(cands, key=len, reverse=True):
                for nc, orig in norm.items():
                    if any(b in nc for b in bad):
                        continue
                    if cand in nc:
                        best = orig; break
                if best: break
        if best:
            out[key] = best
    return out


@dataclass
class Trace:
    """One continuous drive, resampled to a fixed rate."""
    t: np.ndarray          # seconds from start
    acc: np.ndarray        # (N,3) m/s^2, phone frame, gravity removed
    gyro: np.ndarray       # (N,3) rad/s, phone frame
    lat: np.ndarray
    lon: np.ndarray
    xy: np.ndarray         # (N,2) local ENU metres, origin at first sample
    speed: np.ndarray | None
    hz: float
    name: str
    fix_idx: np.ndarray | None = None   # rows where GPS actually updated
    acc_raw: np.ndarray | None = None   # (N,3) accelerometer BEFORE gravity removal
    mag: np.ndarray | None = None       # (N,3) magnetometer, phone frame

    def __len__(self) -> int:
        return len(self.t)

    @property
    def distance(self) -> np.ndarray:
        """Cumulative ground-truth distance travelled, metres."""
        step = np.linalg.norm(np.diff(self.xy, axis=0), axis=1)
        return np.concatenate([[0.0], np.cumsum(step)])


def latlon_to_local_xy(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Equirectangular projection about the first fix. Fine over a few km."""
    R = 6_378_137.0
    lat0, lon0 = np.deg2rad(lat[0]), np.deg2rad(lon[0])
    la, lo = np.deg2rad(lat), np.deg2rad(lon)
    x = (lo - lon0) * np.cos(lat0) * R
    y = (la - lat0) * R
    return np.column_stack([x, y])


def load(path: str, hz: float = 10.0, verbose: bool = True) -> Trace:
    df = pd.read_csv(path, encoding='latin-1', skipinitialspace=True)
    cols = resolve_columns(df)
    missing = [k for k in REQUIRED if k not in cols]
    if verbose:
        print(f"[dataset] {path}")
        print(f"[dataset] resolved: {cols}")
        if missing:
            print(f"[dataset] !! MISSING {missing} -- edit ALIASES in harness/dataset.py")
    if missing:
        raise KeyError(f"unresolved columns: {missing}")

    t = df[cols["t"]].to_numpy(float)
    if len(t) < 100:
        raise ValueError(f"{path}: only {len(t)} rows -- unusable, skip this file")
    t = t - t[0]

    # Units: infer from the SAMPLING INTERVAL, never from the total magnitude.
    # A short recording in milliseconds has a small total, so a magnitude test
    # silently leaves it unconverted and the resampler then invents ~1000x the
    # samples. Median dt is scale-correct regardless of recording length.
    dt_raw = float(np.median(np.diff(t)))
    if "ms" in _norm(cols["t"]) or (1.0 < dt_raw <= 1e4):
        t = t / 1000.0                      # milliseconds
    elif dt_raw > 1e4:
        t = t / 1e6                         # microseconds

    if t[-1] <= t[0]:
        raise ValueError(f"{path}: timestamps do not increase "
                         f"(span {t[-1]-t[0]:.1f}s) -- corrupt, skip this file")

    acc = df[[cols["ax"], cols["ay"], cols["az"]]].to_numpy(float)
    acc_raw_full = acc.copy()
    # If gravity channels exist, remove gravity (IO-VNBD provides them).
    if all(k in cols for k in ("grav_x", "grav_y", "grav_z")):
        g = df[[cols["grav_x"], cols["grav_y"], cols["grav_z"]]].to_numpy(float)
        acc = acc - g
        if verbose:
            print("[dataset] gravity channels found and removed")

    gyro = df[[cols["gx"], cols["gy"], cols["gz"]]].to_numpy(float)
    mag_full = (df[[cols["mag_x"], cols["mag_y"], cols["mag_z"]]].to_numpy(float)
                if all(k in cols for k in ("mag_x","mag_y","mag_z")) else None)
    lat = df[cols["lat"]].to_numpy(float)
    lon = df[cols["lon"]].to_numpy(float)
    speed = df[cols["speed"]].to_numpy(float) if "speed" in cols else None
    # NOTE: IO-VNBD labels this column "(Kmh)" but the values are m/s.
    # Verified against position-derived speed: corr 0.833, ratio 0.987.
    # Do NOT divide by 3.6.

    ok = np.isfinite(lat) & np.isfinite(lon) & (np.abs(lat) > 1e-6)
    if ok.sum() < 100:
        raise ValueError(f"{path}: only {int(ok.sum())} rows have a valid GPS fix "
                         "-- unusable, skip this file")
    t, acc, gyro, lat, lon = t[ok], acc[ok], gyro[ok], lat[ok], lon[ok]
    acc_raw_full = acc_raw_full[ok]
    if mag_full is not None: mag_full = mag_full[ok]
    if speed is not None:
        speed = speed[ok]

    # Uniform resample so outage windows are exact.
    grid = np.arange(t[0], t[-1], 1.0 / hz)
    def rs(a):
        if a.ndim == 1:
            return np.interp(grid, t, a)
        return np.column_stack([np.interp(grid, t, a[:, i]) for i in range(a.shape[1])])

    tr = Trace(
        t=grid - grid[0], acc=rs(acc), acc_raw=rs(acc_raw_full), gyro=rs(gyro),
        lat=rs(lat), lon=rs(lon), xy=np.zeros((len(grid), 2)),
        speed=rs(speed) if speed is not None else None,
        mag=rs(mag_full) if mag_full is not None else None,
        hz=hz, name=path.split("/")[-1],
    )
    tr.xy = latlon_to_local_xy(tr.lat, tr.lon)

    # --- GPS staircase handling -------------------------------------------
    # IO-VNBD carries GPS at ~0.11 Hz inside a 10 Hz file: lat/lon repeat until
    # a new fix lands, then jump. Ground truth must be interpolated between real
    # fixes, and blackout boundaries must land ON real fixes.
    moved = np.r_[True, (np.diff(tr.lat) != 0) | (np.diff(tr.lon) != 0)]
    fix = np.where(moved)[0]
    tr.fix_idx = fix
    if len(fix) > 2:
        gap = float(np.median(np.diff(tr.t[fix])))
        tr.xy = np.column_stack([
            np.interp(tr.t, tr.t[fix], tr.xy[fix, 0]),
            np.interp(tr.t, tr.t[fix], tr.xy[fix, 1]),
        ])
        if verbose:
            print(f"[dataset] GPS fixes: {len(fix)} ({len(fix)/len(tr)*100:.1f}% of rows), "
                  f"~{gap:.1f}s apart -> ground truth interpolated between fixes")
    if verbose:
        print(f"[dataset] {len(tr)} samples @ {hz} Hz, "
              f"{tr.t[-1]/60:.1f} min, {tr.distance[-1]/1000:.2f} km")
    return tr


if __name__ == "__main__":
    load(sys.argv[1])
