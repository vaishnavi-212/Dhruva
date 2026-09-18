#!/usr/bin/env python3
"""Look at a ride recorded with Sensor Logger / Phyphox and answer one question:
ARE THE SPEED BREAKERS VISIBLE?

    python scripts/inspect_ride.py rides/lap1_accel.csv
    python scripts/inspect_ride.py rides/lap1_accel.csv --marks 41 78 96 133

--marks are seconds where you know a breaker was crossed (from your notes).
They get drawn as vertical lines so you can check the spikes line up.

Outputs rides/<name>_inspect.png with four panels.
"""
from __future__ import annotations
import argparse, sys, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, ACC, WARN = "#131A1C", "#0E6E75", "#B4551A"

ALIASES = {
    "t":  ["seconds_elapsed", "secondselapsed", "time", "timestamp", "t"],
    "x":  ["x", "accx", "acc_x", "ax", "accelerationx"],
    "y":  ["y", "accy", "acc_y", "ay", "accelerationy"],
    "z":  ["z", "accz", "acc_z", "az", "accelerationz"],
}


def _norm(s): return "".join(c for c in str(s).lower() if c.isalnum() or c == "_")


def resolve(df):
    norm = {_norm(c): c for c in df.columns}
    out = {}
    for key, cands in ALIASES.items():
        for c in cands:
            if c in norm:
                out[key] = norm[c]; break
    return out


def load(path):
    df = pd.read_csv(path)
    cols = resolve(df)
    print(f"[inspect] columns found: {list(df.columns)}")
    print(f"[inspect] resolved     : {cols}")
    missing = [k for k in ("t", "x", "y", "z") if k not in cols]
    if missing:
        sys.exit(f"[inspect] could not find {missing}. Edit ALIASES at the top of this file.")

    t = df[cols["t"]].to_numpy(float)
    t = t - t[0]
    if t[-1] > 1e6: t /= 1e9        # nanoseconds
    elif t[-1] > 1e4: t /= 1000.0   # milliseconds

    a = df[[cols["x"], cols["y"], cols["z"]]].to_numpy(float)
    dt = np.median(np.diff(t))
    hz = 1.0 / dt if dt > 0 else float("nan")
    print(f"[inspect] {len(t)} samples · {t[-1]:.1f} s · ~{hz:.0f} Hz")
    if hz < 40:
        print(f"[inspect] !! {hz:.0f} Hz is LOW. Speed breakers are short events -- "
              "set the app to its highest rate (100 Hz+) and re-record.")
    return t, a, hz


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--marks", type=float, nargs="*", default=[],
                    help="seconds where you know a breaker was crossed")
    args = ap.parse_args()

    t, a, hz = load(args.csv)

    # Orientation-independent: total acceleration magnitude minus gravity.
    # We do NOT know how the phone was held, so this is the honest first look.
    mag = np.linalg.norm(a, axis=1) - 9.81

    # High-pass: bumps are transients, riding is slow-varying.
    win = max(int(hz * 0.7) | 1, 3)
    baseline = pd.Series(mag).rolling(win, center=True, min_periods=1).median().to_numpy()
    hp = mag - baseline

    # Energy envelope -- what a detector would actually threshold on.
    env = pd.Series(np.abs(hp)).rolling(max(int(hz*0.15)|1, 3),
                                        center=True, min_periods=1).max().to_numpy()

    noise = np.median(np.abs(hp)) * 1.4826          # robust sigma
    thresh = 5 * noise
    peaks = env > thresh
    n_events = int(np.sum(np.diff(peaks.astype(int)) == 1))

    fig, axes = plt.subplots(4, 1, figsize=(13, 11), sharex=False)

    axes[0].plot(t, mag, lw=.6, color=INK)
    axes[0].set_title("1 · Acceleration magnitude (gravity removed) — raw, orientation-independent")
    axes[0].set_ylabel("m/s²")

    axes[1].plot(t, hp, lw=.6, color=ACC)
    axes[1].axhline(thresh, color=WARN, ls="--", lw=1, label=f"5σ = {thresh:.1f} m/s²")
    axes[1].axhline(-thresh, color=WARN, ls="--", lw=1)
    axes[1].set_title(f"2 · High-passed — transients only.  {n_events} candidate events above 5σ")
    axes[1].set_ylabel("m/s²"); axes[1].legend(fontsize=8, loc="upper right")

    axes[2].plot(t, env, lw=.8, color=ACC)
    axes[2].axhline(thresh, color=WARN, ls="--", lw=1)
    axes[2].fill_between(t, 0, env, where=peaks, color=WARN, alpha=.35)
    axes[2].set_title("3 · Energy envelope — this is what a detector thresholds on")
    axes[2].set_ylabel("m/s²")

    for ax in axes[:3]:
        for m in args.marks:
            ax.axvline(m, color="#7A5AA8", lw=1.2, alpha=.75)
        ax.grid(alpha=.2); ax.set_xlabel("seconds")

    axes[3].specgram(hp, NFFT=256, Fs=max(hz, 1), noverlap=200, cmap="magma")
    axes[3].set_title("4 · Spectrogram — engine vibration sits at a steady frequency; "
                      "a bump is a broadband vertical stripe")
    axes[3].set_ylabel("Hz"); axes[3].set_xlabel("seconds")

    if args.marks:
        fig.text(.5, .008, "purple lines = speed breakers you logged by hand",
                 ha="center", fontsize=9, color="#7A5AA8")

    out = os.path.splitext(args.csv)[0] + "_inspect.png"
    fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig)

    print(f"\n[inspect] noise floor (robust sigma): {noise:.2f} m/s²")
    print(f"[inspect] 5-sigma threshold        : {thresh:.2f} m/s²")
    print(f"[inspect] candidate events         : {n_events}")
    if args.marks:
        print(f"[inspect] breakers you logged      : {len(args.marks)}")
        print("[inspect] -> do the purple lines land on the orange shading? "
              "That is the whole test.")
    print(f"[inspect] wrote {out}")


if __name__ == "__main__":
    main()
