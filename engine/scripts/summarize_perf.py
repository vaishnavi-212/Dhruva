#!/usr/bin/env python3
"""Summarise one or more perf_*.csv files written by the app's PerfLog.

For each file: phone test length, time per model run (mean and worst), memory, and battery drain
per hour, split into the stretch where the model was running and where it was standing by.

    python scripts/summarize_perf.py perf_2026-09-20_10-00-00.csv [more.csv ...]
"""
from __future__ import annotations
import sys
import pandas as pd


def drain_per_hour(d: pd.DataFrame) -> float:
    """Battery percentage points lost per hour over a stretch of rows (NaN if under 10 minutes)."""
    if len(d) < 2:
        return float("nan")
    hours = (d.elapsed_s.iloc[-1] - d.elapsed_s.iloc[0]) / 3600.0
    return float("nan") if hours < 1 / 6 else (d.battery_pct.iloc[0] - d.battery_pct.iloc[-1]) / hours


def main(paths):
    for p in paths:
        d = pd.read_csv(p)
        on, off = d[d.ai_active == 1], d[d.ai_active == 0]
        last = d.iloc[-1]
        print(f"\n{p}")
        print(f"  length                 {d.elapsed_s.iloc[-1] / 60:.1f} min, {len(d)} rows")
        print(f"  model runs             {int(last.inferences)}")
        print(f"  time per run           mean {last.infer_ms_mean:.2f} ms, worst {last.infer_ms_max:.2f} ms")
        print(f"  memory (PSS)           median {d.pss_mb.median():.0f} MB, max {d.pss_mb.max():.0f} MB "
              f"(native heap max {d.native_heap_mb.max():.0f} MB)")
        print(f"  battery drain          model running {drain_per_hour(on):.1f} %/h, standing by {drain_per_hour(off):.1f} %/h")
        print(f"  battery temperature    {d.temp_c.min():.1f}-{d.temp_c.max():.1f} °C")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1:])
