#!/usr/bin/env python3
"""Slide charts for the IO-VNBD position results (reads results/iovnbd from scripts/eval_iovnbd.py).

  iovnbd_position_plot.png   one 1 km blackout at motorway speed, drawn on the map. Chosen by rule, not by eye:
                             of all 1 km blackouts at >= 50 km/h, the one whose Dhruva drift is closest to the median
  iovnbd_summary.png         every blackout on all 5 held-out drives: median drift and share inside ISRO's 10%

    python scripts/plot_iovnbd.py
"""
from __future__ import annotations
import json, os, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import dataset as ds
from fetch_osm_iovnbd import clean_fixes

OUT = "results/iovnbd"
INK, NOMAP, DHRUVA, RAW, OSM, ISRO, GRID, MUTED, FROZEN = ("#1F2A37", "#C8553D", "#1F8A7A", "#7FBFB4", "#8D99AE",
                                                           "#D9922E", "#E6E9EE", "#5B6573", "#6C4F9E")
plt.rcParams.update({"font.family": "Avenir Next", "font.size": 12, "axes.edgecolor": "#C9CED6", "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": "white", "axes.facecolor": "white", "savefig.dpi": 200})
NAMES = {"frozen": "GPS frozen", "AI + gyro": "no map\n(AI speed + gyro)", "AI + road": "Dhruva\n(AI speed on road)",
         "AI cal + road": "Dhruva, calibrated\nbefore the cut", "perfect + road": "true speed on road\n(map error only)"}
COLORS = {"frozen": FROZEN, "AI + gyro": NOMAP, "AI + road": RAW, "AI cal + road": DHRUVA, "perfect + road": OSM}


def title(fig, main, sub):
    fig.text(0.04, 0.955, main, fontsize=19, fontweight="demibold", color=INK, va="top")
    fig.text(0.04, 0.895, sub, fontsize=12, color=MUTED, va="top")


def along(road, pts):
    """Arc length of each point along a polyline (nearest vertex; the road is sampled every 2 m)."""
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(road, axis=0), axis=1))]
    return s, s[np.argmin(np.linalg.norm(road[None] - pts[:, None], axis=2), axis=1)]


def at(road, s, q):
    return np.column_stack([np.interp(q, s, road[:, 0]), np.interp(q, s, road[:, 1])])


D = pd.read_csv(f"{OUT}/iovnbd_blackouts.csv")
P = json.load(open(f"{OUT}/paths.json"))
for m in NAMES:
    D[f"pct_{m}"] = 100 * D[f"err_{m}"] / D.dist_m

# ---------- 1. position plot ----------
fast = D[(D.target_m == 1000.0) & (D.speed_kmh >= 50)]
r = fast.loc[(fast["pct_AI cal + road"] - fast["pct_AI cal + road"].median()).abs().idxmin()]
p = {k: np.asarray(v) for k, v in P[f"{r.drive}|{int(r.i0)}|{int(r.i1)}"].items()}
truth, free, road, rp = p["truth"], p["free"], p["road"], p["road_path"]
s, s_rp = along(road, np.vstack([truth[:1], rp]))
cal = at(road, s, s_rp[0] + r.cal_scale * (s_rp[1:] - s_rp[0]))

tr = ds.load(f"data/{r.drive}.csv", verbose=False); fx = clean_fixes(tr)
whole = tr.xy[fx]

fig = plt.figure(figsize=(13, 7.3))
title(fig, f"IO-VNBD position plot: GPS lost for 1 km at {r.speed_kmh:.0f} km/h",
      f"Held-out drive {r.drive} (Coventry, UK), never seen in training. GPS off for {r.dist_m:.0f} m, "
      f"{r.dur_s:.0f} s. Picked as the median Dhruva result of {len(fast)} such blackouts.")
ov = fig.add_axes([0.02, 0.10, 0.25, 0.70]); ov.set_aspect("equal"); ov.axis("off")
ov.plot(whole[:, 0], whole[:, 1], color="#C9CED6", lw=1.4)
ov.plot(truth[:, 0], truth[:, 1], color=ISRO, lw=4)
ov.scatter(*whole[0], s=30, color=INK, zorder=5)
ov.text(0.5, -0.02, f"the whole {tr.xy.shape[0] / tr.hz / 60:.0f} min drive,\nblackout in orange", transform=ov.transAxes,
        ha="center", va="top", fontsize=10.5, color=MUTED)

ax = fig.add_axes([0.29, 0.08, 0.68, 0.76]); ax.set_aspect("equal"); ax.axis("off")
ax.plot(road[:, 0], road[:, 1], color=OSM, lw=10, alpha=0.18, solid_capstyle="round", label="OpenStreetMap road")
ax.plot(truth[:, 0], truth[:, 1], color=INK, lw=2.0, ls=(0, (4, 2)), label="where the car really went (GPS, hidden)")
ax.plot(free[:, 0], free[:, 1], color=NOMAP, lw=2.2, label="no map: AI speed + gyro heading")
ax.plot(cal[:, 0], cal[:, 1], color=DHRUVA, lw=2.6, ls=(0, (1, 1.2)), label="Dhruva: AI speed on the road")
ax.scatter(*truth[0], s=190, facecolor="white", edgecolor=ISRO, lw=3, zorder=6)
ax.annotate("GPS lost here\n(dot freezes here on a phone map)", truth[0], xytext=(20, 0), textcoords="offset points",
            fontsize=10.5, color="#9A6418", fontweight="demibold", ha="left", va="center")
ax.scatter(*truth[-1], s=90, marker="s", color=INK, zorder=6)
ax.annotate("true position\n1 km later", truth[-1], xytext=(0, 12), textcoords="offset points", fontsize=10.5, color=INK, ha="center", va="bottom")
e_free, e_cal = np.linalg.norm(free[-1] - truth[-1]), np.linalg.norm(cal[-1] - truth[-1])
ax.scatter(*free[-1], s=80, color=NOMAP, zorder=6)
ax.annotate(f"no map: {e_free:.0f} m off", free[-1], xytext=(12, -2), textcoords="offset points", fontsize=11,
            color=NOMAP, fontweight="demibold", ha="left", va="center")
ax.scatter(*cal[-1], s=80, color=DHRUVA, zorder=6)
ax.annotate(f"Dhruva: {e_cal:.0f} m off ({100 * e_cal / r.dist_m:.0f}%)", cal[-1], xytext=(-14, -2), textcoords="offset points",
            fontsize=11, color=DHRUVA, fontweight="demibold", ha="right", va="center")
fig.legend(*ax.get_legend_handles_labels(), loc="upper left", bbox_to_anchor=(0.30, 0.78), frameon=False, fontsize=10.5)
fig.text(0.31, 0.50, f"a frozen dot would be\n{r.err_frozen:.0f} m off", fontsize=11, color=FROZEN, fontweight="demibold", va="top")
fig.savefig(f"{OUT}/iovnbd_position_plot.png"); plt.close(fig)
print(f"position plot: {r.drive} {int(r.i0)}-{int(r.i1)}, {r.dist_m:.0f} m at {r.speed_kmh:.0f} km/h, "
      f"frozen {r.err_frozen:.0f} m, no map {e_free:.0f} m, Dhruva cal {e_cal:.0f} m (csv {r['err_AI cal + road']:.0f} m)")

# ---------- 2. summary over every blackout ----------
methods = ["frozen", "AI + gyro", "AI + road", "AI cal + road", "perfect + road"]
fig = plt.figure(figsize=(13, 7.3))
title(fig, "IO-VNBD, 5 held-out drives: the AI model keeps the dot on the road",
      f"{len(D)} GPS blackouts of 0.5, 1 and 2 km, cut every 250 m along drives the model never trained on. "
      "Lower is better.")
bx = fig.add_axes([0.07, 0.13, 0.55, 0.60])
w = 0.16
for j, m in enumerate(methods):
    vals = [D[D.target_m == t][f"pct_{m}"].median() for t in (500.0, 1000.0, 2000.0)]
    xs = np.arange(3) + (j - 2) * w
    bx.bar(xs, vals, width=w * 0.92, color=COLORS[m], zorder=2, label=NAMES[m].replace("\n", " "))
    for x, v in zip(xs, vals):
        bx.text(x, v + 1.2, f"{v:.0f}" if v >= 10 else f"{v:.1f}", ha="center", fontsize=9.5, color=INK)
bx.set_xticks(range(3), [f"GPS off {int(t)} m\n({(D.target_m == t).sum()} blackouts)" for t in (500.0, 1000.0, 2000.0)])
bx.set_ylabel("drift at the end of the blackout, % of distance (median)")
bx.set_ylim(0, 105); bx.grid(axis="y", color=GRID, zorder=0)
bx.axhline(10, color=ISRO, lw=1.6, ls=(0, (5, 3)), zorder=3)
bx.set_xlim(-0.5, 3.05); bx.text(3.03, 10.8, "ISRO\nlimit 10%", color=ISRO, fontsize=10, ha="right", va="bottom")
bx.legend(loc="lower left", bbox_to_anchor=(0.0, 1.02), ncol=3, frameon=False, fontsize=10)

k = fig.add_axes([0.68, 0.13, 0.29, 0.66]); k.axis("off")
F = D[(D.target_m == 1000.0) & (D.speed_kmh >= 50)]
k.text(0, 1.0, "The PS example: 1 km of GPS loss\nat 60 km/h", fontsize=14, fontweight="demibold", color=INK, va="top")
k.text(0, 0.86, f"{len(F)} one-km blackouts at 50 km/h or more\n(median {F.speed_kmh.median():.0f} km/h)",
       fontsize=10.5, color=MUTED, va="top")
rows = [("frozen", "GPS frozen"), ("AI + gyro", "no map"), ("AI + road", "Dhruva"), ("AI cal + road", "Dhruva, calibrated")]
k.text(0.62, 0.72, "median\nerror", fontsize=10, color=MUTED, ha="right", va="bottom")
k.text(1.0, 0.72, "under\n100 m", fontsize=10, color=MUTED, ha="right", va="bottom")
for i, (m, lab) in enumerate(rows):
    y = 0.64 - i * 0.12
    k.text(0, y, lab, fontsize=12.5, color=COLORS[m], fontweight="demibold", va="center")
    k.text(0.62, y, f"{F[f'err_{m}'].median():.0f} m", fontsize=14, color=INK, ha="right", va="center")
    k.text(1.0, y, f"{int((F[f'err_{m}'] < 100).sum())}/{len(F)}", fontsize=12, color=INK, ha="right", va="center")
fig.savefig(f"{OUT}/iovnbd_summary.png"); plt.close(fig)
print("wrote", f"{OUT}/iovnbd_position_plot.png", f"{OUT}/iovnbd_summary.png")
