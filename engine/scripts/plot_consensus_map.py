#!/usr/bin/env python3
"""Slide chart for the shared-map consensus rule (reads results/consensus_map.csv).

Left: one real example on the campus corridor -- two clean contributions, one with a simulated 40 m GPS
fault, what naive averaging makes of them, and what the median consensus makes of them, against the
new rider's own GPS. Right: every held-out ride, every draw.

    python scripts/eval_consensus_map.py && python scripts/plot_consensus_map.py
"""
from __future__ import annotations
import glob, os, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import eval_consensus_map as E

INK, BAD, DHRUVA, AVG, GREY, ISRO, GRID, MUTED = ("#1F2A37", "#C8553D", "#1F8A7A", "#D9922E", "#8D99AE",
                                                  "#D9922E", "#E6E9EE", "#5B6573")
plt.rcParams.update({"font.family": "Avenir Next", "font.size": 12, "axes.edgecolor": "#C9CED6", "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": "white", "axes.facecolor": "white", "savefig.dpi": 200})

D = pd.read_csv("results/consensus_map.csv")
folders = sorted(glob.glob("rides/bike/*/")) + sorted(glob.glob("rides/aug31/*/")) + sorted(glob.glob("rides/app_DhruvaRun_2026-09-1*/"))
rides = {os.path.basename(f.rstrip("/")): t for f in folders if (t := E.load_corridor(f)) is not None}
names = list(rides)

fig = plt.figure(figsize=(13, 7.3))
fig.text(0.04, 0.955, "Shared map: a road enters only when 3 or more contributions agree", fontsize=19,
         fontweight="demibold", color=INK, va="top")
fig.text(0.04, 0.895, f"{len(names)} campus rides, back gate to LHC. Each ride in turn is the new rider; the map is built "
         "from the others. One contribution carries a simulated 40 m GPS fault.", fontsize=12, color=MUTED, va="top")

# ---- left: one example, drawn on the ground
ax = fig.add_axes([0.02, 0.06, 0.40, 0.78]); ax.set_aspect("equal"); ax.axis("off")
rng = np.random.default_rng(7)
rider = rides[names[2]]; others = [rides[n] for n in names if n != names[2]]
g1, g2, g3 = others[0], others[4], others[8]
bad = E.with_fault(g3, rng)
avg = E.consensus([g1, g2, bad], "mean"); med = E.consensus([g1, g2, bad], "median")
ax.plot(rider[:, 0], rider[:, 1], color=INK, lw=1.4, ls=(0, (4, 2)), label="new rider's real track (GPS, hidden)")
for g in (g1, g2):
    ax.plot(g[:, 0], g[:, 1], color=GREY, lw=1.0, alpha=0.8)
ax.plot([], [], color=GREY, lw=1.0, label="two clean contributions")
ax.plot(bad[:, 0], bad[:, 1], color=BAD, lw=2.0, label="one contribution with a 40 m GPS fault")
ax.plot(avg[:, 0], avg[:, 1], color=AVG, lw=2.4, label="naive merge: average of the three")
ax.plot(med[:, 0], med[:, 1], color=DHRUVA, lw=3.2, alpha=0.85, label="Dhruva: median of the three")
k = int(np.argmax(np.linalg.norm(bad - g3, axis=1)))
ax.annotate("40 m GPS fault", bad[k], xytext=(-16, 0), textcoords="offset points", color=BAD, fontsize=11,
            fontweight="demibold", va="center", ha="right")
ax.legend(loc="lower left", frameon=False, fontsize=9.5, bbox_to_anchor=(0.0, -0.04))

# ---- right: all held-out rides
bx = fig.add_axes([0.52, 0.19, 0.45, 0.60])
order = [("1 contributor", "1 clean\nride", GREY), ("3 contributors", "3 clean\nrides", GREY),
         ("all others, median", "12 clean\nrides", GREY),
         ("faulty contributor alone", "the faulty\nride alone", BAD),
         ("3 incl. faulty, averaged", "3 incl. faulty:\naveraged", AVG),
         ("3 incl. faulty, median", "3 incl. faulty:\nmedian (Dhruva)", DHRUVA)]
xs = [0, 1, 2, 3.4, 4.4, 5.4]
for x, (key, lab, col) in zip(xs, order):
    X = D[D.scenario == key]
    v = X.drift_pct.median(); p95 = X.shape_p95_m.median()
    bx.bar(x, v, width=0.72, color=col, zorder=2)
    bx.text(x, v + 0.95, f"{v:.1f}%", ha="center", fontsize=13, fontweight="demibold", color=INK)
    bx.text(x, v + 0.3, f"road bent {p95:.0f} m", ha="center", fontsize=8.5, color=MUTED)
bx.set_xticks(xs, [o[1] for o in order], fontsize=9.5)
bx.set_ylabel("drift from the map alone, % (median)")
bx.set_ylim(0, 13); bx.set_xlim(-0.6, 6.0); bx.grid(axis="y", color=GRID, zorder=0)
bx.text(1.0, 12.4, "clean contributions", ha="center", fontsize=10.5, color=MUTED)
bx.text(4.4, 12.4, "one faulty contribution", ha="center", fontsize=10.5, color=MUTED)
bx.axvline(2.7, color="#C9CED6", lw=1)
fig.text(0.52, 0.03, "Drift: the new rider's trip ridden along the map at their true speed, worst error at\n"
         "300/450/600 m. Road bent: 95th-percentile distance of the rider's track from the map.\n"
         "Contributions are separate rides recorded on one phone; the fault is injected.", fontsize=8.5, color=MUTED)
fig.savefig("results/consensus_map.png"); print("wrote results/consensus_map.png")
