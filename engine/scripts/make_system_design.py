#!/usr/bin/env python3
"""One-slide system design: what runs on the phone, what runs in the cloud, and how the map heals at scale.

Every number on it is measured (RESOURCES 8av): the phone model and perf log, the 7:44 ride, the consensus
experiment (scripts/eval_consensus_map.py), the upload size of a learned road and the raw recording rate.

    python scripts/make_system_design.py   ->  results/system_design.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

INK, TEAL, AMBER, GREY, MUTED, NAVY = "#1F2A37", "#1F8A7A", "#D9922E", "#8D99AE", "#5B6573", "#1F4E79"
LTEAL, LAMBER, LBLUE, LGREY = "#E4F2EF", "#FBF1E1", "#EAF2FA", "#F2F4F7"
plt.rcParams.update({"font.family": "Avenir Next", "savefig.dpi": 200})

fig = plt.figure(figsize=(13, 7.3)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 56.15); ax.axis("off")
fig.text(0.04, 0.955, "Dhruva at scale: navigation on the phone, learning in the cloud", fontsize=20, fontweight="demibold",
         color=INK, va="top")
fig.text(0.04, 0.895, "Every phone navigates by itself, offline. The cloud only improves the shared map and the speed model, "
         "from small opt-in uploads.", fontsize=12, color=MUTED, va="top")


def box(x, y, w, h, fill, edge=None, dashed=False, r=1.0, lw=1.4):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fill,
                                ec=edge or fill, lw=lw, ls=(0, (4, 3)) if dashed else "-"))


def arrow(x0, y0, x1, y1, color=INK, lw=2.0):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, mutation_scale=15))


# ------------------------------------------------------------------ PHONE (edge)
box(2.5, 13.2, 55, 34.5, LTEAL)
ax.text(4.2, 45.6, "ON THE PHONE — works with no network", fontsize=11, fontweight="demibold", color=TEAL, va="center")
steps = ["Sensors\n125 Hz", "AI speed\n15 ms / run", "Fusion +\nroad binding", "Offline route\n+ voice", "Wrong-turn\ncheck"]
for i, s in enumerate(steps):
    x = 4.2 + i * 10.6
    box(x, 36.8, 9.4, 6.6, "white", edge="#A9D8CF")
    ax.text(x + 4.7, 40.1, s, fontsize=9.6, color=INK, ha="center", va="center", linespacing=1.2)
    if i < 4:
        arrow(x + 9.5, 40.1, x + 10.5, 40.1, color=TEAL, lw=1.5)

ax.text(4.2, 33.8, "The map the phone navigates on, looked up top to bottom:", fontsize=10, color=MUTED, va="center")
layers = [("Personal", "your own rides — improves after every ride, stays on your phone", "#CDEBE4"),
          ("Shared", "roads 3+ contributions agree on — downloaded with the city pack", "#DCEAF5"),
          ("Base", "OpenStreetMap (open data, ODbL)", LGREY)]
for i, (name, desc, col) in enumerate(layers):
    y = 27.5 - i * 5.6
    box(4.2, y, 51.4, 4.7, col)
    ax.text(6.0, y + 2.35, name, fontsize=11, fontweight="demibold", color=INK, va="center")
    ax.text(16.0, y + 2.35, desc, fontsize=9.8, color=INK, va="center")
ax.text(4.2, 14.4, "Vehicle unit (optional): the same engine on a 200 Hz IMU box, for buses, ambulances and convoys",
        fontsize=9.2, color=MUTED, va="center", style="italic")

# ------------------------------------------------------------------ CLOUD
box(70.5, 13.2, 27, 34.5, LBLUE)
ax.text(72.2, 45.6, "IN THE CLOUD — optional, batch", fontsize=11, fontweight="demibold", color=NAVY, va="center")
cloud = [("Map merge", "a road is promoted only when 3+\ncontributions agree (median)"),
         ("Model training", "retrain speed on many phones;\nship only if it beats the benchmark"),
         ("Region packs", "versioned city packs + models,\nstaged rollout, rollback")]
for i, (name, desc) in enumerate(cloud):
    y = 35.4 - i * 9.4
    box(72.2, y, 23.6, 8.0, "white", edge="#B7CFE6")
    ax.text(73.6, y + 5.9, name, fontsize=11, fontweight="demibold", color=NAVY, va="center")
    ax.text(73.6, y + 2.8, desc, fontsize=9.1, color=INK, va="center", linespacing=1.2)

# ------------------------------------------------------------------ sync arrows
arrow(57.8, 33.0, 70.2, 33.0, color=AMBER, lw=2.2)
ax.text(64.0, 36.9, "UP  (opt-in, wifi)", fontsize=9.5, fontweight="demibold", color="#9A6418", ha="center", va="center")
ax.text(64.0, 34.6, "road evidence 0.21 KB/km", fontsize=9.2, color=INK, ha="center", va="center")
ax.text(64.0, 30.9, "no trips, no identity,\nends trimmed", fontsize=8.6, color=MUTED, ha="center", va="center", linespacing=1.15)
arrow(70.2, 24.0, 57.8, 24.0, color=NAVY, lw=2.2)
ax.text(64.0, 27.2, "DOWN", fontsize=9.5, fontweight="demibold", color=NAVY, ha="center", va="center")
ax.text(64.0, 21.6, "city pack ~2 MB,\nmodel updates", fontsize=9.2, color=INK, ha="center", va="center", linespacing=1.15)

# ------------------------------------------------------------------ evidence strip
facts = [("0", "server calls to navigate:\ncloud cost does not grow with users"),
         ("33% $\\rightarrow$ 3.5%", "drift, once one earlier ride\nhas learned the road (7:44 ride)"),
         ("10.4% $\\rightarrow$ 2.2%", "a faulty contribution, alone vs\ninside a 3-way consensus"),
         ("0.21 KB / km", "uploaded, against 1.76 MB per\nminute of raw sensor recording")]
for i, (big, small) in enumerate(facts):
    x = 2.5 + i * 24.1
    box(x, 1.6, 23.0, 10.0, "white", edge="#D5DBE3")
    ax.text(x + 1.4, 8.5, big, fontsize=16, fontweight="demibold", color=TEAL if i != 2 else "#B2472F", va="center")
    ax.text(x + 1.4, 4.4, small, fontsize=9.2, color=INK, va="center", linespacing=1.2)

fig.savefig("results/system_design.png"); print("wrote results/system_design.png")
