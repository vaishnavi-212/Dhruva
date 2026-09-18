"""Slide-ready charts: GPS on at the start, off mid-journey, and the self-healing map.
Measured numbers only, except the one panel explicitly labelled PROJECTION."""
import json, os, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(HERE, ".."))                # results/selfheal
os.makedirs(OUT, exist_ok=True)
INK, NOMAP, LEARN, OSM, ISRO, GRID, MUTED = "#1F2A37", "#C8553D", "#1F8A7A", "#8D99AE", "#D9922E", "#E6E9EE", "#5B6573"
plt.rcParams.update({"font.family": "Avenir Next", "font.size": 12, "axes.edgecolor": "#C9CED6", "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": "white", "axes.facecolor": "white", "savefig.dpi": 200})
R = json.load(open(os.path.join(HERE, "selfheal_results.json")))["results"]
BENCH = json.load(open(os.path.join(HERE, "turns_results.json")))
C = np.load(os.path.join(HERE, "curves.npz"))
CUT = json.load(open(os.path.join(HERE, "cut_results.json")))
CP = np.load(os.path.join(HERE, "cut_paths.npz"))
SRC = {"11 Sep 13:46": "the 1:46 pm lap", "11 Sep 19:37": "the 7:40 ride"}


def title(fig, main, sub):
    fig.text(0.04, 0.955, main, fontsize=19, fontweight="demibold", color=INK, va="top")
    fig.text(0.04, 0.895, sub, fontsize=12, color=MUTED, va="top")


def isro(ax, y=10):
    ax.axhline(y, color=ISRO, lw=1.6, ls=(0, (5, 3)), zorder=1)
    ax.text(ax.get_xlim()[1], y, "  ISRO limit 10%", color=ISRO, fontsize=10, va="bottom", ha="right")


def save(fig, name):
    fig.savefig(os.path.join(OUT, name)); plt.close(fig); print("wrote", name)


# ---------- 1. the main demo ride: GPS on, then off mid-journey ----------
st = "19-41-28"; rc = CUT[st]
fig = plt.figure(figsize=(13, 7.3))
title(fig, "Start with GPS on. Lose it mid-journey. Dhruva keeps navigating.",
      f"7:44 ride: GPS on for the first {rc['gps_on_m']:.0f} m, off for the last {rc['gps_off_m']:.0f} m. "
      "The road ahead was learned from the 7:40 ride, ridden the other way.")
ax = fig.add_axes([0.04, 0.08, 0.50, 0.76]); ax.set_aspect("equal"); ax.axis("off")
tt, tr, pm = CP[f"{st}|t"], CP[f"{st}|truth"], CP[f"{st}|prevmap"]
on = tt <= rc["cut_s"]; off = (tt >= rc["cut_s"]) & (tt <= rc["finish_s"])
nm, lv = CP[f"{st}|none"], CP[f"{st}|learned"]
cut, fin = CP[f"{st}|cut_xy"], CP[f"{st}|fin_xy"]
ax.plot(pm[:, 0], pm[:, 1], color=LEARN, lw=9, alpha=0.16, solid_capstyle="round", label="road learned from the 7:40 ride's GPS")
ax.plot(tr[on, 0], tr[on, 1], color=INK, lw=2.6, label="GPS on")
ax.plot(tr[off, 0], tr[off, 1], color=INK, lw=1.3, ls=(0, (4, 2)), label="GPS off: where the rider really went")
ax.plot(nm[:, 0], nm[:, 1], color=NOMAP, lw=2.2, label="no map: gyro heading + speed")
ax.plot(lv[:, 0], lv[:, 1], color=LEARN, lw=2.4, ls=(0, (1, 1.4)), label="Dhruva on the learned road")
ax.scatter(*tr[0], s=70, color=INK, zorder=5); ax.annotate("start, GPS on", tr[0], xytext=(10, -14), textcoords="offset points", fontsize=10, color=INK)
ax.scatter(*cut, s=190, facecolor="white", edgecolor=ISRO, lw=3, zorder=6)
ax.annotate("GPS switched off", cut, xytext=(0, 16), textcoords="offset points", fontsize=10.5, color="#9A6418", fontweight="demibold", ha="center")
ax.scatter(*fin, s=90, marker="s", color=INK, zorder=5); ax.annotate("rider stops", fin, xytext=(-12, -4), textcoords="offset points", fontsize=10, color=INK, ha="right", va="top")
en = float(np.linalg.norm(CP[f"{st}|none_end"] - fin)); ep = float(np.linalg.norm(CP[f"{st}|learned_end"] - fin))
ax.scatter(*CP[f"{st}|none_end"], s=80, color=NOMAP, zorder=5)
ax.annotate(f"no map ends {en:.0f} m away", CP[f"{st}|none_end"], xytext=(-6, -14), textcoords="offset points", fontsize=10.5, color=NOMAP, fontweight="demibold", ha="right", va="top")
ax.scatter(*CP[f"{st}|learned_end"], s=80, color=LEARN, zorder=5)
ax.annotate(f"Dhruva ends {ep:.0f} m away", fin, xytext=(22, 10), textcoords="offset points", fontsize=10.5, color=LEARN, fontweight="demibold", ha="left", va="bottom")
ax.legend(loc="lower left", frameon=False, fontsize=10, bbox_to_anchor=(0.0, -0.08))
bx = fig.add_axes([0.63, 0.16, 0.33, 0.62])
vals = [rc["none"]["median"], rc["learned"]["median"], rc["osm"]["median"]]
labels = ["no map", f"road learned from\nthe 7:40 ride\n{rc['learned']['passes']}/11 models pass", f"OpenStreetMap\n{rc['osm']['passes']}/11 models pass"]
bx.bar(range(3), vals, color=[NOMAP, LEARN, OSM], width=0.62, zorder=2)
bx.set_xticks(range(3), labels, fontsize=10.5); bx.set_ylabel("drift after GPS off, % (11-model median)")
bx.set_ylim(0, 40); bx.grid(axis="y", color=GRID, zorder=0); isro(bx)
for i, v in enumerate(vals):
    bx.text(i, v + 1, f"{v:.1f}%", ha="center", fontsize=15, fontweight="demibold", color=INK)
save(fig, "heal_1_one_ride_teaches_the_next.png")

# ---------- 2. both evening demo rides, GPS off mid-journey ----------
order = ["19-37-43", "19-41-28"]
fig = plt.figure(figsize=(13, 7.3))
title(fig, "Both demo rides, GPS off mid-journey: every model inside ISRO's limit on the road",
      "11 Sept evening. Scored from the moment the rider switched GPS off to where they stopped. "
      "Right: speed replaced by the truth, leaving the map alone.")
names = [f"{CUT[s]['ride']}\nGPS off for {CUT[s]['gps_off_m']:.0f} m\nroad from {SRC[CUT[s]['learned_from']]}" for s in order]
for pi, key in enumerate(["median", "perfect_speed"]):
    ax = fig.add_axes([0.06 + pi * 0.49, 0.16, 0.42, 0.57])
    x = np.arange(len(order)); w = 0.26
    for si, (state, col, lab) in enumerate([("none", NOMAP, "no map"), ("learned", LEARN, "road learned from an earlier ride"),
                                           ("osm", OSM, "OpenStreetMap")]):
        vals = [CUT[s][state][key] for s in order]
        ax.bar(x + (si - 1) * w, vals, w * 0.92, color=col, label=lab, zorder=2)
        for xi, s, v in zip(x, order, vals):
            ax.text(xi + (si - 1) * w, v + 0.8, f"{v:.1f}%", ha="center", fontsize=10.5, color=INK, fontweight="demibold")
            if pi == 0:
                ax.text(xi + (si - 1) * w, v + 3.0, f"{CUT[s][state]['passes']}/11", ha="center", fontsize=9.5, color=col)
    ax.set_xlim(-0.55, 1.55); ax.set_ylim(0, 42)
    ax.set_xticks(x, names, fontsize=10); ax.set_ylabel(["drift % (11-model median)", "drift % with perfect speed"][pi])
    ax.grid(axis="y", color=GRID, zorder=0); isro(ax)
    ax.set_title(["The real system", "The map alone"][pi], loc="left", fontsize=13, color=INK, fontweight="demibold")
    if pi == 0:
        h, l = ax.get_legend_handles_labels()
        fig.legend(h, l, loc="upper left", bbox_to_anchor=(0.04, 0.855), ncol=3, frameon=False, fontsize=11)
save(fig, "heal_2_all_four_rides.png")

# ---------- 3. more rides: a more dependable learned road ----------
KS = [1, 2, 3, 5, 8, 11]
med = [np.median([r["learned"][f"any_{k}"]["oracle_drift"] for r in R]) for k in KS]
allv = [np.concatenate([r["learned"][f"any_{k}"]["oracle_drift_all"] for r in R]) for k in KS]
p90 = [np.percentile(a, 90) for a in allv]; worst = [a.max() for a in allv]
osm_med = np.median([r["osm"]["oracle_drift"] for r in R])
fig = plt.figure(figsize=(13, 7.3))
title(fig, "Every extra ride makes the learned road more dependable",
      "Map error alone (perfect speed), 12 passes of the surveyed campus path, each scored on a map built only from OTHER passes.")
ax = fig.add_axes([0.07, 0.12, 0.62, 0.68])
ax.fill_between(KS, med, p90, color=LEARN, alpha=0.12, lw=0, label="middle-to-bad maps")
ax.plot(KS, worst, color=LEARN, lw=1.1, alpha=0.45, ls=(0, (3, 2)), marker="o", ms=4, label="single worst map (fewer maps at 11)")
ax.plot(KS, p90, color=LEARN, lw=2, marker="o", ms=6, label="bad maps: 90th percentile")
ax.plot(KS, med, color=LEARN, lw=3, marker="o", ms=8, label="median map")
ax.axhline(osm_med, color=OSM, lw=1.6, ls="--"); ax.text(6.5, osm_med - 0.12, "OpenStreetMap, median", color=OSM, fontsize=10, va="top", ha="center")
for k, v in zip(KS, p90):
    ax.text(k, v + 0.22, f"{v:.1f}%", ha="center", fontsize=10, color=LEARN, fontweight="demibold")
ax.text(KS[0] - 0.15, med[0] + 0.22, f"{med[0]:.1f}%", ha="center", fontsize=10, color=LEARN)
ax.text(KS[-1], med[-1] + 0.22, f"{med[-1]:.1f}%", ha="center", fontsize=10, color=LEARN)
ax.set_xticks(KS); ax.set_xlabel("earlier GPS rides fused into the map"); ax.set_ylabel("drift % with perfect speed")
ax.set_ylim(0, 8); ax.set_xlim(0.5, 12.3); ax.grid(axis="y", color=GRID); ax.legend(frameon=False, fontsize=10, loc="upper right")
nx = fig.add_axes([0.73, 0.12, 0.25, 0.68]); nx.axis("off")
gap = max(m - osm_med for m in med[1:])
notes = [("Bad maps (90th pct)", f"{p90[0]:.1f}% to {p90[-1]:.1f}%", "one ride can be a bad one;\nfusing rides removes it"),
         ("Median map", f"{med[0]:.2f}% to {med[-1]:.2f}%", f"within {gap:.1f} points of\nOpenStreetMap from 2 rides on"),
         ("Real rides", "12 passes", "both directions\n30 Aug to 11 Sept")]
for i, (h, big, small) in enumerate(notes):
    y = 0.97 - i * 0.34
    nx.text(0, y, h.upper(), fontsize=9.5, color=MUTED, va="top", fontweight="demibold")
    nx.text(0, y - 0.06, big, fontsize=17, color=INK, va="top", fontweight="demibold")
    nx.text(0, y - 0.15, small, fontsize=10, color=MUTED, va="top", linespacing=1.3)
save(fig, "heal_3_more_rides_dependable.png")

# ---------- 4. without the road vs with the road, every benchmark ride ----------
B = sorted(BENCH, key=lambda b: b["none_drift"])
fig = plt.figure(figsize=(13, 7.3))
nmed = np.median([b["none_drift"] for b in B]); rmed = np.median([b["road_drift"] for b in B])
title(fig, f"Even at its best, no map drifts {nmed:.0f}%. With the road: {rmed:.1f}%.",
      "The 11 benchmark rides, each scored with the model that never saw it. Same IMU, same speed. Only the heading source changes.")
ax = fig.add_axes([0.24, 0.10, 0.70, 0.72])
for i, b in enumerate(B):
    ax.plot([b["road_drift"], b["none_drift"]], [i, i], color="#D5DAE1", lw=2.5, zorder=1)
    ax.scatter(b["none_drift"], i, s=90, color=NOMAP, zorder=3); ax.scatter(b["road_drift"], i, s=90, color=LEARN, zorder=3)
    ax.text(b["none_drift"] + 2.5, i, f"{b['none_drift']:.0f}%", va="center", fontsize=10, color=NOMAP)
    ax.text(b["road_drift"] - 1.5, i, f"{b['road_drift']:.1f}%", va="center", ha="right", fontsize=10, color=LEARN)
lab = []
for b in B:
    day = "30 Aug" if "08-30" in b["ride"] else "31 Aug"
    extra = " · turn-heavy route" if b["dist_m"] > 2000 else ""
    lab.append(f"{day} {b['ride'][-8:-3].replace('-', ':')} · {b['dist_m']:.0f} m{extra}")
ax.set_yticks(range(len(B)), lab, fontsize=10); ax.set_xlim(-8, 150); ax.set_xlabel("drift, % of distance without GNSS")
ax.axvline(10, color=ISRO, lw=1.6, ls=(0, (5, 3))); ax.text(10.8, len(B) - 0.4, "ISRO limit 10%", color=ISRO, fontsize=10)
ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0); ax.grid(axis="x", color=GRID, zorder=0)
ax.scatter([], [], s=90, color=NOMAP, label=f"no map (gyro heading, best-case bias): median {nmed:.0f}%, {sum(b['none_drift']<10 for b in B)}/11 pass")
ax.scatter([], [], s=90, color=LEARN, label=f"heading from the road: median {rmed:.1f}%, {sum(b['road_drift']<10 for b in B)}/11 pass")
ax.legend(frameon=False, fontsize=10.5, loc="lower right")
save(fig, "heal_4_without_vs_with_road.png")

# ---------- 5. error keeps growing without the road; projection clearly marked ----------
fig = plt.figure(figsize=(13, 7.3))
title(fig, "Without the road the error runs away; on the road it stays small",
      "Error in metres against distance ridden since GNSS was lost. 11 benchmark rides, measured.")
ax = fig.add_axes([0.07, 0.12, 0.55, 0.68])
grid = np.arange(0, 820, 10); curves = {"none": [], "road": []}
rides = sorted({k.split("|")[1] for k in C.files if k.startswith("bench|")})
for rd in rides:
    d = C[f"bench|{rd}|dist"]; u = np.r_[True, np.diff(d) > 0]
    for key, col in (("none", NOMAP), ("road", LEARN)):
        e = C[f"bench|{rd}|{key}"]
        ax.plot(d, e, color=col, lw=0.9, alpha=0.28)
        curves[key].append(np.interp(grid, d[u], e[u]))
for key, col, lab in (("none", NOMAP, "no map"), ("road", LEARN, "on the road")):
    m = np.median(curves[key], axis=0); ax.plot(grid, m, color=col, lw=3.2, label=f"{lab}, median")
    ax.text(grid[-1] + 15, m[-1], f"{m[-1]:.0f} m", color=col, fontsize=12, fontweight="demibold", va="center")
ax.set_xlim(0, 2400); ax.set_ylim(0, 1300); ax.set_xlabel("metres ridden without GNSS"); ax.set_ylabel("distance from true position, m")
ax.grid(color=GRID); ax.legend(frameon=False, loc="upper left", fontsize=11)
ax.text(1500, 150, "the 2.3 km turn-heavy ride\ncontinues past 800 m", fontsize=9.5, color=MUTED)
px = fig.add_axes([0.70, 0.12, 0.27, 0.68])
px.set_facecolor("#FAF6EF")
for s in px.spines.values(): s.set_visible(False)
km = 5.0; vn, vr = nmed / 100 * km * 1000, rmed / 100 * km * 1000
px.bar([0, 1], [vn, vr], color=[NOMAP, LEARN], width=0.6, hatch="//", edgecolor="white", lw=0)
px.text(0, vn + 60, f"≈{vn/1000:.1f} km off", ha="center", fontsize=13, fontweight="demibold", color=NOMAP)
px.text(1, vr + 60, f"≈{round(vr, -1):.0f} m\nstill on the road", ha="center", fontsize=12, fontweight="demibold", color=LEARN)
px.set_xticks([0, 1], ["no map", "Dhruva"]); px.set_ylim(0, 4300); px.set_yticks([])
px.set_title("PROJECTION, NOT MEASURED\nA 5 km stretch with no GNSS, if the\nmeasured median drift rates held", fontsize=10.5, color="#8A5A1E", loc="left", fontweight="demibold")
save(fig, "heal_5_error_grows_projection.png")
