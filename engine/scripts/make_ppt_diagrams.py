#!/usr/bin/env python3
"""The five system diagrams for the finale PPT, one consistent visual language.

  teal  = on the phone     navy  = in the cloud     amber = GPS lost     grey = inputs and stored data

  1_low_level_design.png     inside the phone during the ride: sensors -> AI speed -> position -> guidance
  2_high_level_design.png    phone and cloud: before, during and after the ride
  3_data_flow.png            data flow diagram: who produces which data, where it goes, where it is kept
  4_user_journey.png         what the rider does and sees, end to end
  5_full_architecture.png    every component, on the phone, in the cloud, and where the AI model comes from

    python scripts/make_ppt_diagrams.py   ->  results/diagrams/
"""
from __future__ import annotations
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon

INK, MUTED = "#1F2A37", "#5B6573"
TEAL, LTEAL, TEDGE = "#1F8A7A", "#E4F2EF", "#A9D8CF"
NAVY, LNAVY, NEDGE = "#1F4E79", "#EAF2FA", "#B7CFE6"
AMBER, LAMBER, AEDGE = "#C98A2E", "#FBF1E1", "#E8C790"
GREY, LGREY, GEDGE = "#8D99AE", "#F2F4F7", "#D5DBE3"
PURPLE, LPURPLE, PEDGE = "#6C4F9E", "#F1ECF8", "#CFC2E6"
plt.rcParams.update({"font.family": "Avenir Next", "savefig.dpi": 200})
OUT = "results/diagrams"
os.makedirs(OUT, exist_ok=True)


def canvas(title, subtitle):
    fig = plt.figure(figsize=(13, 7.3)); ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100); ax.set_ylim(0, 56.15); ax.axis("off")
    fig.text(0.035, 0.955, title, fontsize=21, fontweight="demibold", color=INK, va="top")
    fig.text(0.035, 0.893, subtitle, fontsize=12, color=MUTED, va="top")
    return fig, ax


def panel(ax, x, y, w, h, fill, label=None, color=INK, r=1.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fill, ec=fill, lw=0))
    if label:
        ax.text(x + 1.5, y + h - 1.9, label, fontsize=11, fontweight="demibold", color=color, va="center")


def box(ax, x, y, w, h, title, body=None, fill="white", edge=GEDGE, tcolor=INK, ts=11.5, bs=9.6, r=0.9, lw=1.4):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fill, ec=edge, lw=lw))
    if body is None:
        ax.text(x + w / 2, y + h / 2, title, fontsize=ts, fontweight="demibold", color=tcolor, ha="center", va="center",
                linespacing=1.2)
    else:
        ax.text(x + w / 2, y + h - 2.0, title, fontsize=ts, fontweight="demibold", color=tcolor, ha="center", va="center")
        ax.text(x + w / 2, y + (h - 3.2) / 2, body, fontsize=bs, color=INK, ha="center", va="center", linespacing=1.3)


def arrow(ax, x0, y0, x1, y1, color=INK, lw=1.8, label=None, lx=None, ly=None, lc=None, ls=9.2, ha="center"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, mutation_scale=15, shrinkA=0, shrinkB=0))
    if label:
        ax.text(lx if lx is not None else (x0 + x1) / 2, ly if ly is not None else (y0 + y1) / 2 + 1.0, label,
                fontsize=ls, color=lc or color, ha=ha, va="center", linespacing=1.2)


def path(ax, pts, color=INK, lw=1.8, label=None, lx=None, ly=None, ls=9.2, ha="center"):
    """An elbow arrow through a list of points; the arrowhead is on the last point."""
    xs, ys = zip(*pts)
    ax.plot(xs[:-1], ys[:-1], color=color, lw=lw, solid_capstyle="round")
    arrow(ax, xs[-2], ys[-2], xs[-1], ys[-1], color=color, lw=lw)
    if label:
        ax.text(lx, ly, label, fontsize=ls, color=color, ha=ha, va="center", linespacing=1.2)


def step(ax, x, y, n, color=INK, r=1.35, fs=10.5):
    ax.add_patch(plt.Circle((x, y), r, fc=color, ec="white", lw=1.6, zorder=6))
    ax.text(x, y, str(n), fontsize=fs, fontweight="demibold", color="white", ha="center", va="center", zorder=7)


def save(fig, name):
    fig.savefig(os.path.join(OUT, name)); plt.close(fig); print("wrote", name)


# =========================================================== 1. LOW-LEVEL: inside the phone during the ride
fig, ax = canvas("Inside the phone: how Dhruva finds you and guides you",
                 "Everything here runs on the phone, with no internet. When GPS is lost, the same pipeline keeps "
                 "navigating on the motion sensors and the offline map.")
ax.set_ylim(2.6, 58.75)
ax.text(2.0, 48.3, "INPUTS", fontsize=11, fontweight="demibold", color=MUTED, va="center")
panel(ax, 19.5, 7.3, 62.0, 39.9, LTEAL)
ax.text(21.0, 48.3, "DHRUVA ENGINE — on the phone", fontsize=11, fontweight="demibold", color=TEAL, va="center")
ax.text(84.0, 48.3, "OUTPUTS", fontsize=11, fontweight="demibold", color=MUTED, va="center")

box(ax, 2.0, 37.0, 15.0, 8.5, "Motion sensors", "accelerometer, gyroscope,\ngravity · 125 per second", fill=LGREY)
box(ax, 2.0, 23.8, 15.0, 7.4, "GPS", "position · once a second", fill=LGREY)
box(ax, 2.0, 8.9, 15.0, 7.4, "Offline map", "roads + places,\nstored on the phone", fill=LGREY)

box(ax, 21.0, 38.3, 12.8, 7.2, "Sensor features", "levelled to the road,\nlast 30 s at 10 Hz", edge=TEDGE, ts=11)
box(ax, 21.0, 29.3, 12.8, 7.2, "Heading", "gyroscope turn rate,\nbias removed", edge=TEDGE, ts=11)
box(ax, 36.3, 38.3, 13.2, 7.2, "AI speed model", "1-D ResNet on the phone,\nnew speed every 0.5 s", edge=TEAL, tcolor=TEAL, ts=11, lw=2.0)
box(ax, 36.3, 23.8, 13.2, 7.4, "GPS check", "good: follow GPS\nlost: dead reckoning", edge=TEDGE, ts=11)
box(ax, 52.0, 23.8, 12.6, 21.7, "Position",
    "Fusion engine\n(10 times a second)\ncombines GPS, AI speed,\nheading and the road\n\nRoad binding\nkeeps the dot on\nthe planned route",
    edge=TEAL, tcolor=TEAL, ts=12, lw=2.0)
box(ax, 21.0, 8.9, 28.5, 7.4, "Route planning (offline)", "search the destination · fastest route · list of turns",
    edge=TEDGE, ts=11)
box(ax, 67.4, 8.9, 12.6, 29.5, "Guidance",
    "turn-by-turn\ninstructions\n\narrival\n\nwrong-turn check:\ngyroscope vs route,\nfinds the road taken,\nplans a new route",
    edge=TEDGE, ts=12)

box(ax, 84.0, 37.0, 14.0, 8.5, "Map screen", "your dot, confidence\ncircle and the route", fill=LGREY)
box(ax, 84.0, 23.8, 14.0, 7.4, "Voice", "“Turn right in 100 m”", fill=LGREY)
box(ax, 84.0, 8.9, 14.0, 7.4, "Trip record", "summary + the road\nyou rode", fill=LGREY)

arrow(ax, 17.0, 41.9, 21.0, 41.9, color=TEAL)
path(ax, [(17.0, 38.6), (19.0, 38.6), (19.0, 32.9), (21.0, 32.9)], color=TEAL)
arrow(ax, 33.8, 41.9, 36.3, 41.9, color=TEAL)
arrow(ax, 49.5, 41.9, 52.0, 41.9, color=TEAL)
arrow(ax, 33.8, 32.9, 52.0, 32.9, color=TEAL, label="heading", lx=42.9, ly=34.1)
arrow(ax, 17.0, 27.5, 36.3, 27.5, color=TEAL)
arrow(ax, 49.5, 27.5, 52.0, 27.5, color=TEAL)
arrow(ax, 17.0, 12.6, 21.0, 12.6, color=TEAL)
arrow(ax, 49.5, 14.3, 67.4, 14.3, color=TEAL, label="route + turns", lx=61.8, ly=15.5)
arrow(ax, 58.3, 14.3, 58.3, 23.8, color=TEAL, label="planned\nroute", lx=56.0, ly=19.3, ha="right")
arrow(ax, 67.4, 10.6, 49.5, 10.6, color=AMBER, label="wrong turn: new route", lx=58.3, ly=9.4, lc="#9A6418")
arrow(ax, 64.6, 34.8, 67.4, 34.8, color=TEAL)
arrow(ax, 64.6, 41.9, 84.0, 41.9, color=TEAL, label="your dot", lx=74.3, ly=43.1)
arrow(ax, 80.0, 27.5, 84.0, 27.5, color=TEAL)
arrow(ax, 80.0, 12.6, 84.0, 12.6, color=TEAL)
save(fig, "1_low_level_design.png")


# =========================================================== 2. HIGH-LEVEL: phone and cloud
fig, ax = canvas("System design: the phone and the cloud",
                 "The phone does all the navigating. The cloud keeps the maps fresh and shares the roads that "
                 "riders confirm.")
panel(ax, 2.5, 5.0, 42.5, 42.2, LTEAL, "ON THE PHONE — Dhruva app", TEAL)
box(ax, 5.0, 33.2, 37.5, 9.8, "Navigation engine",
    "sensors · AI speed model · GPS check · fusion\nroad binding · offline routing · voice guidance", edge=TEAL, tcolor=TEAL, ts=12.5)
ax.text(5.0, 30.4, "The map on the phone, checked top to bottom:", fontsize=10, color=MUTED, va="center")
for i, (t, d, c) in enumerate([("Your roads", "learned on your own rides", "#CDEBE4"),
                               ("Shared roads", "confirmed by 3 or more riders", "#DCEAF5"),
                               ("OpenStreetMap", "the base map of every road", LGREY)]):
    y = 23.6 - i * 5.4
    ax.add_patch(FancyBboxPatch((5.0, y), 37.5, 4.5, boxstyle="round,pad=0,rounding_size=0.8", fc=c, ec=c))
    ax.text(6.7, y + 2.25, t, fontsize=11, fontweight="demibold", color=INK, va="center")
    ax.text(20.0, y + 2.25, d, fontsize=10, color=INK, va="center")
ax.add_patch(FancyBboxPatch((5.0, 6.6), 37.5, 4.2, boxstyle="round,pad=0,rounding_size=0.8", fc="white", ec=TEAL, lw=1.4))
ax.text(23.75, 8.7, "DURING THE RIDE: fully offline, nothing sent or received", fontsize=10.2, fontweight="demibold",
        color=TEAL, ha="center", va="center")

panel(ax, 60.0, 5.0, 37.5, 42.2, LNAVY, "IN THE CLOUD", NAVY)
box(ax, 81.0, 33.2, 14.5, 8.4, "OpenStreetMap", "open map data\nof all of India", fill=LGREY, ts=11)
box(ax, 81.0, 19.8, 14.5, 9.0, "Map builder", "cuts the map into\n10 km map tiles", edge=NEDGE, tcolor=NAVY, ts=11.5)
box(ax, 62.0, 19.8, 15.5, 9.0, "Map tiles", "all of India,\nversioned", edge=NEDGE, tcolor=NAVY, ts=11.5)
box(ax, 62.0, 7.0, 15.5, 8.6, "Road shapes", "received from\nriders' phones", edge=NEDGE, tcolor=NAVY, ts=11.5)
box(ax, 81.0, 7.0, 14.5, 8.6, "Map merge", "adds a road when\n3 riders agree", edge=NEDGE, tcolor=NAVY, ts=11.5)
arrow(ax, 88.25, 33.2, 88.25, 28.8, color=NAVY)
arrow(ax, 81.0, 24.3, 77.5, 24.3, color=NAVY)
arrow(ax, 77.5, 11.3, 81.0, 11.3, color=NAVY)
arrow(ax, 88.25, 15.6, 88.25, 19.8, color=NAVY, label="confirmed\nroads", lx=89.2, ly=17.7, ha="left")

arrow(ax, 62.0, 25.8, 45.0, 25.8, color=NAVY, lw=2.3)
ax.text(52.5, 30.0, "BEFORE THE RIDE", fontsize=10, fontweight="demibold", color=NAVY, ha="center", va="center")
ax.text(52.5, 27.6, "map tiles for your\narea and route", fontsize=9.3, color=INK, ha="center", va="center", linespacing=1.2)
ax.text(52.5, 22.9, "then weekly: updated tiles", fontsize=9.3, color=INK, ha="center", va="center")
arrow(ax, 45.0, 11.3, 62.0, 11.3, color=AMBER, lw=2.3)
ax.text(52.5, 15.4, "AFTER THE RIDE", fontsize=10, fontweight="demibold", color="#9A6418", ha="center", va="center")
ax.text(52.5, 13.3, "new road shapes", fontsize=9.3, color=INK, ha="center", va="center")
ax.text(52.5, 9.2, "opt-in, on wifi,\nanonymous", fontsize=9.0, color=MUTED, ha="center", va="center", linespacing=1.2)
save(fig, "2_high_level_design.png")


# =========================================================== 3. DATA FLOW DIAGRAM
fig, ax = canvas("Data flow: what data moves where",
                 "Rectangles are sources, rounded boxes are processes, open boxes are where data is kept.")
panel(ax, 1.5, 36.2, 97.0, 11.7, LNAVY, "CLOUD", NAVY)
panel(ax, 1.5, 1.8, 97.0, 32.8, LTEAL, "PHONE", TEAL)


def entity(x, y, w, h, t):
    ax.add_patch(plt.Rectangle((x, y), w, h, fc=LGREY, ec=GREY, lw=1.4))
    ax.text(x + w / 2, y + h / 2, t, fontsize=10.8, fontweight="demibold", color=INK, ha="center", va="center", linespacing=1.2)


def process(x, y, w, h, n, t, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.4", fc="white", ec=color, lw=1.8))
    ax.text(x + w / 2, y + h / 2 - 0.2, t, fontsize=10.6, fontweight="demibold", color=INK, ha="center", va="center", linespacing=1.2)
    step(ax, x + 0.4, y + h - 0.4, n, color, r=1.15, fs=9.5)


def store(x, y, w, h, t):
    ax.plot([x, x + w], [y + h, y + h], color=INK, lw=1.4); ax.plot([x, x + w], [y, y], color=INK, lw=1.4)
    ax.plot([x, x], [y, y + h], color=INK, lw=1.4)
    ax.add_patch(plt.Rectangle((x, y), w, h, fc="white", ec="none", zorder=0.5))
    ax.text(x + w / 2, y + h / 2, t, fontsize=10.4, color=INK, ha="center", va="center", linespacing=1.2)


entity(4.0, 38.0, 13.0, 6.0, "OpenStreetMap")
process(28.0, 37.6, 15.5, 6.8, 1, "Build map\ntiles", NAVY)
store(55.0, 38.0, 14.0, 6.0, "Map tiles\n(all India)")
process(80.0, 37.6, 15.5, 6.8, 6, "Merge shared\nroads", NAVY)
arrow(ax, 17.0, 41.0, 28.0, 41.0, color=NAVY, label="roads + places", ly=42.2)
arrow(ax, 43.5, 41.0, 55.0, 41.0, color=NAVY, label="map tiles", ly=42.2)
path(ax, [(87.75, 44.4), (87.75, 46.4), (35.75, 46.4), (35.75, 44.4)], color=NAVY,
     label="roads 3+ riders agree on", lx=61.9, ly=45.4)

entity(3.5, 26.0, 13.0, 5.6, "Rider\n(destination)")
entity(3.5, 16.6, 13.0, 5.6, "Motion\nsensors")
entity(3.5, 8.6, 13.0, 5.6, "GPS")
process(26.0, 24.9, 15.5, 7.2, 2, "Plan route", TEAL)
store(55.0, 25.5, 14.0, 6.0, "Map on the phone\n(tiles + your roads)")
process(80.0, 24.9, 15.5, 7.2, 5, "Learn the\nroad ridden", TEAL)
process(26.0, 10.9, 15.5, 7.2, 3, "Estimate\nposition", TEAL)
process(51.0, 10.9, 15.5, 7.2, 4, "Guide rider", TEAL)
store(80.5, 11.5, 15.5, 6.0, "Trip record\n(GPS track)")
entity(80.5, 2.6, 15.5, 5.8, "Rider\n(sees + hears)")

arrow(ax, 62.0, 38.0, 62.0, 31.5, color=NAVY, label="tiles for your area", lx=63.0, ly=34.8, ha="left")
arrow(ax, 16.5, 28.8, 26.0, 28.8, color=TEAL, label="destination", ly=30.0)
arrow(ax, 55.0, 28.5, 41.5, 28.5, color=TEAL, label="roads + places", ly=29.7)
arrow(ax, 80.0, 28.5, 69.0, 28.5, color=TEAL, label="your roads", ly=29.7)
arrow(ax, 33.75, 24.9, 33.75, 18.1, color=TEAL, label="planned route", lx=34.7, ly=21.5, ha="left")
arrow(ax, 16.5, 19.4, 26.0, 16.6, color=TEAL, label="motion, 125/s", lx=21.0, ly=20.4)
arrow(ax, 16.5, 11.4, 26.0, 12.4, color=TEAL, label="position fix, 1/s", lx=21.3, ly=10.3)
arrow(ax, 41.5, 14.5, 51.0, 14.5, color=TEAL, label="position", ly=15.7)
path(ax, [(41.5, 26.6), (46.0, 26.6), (46.0, 16.8), (51.0, 16.8)], color=TEAL, label="turns", lx=47.2, ly=21.5, ha="left")
arrow(ax, 66.5, 14.5, 80.5, 14.5, color=TEAL, label="the trip, as it happens", ly=15.7)
path(ax, [(58.75, 10.9), (58.75, 5.5), (80.5, 5.5)], color=TEAL, label="dot on the map + voice", lx=69.6, ly=6.7)
arrow(ax, 88.25, 17.5, 88.25, 24.9, color=TEAL, label="after the trip", lx=89.2, ly=21.2, ha="left")
arrow(ax, 87.75, 32.1, 87.75, 37.6, color=AMBER, lw=2.1, label="new road shapes\n(opt-in, wifi)", lx=88.8, ly=34.8,
      lc="#9A6418", ha="left")
save(fig, "3_data_flow.png")


# =========================================================== 4. USER JOURNEY
fig, ax = canvas("How a rider uses Dhruva",
                 "Start the trip with GPS. If GPS is lost on the way, Dhruva keeps guiding you to the destination.")
panel(ax, 2.0, 7.0, 15.0, 38.0, LNAVY, "BEFORE", NAVY)
panel(ax, 83.0, 7.0, 15.0, 38.0, LNAVY, "AFTER", NAVY)
ax.text(50.0, 46.0, "DURING THE RIDE", fontsize=11, fontweight="demibold", color=TEAL, ha="center", va="center")

y_road = 25.0
ax.add_patch(plt.Rectangle((43.0, y_road - 7.0), 21.5, 14.0, fc=LAMBER, ec="none"))
ax.text(53.0, y_road - 4.6, "NO GPS", fontsize=10.5, fontweight="demibold", color="#9A6418", ha="center", va="center")
ax.plot([20.5, 79.5], [y_road, y_road], color="#3F4B5B", lw=9, solid_capstyle="round")
for x0 in range(22, 80, 3):
    ax.plot([x0, x0 + 1.4], [y_road, y_road], color="white", lw=1.4)
ax.text(20.5, y_road - 3.0, "start", fontsize=9.5, color=MUTED, ha="center", va="center")
ax.text(79.5, y_road - 3.0, "destination", fontsize=9.5, color=MUTED, ha="center", va="center")

box(ax, 3.3, 22.6, 12.4, 16.5, "Download\nyour area",
    "once, on wifi:\nthe offline map\nfor your city and\nyour route", fill="white", edge=NEDGE, tcolor=NAVY, ts=11)
step(ax, 9.5, 41.2, 1, NAVY)
box(ax, 3.3, 8.6, 12.4, 11.6, "Mount the phone", "open Dhruva,\nwait for GPS", fill="white", edge=NEDGE, tcolor=NAVY, ts=10.8)

stations = [
    (25.0, 2, "above", "Search the\ndestination", "works offline:\n“Unkal Lake”"),
    (32.5, 3, "below", "Ride with voice", "turn-by-turn\ninstructions"),
    (46.5, 4, "below", "GPS lost", "tunnel or underpass:\nthe dot keeps moving,\nthe voice keeps guiding"),
    (57.0, 5, "above", "Wrong turn?", "Dhruva notices\nand plans a new route"),
    (66.0, 6, "below", "GPS back", "carries on\nseamlessly"),
    (75.5, 7, "above", "Arrived", "“You have arrived”"),
]
for x, n, where, t, d in stations:
    col = AMBER if n in (4, 5) else TEAL
    step(ax, x, y_road, n, col, r=1.6, fs=11.5)
    w = 12.8 if n != 4 else 13.6
    if where == "above":
        box(ax, x - w / 2, y_road + 8.8, w, 9.2, t, d, edge=AEDGE if n in (4, 5) else TEDGE,
            tcolor="#9A6418" if n in (4, 5) else TEAL, ts=11, bs=9.2)
        ax.plot([x, x], [y_road + 1.6, y_road + 8.8], color=GREY, lw=1.0)
    else:
        box(ax, x - w / 2, y_road - 18.6, w, 9.8, t, d, edge=AEDGE if n in (4, 5) else TEDGE,
            tcolor="#9A6418" if n in (4, 5) else TEAL, ts=11, bs=9.2)
        ax.plot([x, x], [y_road - 1.6, y_road - 8.8], color=GREY, lw=1.0)

box(ax, 84.3, 22.6, 12.4, 16.5, "Road\nremembered", "the road you rode\nis saved on your\nphone for next time",
    fill="white", edge=NEDGE, tcolor=NAVY, ts=11)
step(ax, 90.5, 41.2, 8, NAVY)
box(ax, 84.3, 8.6, 12.4, 11.6, "Share it", "optionally, on wifi:\nhelps every rider", fill="white", edge=NEDGE,
    tcolor=NAVY, ts=10.8)
save(fig, "4_user_journey.png")


# =========================================================== 5. FULL ARCHITECTURE
fig, ax = canvas("Full architecture",
                 "Every component, where it runs, and how the parts connect.")
panel(ax, 2.0, 3.0, 61.0, 44.8, LTEAL, "PHONE — Android app (Kotlin), works offline", TEAL)
panel(ax, 66.0, 18.6, 32.0, 29.2, LNAVY, "CLOUD", NAVY)
panel(ax, 66.0, 3.0, 32.0, 13.8, LPURPLE, "AI MODEL — built in our lab", PURPLE)

layers = [
    ("App screens", 36.0, 5.8, [("Map screen", "dot · circle · route"), ("Destination search", "offline"),
                                ("Voice guidance", "text-to-speech"), ("Trip summary", "drift · arrival")]),
    ("Navigation engine", 20.4, 12.8, None),
    ("Data on the phone", 12.2, 5.6, [("Offline map", "roads + places"), ("Map pictures", "saved for offline"),
                                      ("Your roads", "learned rides"), ("Trip records", "summaries")]),
    ("Phone hardware", 3.8, 5.6, [("Motion sensors", "accel · gyro · gravity"), ("GPS receiver", "satellites"),
                                  ("Processor", "runs the AI model"), ("Storage", "maps + records")]),
]
for name, y, h, cells in layers:
    ax.text(3.6, y + h + 1.2, name.upper(), fontsize=9.3, fontweight="demibold", color=TEAL, va="center")
    if cells is None:
        continue
    for i, (t, d) in enumerate(cells):
        box(ax, 3.6 + i * 14.6, y, 13.6, h, t, d, edge=TEDGE if name != "Phone hardware" else GEDGE,
            fill="white" if name != "Phone hardware" else LGREY, ts=10.4, bs=8.8)
engine = [("Sensor features", "levelled, 10 Hz"), ("AI speed model", "ONNX Runtime"), ("Heading", "gyro, bias removed"),
          ("GPS check", "GPS or dead reckoning"), ("Fusion engine", "10 times a second"), ("Road binding", "dot on the route"),
          ("Router + search", "fastest route"), ("Guidance", "turns · arrival"), ("Wrong-turn check", "finds the road taken"),
          ("Fix filter", "drops bad GPS")]
for i, (t, d) in enumerate(engine):
    r, c = divmod(i, 5)
    col = TEAL if t == "AI speed model" else INK
    box(ax, 3.6 + c * 11.72, 27.2 - r * 6.8, 10.9, 6.0, t, d, edge=TEAL if t == "AI speed model" else TEDGE,
        tcolor=col, ts=9.9, bs=8.3, lw=2.0 if t == "AI speed model" else 1.4)

box(ax, 68.0, 36.2, 13.8, 7.6, "OpenStreetMap", "open map data", fill=LGREY, ts=10.6, bs=8.8)
box(ax, 83.2, 36.2, 13.2, 7.6, "Map builder", "10 km map tiles", edge=NEDGE, tcolor=NAVY, ts=10.6, bs=8.8)
box(ax, 68.0, 24.0, 13.8, 8.4, "Map tiles", "all of India,\nversioned", edge=NEDGE, tcolor=NAVY, ts=10.6, bs=8.8)
box(ax, 83.2, 24.0, 13.2, 8.4, "Map merge", "roads 3+ riders\nagree on", edge=NEDGE, tcolor=NAVY, ts=10.6, bs=8.8)
arrow(ax, 81.8, 40.0, 83.2, 40.0, color=NAVY)
arrow(ax, 89.8, 36.2, 81.8, 32.4, color=NAVY)
arrow(ax, 89.8, 32.4, 89.8, 36.2, color=NAVY)

box(ax, 88.0, 4.6, 9.0, 8.0, "Datasets", "IO-VNBD +\nour rides", edge=PEDGE, tcolor=PURPLE, ts=10.2, bs=8.6)
box(ax, 77.5, 4.6, 9.0, 8.0, "Training", "PyTorch,\n1-D ResNet", edge=PEDGE, tcolor=PURPLE, ts=10.2, bs=8.6)
box(ax, 68.0, 4.6, 8.0, 8.0, "AI model", "ONNX file", edge=PURPLE, tcolor=PURPLE, ts=10.2, bs=8.6, lw=2.0)
arrow(ax, 88.0, 8.6, 86.5, 8.6, color=PURPLE)
arrow(ax, 77.5, 8.6, 76.0, 8.6, color=PURPLE)
arrow(ax, 68.0, 8.6, 63.0, 8.6, color=PURPLE, lw=2.3)
ax.text(64.5, 12.0, "shipped\ninside\nthe app", fontsize=8.8, color=PURPLE, ha="center", va="center", linespacing=1.15)

arrow(ax, 68.0, 29.4, 63.0, 29.4, color=NAVY, lw=2.3)
ax.text(64.5, 31.9, "map\ntiles", fontsize=8.8, color=NAVY, ha="center", va="center", linespacing=1.15)
path(ax, [(63.0, 21.8), (89.8, 21.8), (89.8, 24.0)], color=AMBER, lw=2.1)
ax.text(76.0, 20.6, "new road shapes (opt-in)", fontsize=8.8, color="#9A6418", ha="center", va="center")
save(fig, "5_full_architecture.png")
