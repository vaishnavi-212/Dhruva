"""One-slide workflow diagram for the pitch: how Dhruva works, in plain words."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

INK, TEAL, AMBER, GREY, MUTED, NAVY = "#1F2A37", "#1F8A7A", "#D9922E", "#8D99AE", "#5B6573", "#1F4E79"
LTEAL, LAMBER, LBLUE, LGREY = "#E4F2EF", "#FBF1E1", "#EAF2FA", "#F2F4F7"
plt.rcParams.update({"font.family": "Avenir Next", "savefig.dpi": 200})
import os
OUT = [os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dhruva_architecture.png"))]

fig = plt.figure(figsize=(13, 7.3)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 56.15); ax.axis("off")
fig.text(0.04, 0.955, "How Dhruva works", fontsize=21, fontweight="demibold", color=INK, va="top")
fig.text(0.04, 0.895, "Start with GPS on. When GPS drops, distance comes from the AI model and direction comes from the road.",
         fontsize=12.5, color=MUTED, va="top")


def box(x, y, w, h, fill, edge=None, dashed=False, r=1.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fill,
                                ec=edge or fill, lw=1.6, ls=(0, (4, 3)) if dashed else "-"))


def arrow(x0, y0, x1, y1, color=INK, lw=2.2):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, mutation_scale=16))


# ---- band 1: while GPS is on ----
box(2.5, 36.5, 95, 9.2, LAMBER)
ax.text(4.2, 44.2, "WHILE GPS IS ON", fontsize=11, fontweight="demibold", color="#9A6418", va="center")
chips = ["Track where you are", "Save the road you ride\n(self-healing map)", "Remember your speed and\ndirection before GPS drops"]
for i, c in enumerate(chips):
    x = 22 + i * 25.5
    box(x, 37.6, 22.5, 6.9, "white", edge="#E8C790")
    ax.text(x + 11.25, 41.05, c, fontsize=11, color=INK, ha="center", va="center", linespacing=1.25)
    if i < 2: arrow(x + 22.7, 41.05, x + 25.3, 41.05, color="#C98A2E", lw=1.8)

# ---- GPS drops marker ----
ax.text(50, 34.1, "GPS drops: tunnel · underpass · dense lanes", fontsize=11, color="#9A6418", ha="center", va="center",
        fontweight="demibold", bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=AMBER, lw=1.4))
arrow(50, 36.3, 50, 35.4, color=AMBER, lw=1.8)

# ---- band 2: the pipeline ----
steps = [("Phone sensors", "accelerometer, gyroscope,\ngravity · 100 times a second", LBLUE, NAVY, False),
         ("Level the phone", "remove tilt so motion and\nturns read correctly", LBLUE, NAVY, False),
         ("AI speed model", "reads 30 s of motion and\nworks out distance travelled", LTEAL, TEAL, True),
         ("Road binding", "moves that distance along\nthe road; the road gives\nthe direction", LTEAL, TEAL, False),
         ("On screen", "position · 90% confidence\ncircle · voice guidance", LBLUE, NAVY, False)]
w, g, x0, y, h = 16.9, 2.6, 2.5, 17.5, 13.2
for i, (t, b, fill, col, dashed) in enumerate(steps):
    x = x0 + i * (w + g)
    box(x, y, w, h, fill, edge=col if dashed else None, dashed=dashed)
    ax.text(x + w / 2, y + h - 2.6, t, fontsize=13.5, fontweight="demibold", color=col, ha="center", va="center")
    ax.text(x + w / 2, y + h / 2 - 1.0, b, fontsize=10.2, color=INK, ha="center", va="center", linespacing=1.3)
    if i < 4: arrow(x + w + 0.25, y + h / 2, x + w + g - 0.25, y + h / 2)
    if dashed:
        ax.text(x + w / 2, y - 1.6, "laptop today · phone next", fontsize=9, color=TEAL, ha="center", va="center", style="italic")
arrow(50, 32.7, 50, 31.0, color=AMBER, lw=1.8)

# ---- road map store feeding road binding ----
xb = x0 + 3 * (w + g)
box(54, 3.2, 27, 9.6, "white", edge=TEAL)
ax.text(67.5, 10.3, "Road map", fontsize=12.5, fontweight="demibold", color=TEAL, ha="center", va="center")
ax.text(67.5, 6.6, "OpenStreetMap roads + roads\nsaved while GPS was on", fontsize=10.2, color=INK, ha="center", va="center", linespacing=1.3)
arrow(xb + w / 2, 13.0, xb + w / 2, 17.2, color=TEAL)

# ---- proof strip ----
box(2.5, 3.2, 47.5, 9.6, LGREY)
ax.text(4.2, 10.4, "CHECKED THE ISRO WAY", fontsize=10, fontweight="demibold", color=MUTED, va="center")
ax.text(4.2, 6.5, "drift = error ÷ distance without GPS, limit 10%\n"
                  "11 real rides: 6.8% median  ·  demo rides after GPS off: 3.4% and 4.5%",
        fontsize=10.8, color=INK, va="center", linespacing=1.45)

# ---- GPS returns ----
xr = x0 + 4 * (w + g)
box(83, 3.2, 14.4, 9.6, LAMBER)
ax.text(90.2, 10.3, "GPS comes back", fontsize=12, fontweight="demibold", color="#9A6418", ha="center", va="center")
ax.text(90.2, 6.6, "dot glides back,\nat most 2.5 m jump", fontsize=10.2, color=INK, ha="center", va="center", linespacing=1.3)
arrow(xr + w / 2, 13.0, xr + w / 2, 17.2, color=AMBER)

for o in OUT:
    fig.savefig(o)
print("saved")
