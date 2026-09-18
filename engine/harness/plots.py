"""Plots. The position plot is a REQUIRED submission artifact for SIH26168."""
from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, ACC, TRUTH, BASE = "#131A1C", "#0E6E75", "#5F6E70", "#B4551A"


def position_plot(trace, outage, preds: dict, path: str, title: str = "") -> str:
    """THE submission plot: predicted track vs ground truth through a blackout."""
    sl = outage.slice()
    truth = trace.xy[sl]
    fig, ax = plt.subplots(figsize=(8, 7))

    ax.plot(trace.xy[:, 0], trace.xy[:, 1], color="#DCE1E0", lw=1,
            zorder=1, label="full drive (GNSS)")
    ax.plot(truth[:, 0], truth[:, 1], color=TRUTH, lw=3, zorder=3,
            label="ground truth (blackout)")

    colors = [ACC, BASE, "#7A5AA8", "#2E7D32"]
    for i, (name, xy) in enumerate(preds.items()):
        err = np.linalg.norm(xy[-1] - truth[-1])
        ax.plot(xy[:, 0], xy[:, 1], color=colors[i % len(colors)], lw=2,
                ls="--" if i else "-", zorder=4,
                label=f"{name} (final drift {err:.1f} m)")
        ax.scatter(*xy[-1], color=colors[i % len(colors)], s=45, zorder=5)

    ax.scatter(*truth[0], color=INK, s=70, marker="o", zorder=6, label="GNSS lost")
    ax.scatter(*truth[-1], color=INK, s=90, marker="*", zorder=6, label="GNSS regained")

    dist = np.linalg.norm(np.diff(truth, axis=0), axis=1).sum()
    ax.set_title(title or f"Blackout: {dist:.0f} m travelled", color=INK, fontsize=13)
    ax.set_xlabel("east (m)"); ax.set_ylabel("north (m)")
    ax.set_aspect("equal", "datalim"); ax.grid(alpha=.2)
    ax.legend(fontsize=8, loc="best", framealpha=.95)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)
    return path


def drift_curve(results_by_model: dict, path: str, x: str = "distance_m") -> str:
    """Drift vs distance (or duration). The chart that decides the project."""
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = [ACC, BASE, "#7A5AA8", "#2E7D32"]
    for i, (name, results) in enumerate(results_by_model.items()):
        xs = np.array([getattr(r, x) for r in results])
        ys = np.array([r.drift_pct for r in results])
        order = np.argsort(xs)
        xs, ys = xs[order], ys[order]
        bins = np.unique(np.round(xs, -1))
        med = [np.median(ys[np.round(xs, -1) == b]) for b in bins]
        ax.plot(bins, med, "o-", color=colors[i % len(colors)], lw=2, label=name)

    ax.axhline(10.0, color=INK, ls=":", lw=1.5)
    ax.text(ax.get_xlim()[1], 10.4, "ISRO limit: 10%", ha="right", fontsize=9, color=INK)
    ax.set_xlabel("distance travelled during blackout (m)" if x == "distance_m"
                  else "blackout duration (s)")
    ax.set_ylabel("positional drift (% of distance)")
    ax.set_yscale("log"); ax.grid(alpha=.25, which="both")
    ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)
    return path
