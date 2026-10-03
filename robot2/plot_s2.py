"""Figure for S2 (temporal analysis), drawn from cached results2.json.

    python plot_s2.py
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from env2 import COLS, CORR, X_START, H

SURF, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
REACH = H * 0.101          # x-distance covered by one 12-step chunk


def plot(r, path="figs/s2_temporal.png"):
    plt.rcParams.update({
        "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
        "text.color": INK, "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
        "axes.titlesize": 11, "axes.titleweight": "semibold", "legend.frameon": False, "lines.linewidth": 2})
    x = np.array(r["x"]); g0 = 0.1
    bins = np.linspace(X_START, COLS[1]["x"] + CORR, 23); xc = 0.5 * (bins[1:] + bins[:-1])
    b = np.digitize(x, bins) - 1

    def binned(key):
        v = np.array(r[key], float)
        out = np.full(len(xc), np.nan)
        for k in range(len(xc)):
            m = (b == k) & ~np.isnan(v)
            if m.any():
                out[k] = v[m].mean()
        return out

    # B's transition first enters the chunk when the robot is one chunk-reach before A's exit
    x_vis = COLS[0]["x"] + CORR - REACH
    fig, axs = plt.subplots(2, 1, figsize=(10, 7), sharex=True, constrained_layout=True)
    for ax in axs:
        for c in COLS:
            ax.axvspan(c["x"] - CORR, c["x"] + CORR, color="#f0efec", lw=0, zorder=0)
        ax.axvline(x_vis, color=MUTED, lw=1, ls=":")
    ax = axs[0]
    ax.plot(xc, binned("HA"), color=BLUE, label="closed loop: column-A choice")
    ax.plot(xc, binned("HB"), color=ORANGE, label="closed loop: column-B choice")
    ax.plot(xc, binned("Hchunk"), color=INK2, lw=1.4, ls=(0, (4, 3)), label="one chunk: the gap it is heading for")
    ax.set_ylim(-0.05, 1.45)
    ax.set_ylabel("entropy (bits)")
    ax.legend(fontsize=8.5, loc="center right", bbox_to_anchor=(1.0, 0.45))
    ax.set_title(f"Along 40 rollouts at goal g = {g0}: when each choice is made, and when the flow can see it")
    for c in COLS:
        ax.text(c["x"], 1.40, f"column {c['name']}", ha="center", va="top", fontsize=9, color=INK2)
    ax.text(x_vis - 0.04, 1.0, "column-B turn\nwithin one chunk →", ha="right", fontsize=8.5, color=INK2)
    ax = axs[1]
    ax.plot(xc, binned("ftle_max"), color=BLUE, label="max FTLE over 32 seeds")
    ax.plot(xc, binned("ftle_med"), color=AQUA, label="median FTLE")
    ax.plot(xc, binned("ridge"), color=ORANGE, marker="o", ms=3.5,
            label="ridge height (FTLE at bisected boundary)")
    ax.set_xlabel("robot x position at replanning"); ax.set_ylabel("FTLE")
    ax.legend(fontsize=8.5, loc="upper right")
    fig.savefig(path, dpi=160); plt.close(fig)


if __name__ == "__main__":
    plot(json.load(open("results2.json"))["S2"])
