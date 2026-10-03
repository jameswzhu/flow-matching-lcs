"""Figure for S3 (per-seed flip prediction), drawn from results2.json + s3_rows.npz.

    python plot_s3.py
"""
import json
import numpy as np
import matplotlib.pyplot as plt

SURF, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
SCORES = ["FTLE (noise)", "goal sensitivity log‖∂Φ/∂g‖", "joint FTLE (noise, goal)",
          "Eulerian max σ(∂v/∂a)", "log-norm bound ∫μ(∂v/∂a)", "state entropy (no seed info)"]
COLORS = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, MUTED]
TARGETS = {"open loop, noise": "ol_noise", "open loop, goal": "ol_goal",
           "closed loop, noise": "cl_noise", "closed loop, goal": "cl_goal"}


def plot(res, rows, path="figs/s3_prediction.png"):
    plt.rcParams.update({
        "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
        "text.color": INK, "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
        "axes.titlesize": 11, "axes.titleweight": "semibold", "legend.frameon": False, "lines.linewidth": 2})
    tn = list(TARGETS)
    fig = plt.figure(figsize=(16, 5.6), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 0.09], width_ratios=[1.25, 1.25, 1])
    axs = [fig.add_subplot(gs[0, k]) for k in range(3)]
    for ax, key, ttl in [(axs[0], "overall", "Across states and seeds"),
                         (axs[1], "within", "Within a state: which seeds flip?")]:
        w = 0.13
        for k, sn in enumerate(SCORES):
            v = np.array([res[t][sn][key] for t in tn])
            ci = np.array([res[t][sn][key + "_ci"] for t in tn])
            xs = np.arange(len(tn)) + (k - 2.5) * w
            ax.bar(xs, v, width=w - 0.02, color=COLORS[k], label=sn)
            if not (key == "within" and sn.startswith("state")):
                ax.errorbar(xs, v, yerr=[v - ci[:, 0], ci[:, 1] - v], fmt="none", ecolor=INK2, lw=1, capsize=2)
        ax.axhline(0.5, color=MUTED, lw=1, ls=":")
        npos = [int(rows[TARGETS[t]].sum()) for t in tn]
        ax.set_xticks(range(len(tn)))
        ax.set_xticklabels([f"{t}\n{n} flips ({100 * res[t]['base_rate']:.1f}%)" for t, n in zip(tn, npos)], fontsize=8.5)
        ax.set_ylim(0.3, 1.0); ax.set_ylabel("AUROC"); ax.set_title(ttl + (" (state entropy = 0.5 by construction)" if key == "within" else ""))
    ax = axs[2]
    f = rows["ftle"]
    qs = [0, .5, .8, .9, .95, .98, .99, .995, 1.0]
    edges = np.quantile(f, qs)
    bd = np.clip(np.searchsorted(edges, f, side="right") - 1, 0, len(qs) - 2)
    labels = [f"{int(100 * a)}–{100 * b:g}" for a, b in zip(qs[:-1], qs[1:])]
    for (t, key), c in zip(TARGETS.items(), [BLUE, ORANGE, AQUA, YELLOW]):
        y = rows[key]
        rate = np.array([y[bd == k].mean() for k in range(len(labels))])
        ax.plot(np.arange(len(labels))[rate > 0], rate[rate > 0], color=c, marker="o", ms=4, label=t)
    ax.set_yscale("log"); ax.set_ylim(1e-4, 1.2)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=35, fontsize=8)
    ax.set_xlabel("FTLE percentile band"); ax.set_ylabel("flip rate (log scale; empty bands omitted)")
    ax.set_title("Flips sit in the top 1% of FTLE")
    ax.legend(fontsize=8.5, loc="upper left")
    lax = fig.add_subplot(gs[1, :2]); lax.axis("off")
    h, l = axs[0].get_legend_handles_labels()
    lax.legend(h, l, ncol=6, fontsize=8.5, loc="center")
    fig.suptitle("Predicting mode flips seed by seed (360 states × 24 seeds; 95% cluster-bootstrap CIs)",
                 fontweight="semibold")
    fig.savefig(path, dpi=160); plt.close(fig)


if __name__ == "__main__":
    plot(json.load(open("results2.json"))["S3"], dict(np.load("s3_rows.npz")))
