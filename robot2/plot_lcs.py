"""Analysis + figures for the LCS study (lcs_eval.py analyze)."""
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from lcs import p_flip_noise
from plot_mc import auroc

SURF, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7"
EPS = [0.03, 0.05, 0.1, 0.2, 0.4]
DGS = [0.02, 0.05, 0.1, 0.2]


def style():
    plt.rcParams.update({
        "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
        "text.color": INK, "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
        "axes.titlesize": 11, "axes.titleweight": "semibold", "legend.frameon": False, "lines.linewidth": 2})


def brier(p, y):
    return float(np.mean((p - y) ** 2))


def ece(p, y, nb=10):
    """Expected calibration error with quantile bins over the positive-probability range."""
    edges = np.unique(np.quantile(p, np.linspace(0, 1, nb + 1)))
    b = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
    e = 0.0
    for k in range(len(edges) - 1):
        m = b == k
        if m.any():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(e)


def logistic_fit(x, y, iters=50):
    X = np.column_stack([np.ones_like(x), x]); w = np.zeros(2)
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ w))
        Hm = (X * (p * (1 - p))[:, None]).T @ X + 1e-6 * np.eye(2)
        w -= np.linalg.solve(Hm, X.T @ (p - y))
    return w


def reliability(p, y, edges):
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi)
        if m.sum() >= 15:
            out.append((p[m].mean(), y[m].mean(), int(m.sum())))
    return np.array(out)


def analyze():
    P = np.load("mc/prep.npz"); S3 = np.load("s3_rows.npz")
    Sc = np.load("lcs/scan.npz"); Rf = np.load("lcs/refine.npz"); Sw = np.load("lcs/sweep.npz")
    dp, dm, rid, vel = Rf["d_plus"], Rf["d_minus"], Rf["ridge"], Rf["vel"]
    f1 = P["ftle1"]; N = len(f1)
    res = {}
    # ---------------------------------------------------------------- L1 geometry
    near = np.where(np.nan_to_num(dp, nan=9) <= np.nan_to_num(dm, nan=9), 0, 1)
    d_near = np.where(near == 0, dp, dm)
    has = ~np.isnan(d_near)
    r_near = np.where(near == 0, rid[:, 0], -rid[:, 1])
    err = np.abs(r_near - d_near)[has]
    res["geometry"] = dict(frac_cases_with_LCS_within_3=float(has.mean()),
                           ridge_minus_separatrix_median=float(np.median(err)),
                           ridge_within_0p1_of_separatrix=float((err < 0.1).mean()),
                           ridge_within_0p1_when_margin_lt_0p5=float((err[d_near[has] < 0.5] < 0.1).mean()))
    # ---------------------------------------------------------------- L2 noise flips across eps
    w = logistic_fit(f1, Sw["noise_0.1"].astype(float))
    res["noise"] = {}
    for eps in EPS:
        y = Sw[f"noise_{eps}"].astype(float)
        p_lcs = p_flip_noise(dp, dm, eps)
        p_ftle = 1 / (1 + np.exp(-(w[0] + w[1] * f1)))
        res["noise"][str(eps)] = dict(
            flips=int(y.sum()), observed_rate=float(y.mean()), lcs_predicted_rate=float(p_lcs.mean()),
            ftle_logistic_predicted_rate=float(p_ftle.mean()),
            auroc_lcs=auroc(p_lcs + 1e-12 * -np.nan_to_num(np.fmin(dp, dm), nan=99), y),
            auroc_ftle=auroc(f1, y), brier_lcs=brier(p_lcs, y), brier_ftle_logistic=brier(p_ftle, y),
            brier_base_rate=brier(np.full(N, y.mean()), y), ece_lcs=ece(p_lcs, y), ece_ftle_logistic=ece(p_ftle, y))
    # the S3 closed-loop target (eps = 0.1, 4 directions, final route)
    p_cl = p_flip_noise(dp, dm, 0.1)
    res["closed_loop_noise"] = dict(auroc_lcs=auroc(p_cl - 1e-9 * np.nan_to_num(np.fmin(dp, dm), nan=99), S3["cl_noise"]),
                                    auroc_ftle=auroc(f1, S3["cl_noise"]), flips=int(S3["cl_noise"].sum()))
    # ---------------------------------------------------------------- L2b goal flips
    mg = np.fmin(np.where(np.isnan(dp), np.inf, dp / np.abs(vel[:, 0])),
                 np.where(np.isnan(dm), np.inf, dm / np.abs(vel[:, 1])))     # goal change that reaches an LCS
    res["goal"] = {}
    for dg in DGS:
        y = Sw[f"goal_{dg}"]
        pred = mg <= dg
        tp = (pred & y).sum()
        res["goal"][str(dg)] = dict(flips=int(y.sum()), predicted=int(pred.sum()),
                                    precision=float(tp / max(pred.sum(), 1)), recall=float(tp / max(y.sum(), 1)),
                                    accuracy=float((pred == y).mean()),
                                    auroc_lcs_margin=auroc(-np.where(np.isinf(mg), 1e6, mg), y),
                                    auroc_one_chunk_gsens=auroc(S3["gsens"], y))
    # ---------------------------------------------------------------- L3 timing
    T = np.load("lcs/timing.npz")
    ok = ~np.isnan(T["mc_next"]); rz = ~np.isnan(T["realized"])
    res["timing"] = dict(
        n_replans=int(len(T["lcs_mass"])), n_realized=int(rz.sum()), realized_switches=int(np.nansum(T["realized"])),
        spearman_lcs_mass_vs_mc_now=float(spearmanr(T["lcs_mass"], T["mc_now"])[0]),
        spearman_lcs_mass_vs_mc_next=float(spearmanr(T["lcs_mass"][ok], T["mc_next"][ok])[0]),
        mae_lcs_mass_vs_mc_next=float(np.abs(T["lcs_mass"][ok] - T["mc_next"][ok]).mean()),
        mae_mc_now_vs_mc_next=float(np.abs(T["mc_now"][ok] - T["mc_next"][ok]).mean()),
        auroc_realized_lcs_mass=auroc(T["lcs_mass"][rz], T["realized"][rz].astype(bool)),
        auroc_realized_mc_now=auroc(T["mc_now"][rz], T["realized"][rz].astype(bool)),
        auroc_realized_ftle=auroc(T["ftle"][rz], T["realized"][rz].astype(bool)))
    json.dump(res, open("results_lcs.json", "w"), indent=1, default=float)
    print(json.dumps(res, indent=1, default=float))
    figs(res, dict(dp=dp, dm=dm, rid=rid, d_near=d_near, r_near=r_near, has=has, f1=f1, w=w, mg=mg), Sw, Sc, T, P)
    return res


def figs(res, D, Sw, Sc, T, P):
    style()
    fig, axs = plt.subplots(1, 3, figsize=(16.5, 4.9), constrained_layout=True)
    # (a) label-free ridge vs separatrix
    ax = axs[0]
    m = D["has"]
    ax.scatter(D["d_near"][m], D["r_near"][m], s=4, color=BLUE, alpha=.35, lw=0)
    ax.plot([0, 3], [0, 3], color=MUTED, lw=1, ls=":")
    ax.set_xlim(0, 3); ax.set_ylim(0, 3)
    ax.set_xlabel("distance to separatrix along LCS normal (label change)")
    ax.set_ylabel("distance to FTLE ridge (label-free)")
    g = res["geometry"]
    ax.set_title(f"The FTLE ridge is the separatrix\n{100 * g['ridge_within_0p1_when_margin_lt_0p5']:.0f}% within 0.1 when it is within 0.5 of the seed")
    # (b) predicted vs observed flip rate across nudge sizes (no refitting)
    ax = axs[1]
    eps = np.array(EPS)
    obs = np.array([res["noise"][str(e)]["observed_rate"] for e in EPS])
    lcs = np.array([res["noise"][str(e)]["lcs_predicted_rate"] for e in EPS])
    ftl = np.array([res["noise"][str(e)]["ftle_logistic_predicted_rate"] for e in EPS])
    ax.plot(eps, 100 * ftl, color=MUTED, ls=(0, (4, 3)), marker="x", ms=5, label="FTLE → logistic fitted at ε = 0.1")
    ax.plot(eps, 100 * lcs, color=BLUE, marker="o", ms=5, label="LCS margin → analytic (no fitting)")
    ax.plot(eps, 100 * obs, color=INK, ls="none", marker="s", ms=6, label="observed")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks(eps); ax.set_xticklabels([f"{e:g}" for e in EPS])
    ax.set_yticks([0.2, 0.5, 1, 2]); ax.set_yticklabels(["0.2", "0.5", "1", "2"]); ax.minorticks_off()
    ax.set_xlabel("size ε of the seed nudge (4 random directions)"); ax.set_ylabel("flip rate (%)")
    bss = [1 - res["noise"][str(e)]["brier_lcs"] / res["noise"][str(e)]["brier_base_rate"] for e in EPS]
    ax.set_title("Seed nudges of any size: predicted flip rate")
    ax.text(0.98, 0.04, "Brier skill (LCS): " + ", ".join(f"{v:.2f}" for v in bss), transform=ax.transAxes,
            ha="right", fontsize=8.5, color=INK2)
    ax.legend(fontsize=8.5, loc="upper left")
    # (c) goal flips vs predicted goal margin
    ax = axs[2]
    mg = D["mg"]
    bins = np.logspace(-3, 1, 25); xc = np.sqrt(bins[1:] * bins[:-1])
    for dg, c in zip(DGS, [MAGENTA, BLUE, AQUA, ORANGE]):
        y = Sw[f"goal_{dg}"]
        fr = [y[(mg >= lo) & (mg < hi)].mean() if ((mg >= lo) & (mg < hi)).sum() >= 10 else np.nan for lo, hi in zip(bins[:-1], bins[1:])]
        ax.plot(xc, fr, color=c, marker="o", ms=3.5, label=f"goal ± {dg}")
        ax.axvline(dg, color=c, lw=1, ls=":")
    ax.set_xscale("log"); ax.set_xlabel("goal change needed to reach the LCS (predicted)")
    ax.set_ylabel("observed flip frequency"); ax.set_ylim(-0.03, 1.03)
    ax.set_title("Goal nudges: flips start where predicted")
    ax.legend(fontsize=8.5, loc="upper right")
    fig.suptitle("Locating the LCS of the generative flow from each seed turns FTLE into a quantitative switching margin",
                 fontweight="semibold")
    fig.savefig("figs/lcs_prediction.png", dpi=160); plt.close(fig)

    # timing figure
    Q = np.load("lcs/persist.npz")
    col, lab, mar, mnx, ft = Q["col"], Q["lab"], Q["margin"], Q["margin_next"], Q["ftle"]
    R_ = col.shape[1]
    valid = np.stack([(col[:, k] >= 0) & (col[:, k + 1] == col[:, k]) for k in range(R_ - 1)], 1)
    sw = np.stack([valid[:, k] & (lab[:, k + 1] != lab[:, k]) for k in range(R_ - 1)], 1)
    y = sw[valid]; m = mar[:, :-1][valid]; mn = mnx[:, :-1][valid]; f = ft[:, :-1][valid]
    fin = lambda v: np.where(np.isinf(v), 99, v)
    pred = mn <= 0
    res["persistent_seed"] = dict(
        steps=int(len(y)), switches=int(y.sum()),
        auroc_margin=auroc(-fin(m), y), auroc_moved_margin=auroc(-fin(mn), y), auroc_ftle=auroc(f, y),
        lcs_within_3_before_switch=float(np.isfinite(m[y]).mean()), lcs_within_3_otherwise=float(np.isfinite(m[~y]).mean()),
        rule_moved_margin_le_0=dict(predicted=int(pred.sum()), precision=float((pred & y).sum() / max(pred.sum(), 1)),
                                    recall=float((pred & y).sum() / max(y.sum(), 1))))
    json.dump(res, open("results_lcs.json", "w"), indent=1, default=float)
    t = res["timing"]; q = res["persistent_seed"]
    fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.6), constrained_layout=True)
    ax = axs[0]
    names = ["LCS mass from\nthe drawn seed", "FTLE of\nthe drawn seed", "Monte Carlo\n(256 fresh seeds)"]
    vals = [t["auroc_realized_lcs_mass"], t["auroc_realized_ftle"], t["auroc_realized_mc_now"]]
    ax.bar(range(3), vals, color=[BLUE, AQUA, MUTED], width=0.6)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=9)
    ax.axhline(0.5, color=MUTED, lw=1, ls=":"); ax.set_ylim(0.3, 1.0)
    ax.set_xticks(range(3)); ax.set_xticklabels(names, fontsize=9); ax.set_ylabel("AUROC")
    ax.set_title(f"Fresh noise every replan: will the next plan switch gap?\n({t['realized_switches']} switches in {t['n_realized']} replans)", fontsize=10.5)
    ax = axs[1]
    bins = np.r_[0, 0.1, 0.25, 0.5, 1, 2, 3, 100]
    lbl = ["<0.1", "0.1–0.25", "0.25–0.5", "0.5–1", "1–2", "2–3", "none ≤3"]
    h1 = np.histogram(fin(m[y]), bins)[0] / max(y.sum(), 1)
    h0 = np.histogram(fin(m[~y]), bins)[0] / max((~y).sum(), 1)
    xs = np.arange(len(lbl))
    ax.bar(xs - 0.2, 100 * h1, width=0.38, color=ORANGE, label=f"a switch follows ({q['switches']})")
    ax.bar(xs + 0.2, 100 * h0, width=0.38, color=MUTED, label=f"no switch follows ({q['steps'] - q['switches']})")
    ax.set_xticks(xs); ax.set_xticklabels(lbl, fontsize=8.5)
    ax.set_xlabel("seed's distance to the LCS at this replan"); ax.set_ylabel("% of replans")
    ax.set_title(f"Fixed seed reused every replan: the LCS approaches before a switch\n(AUROC {q['auroc_margin']:.2f}; FTLE {q['auroc_ftle']:.2f})", fontsize=10.5)
    ax.legend(fontsize=8.5)
    fig.savefig("figs/lcs_timing.png", dpi=160); plt.close(fig)


if __name__ == "__main__":
    analyze()
