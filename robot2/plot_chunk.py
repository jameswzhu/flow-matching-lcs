"""Analysis + figures for the own-chunk LCS study (chunk_eval.py analyze)."""
import json
import numpy as np
import matplotlib.pyplot as plt
from env2 import chunk_pending_label
from lcs import p_flip_noise
from plot_mc import auroc

SURF, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
EPS = [0.03, 0.05, 0.1, 0.2, 0.4]
TAU_GRID = [0.05, 0.08, 0.1, 0.15, 0.2, 0.3, 0.5]


def style():
    plt.rcParams.update({
        "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
        "text.color": INK, "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
        "axes.titlesize": 11, "axes.titleweight": "semibold", "legend.frameon": False, "lines.linewidth": 2})


def tie(margin):
    """Tiny tie-breaker so seeds with zero predicted probability are ranked by margin."""
    return -1e-9 * np.nan_to_num(margin, nan=99)


def skill(p, y):
    return float(1 - np.mean((p - y) ** 2) / max(np.mean((y.mean() - y) ** 2), 1e-12))


def analyze():
    P = np.load("mc/prep.npz"); Rf = np.load("lcs/refine.npz"); Sw = np.load("lcs/sweep.npz")
    D = np.load("chunk/dirs.npz"); G = np.load("chunk/goal.npz"); L = np.load("chunk/lf.npz")
    dp, dm, vel = Rf["d_plus"], Rf["d_minus"], Rf["vel"]
    N = len(dp); grp = np.repeat(np.arange(N // 24), 24)
    col, _, reached = chunk_pending_label(P["Sp"], P["a0"].astype(float))
    f1 = P["ftle1"]; sep = np.fmin(np.nan_to_num(dp, nan=np.inf), np.nan_to_num(dm, nan=np.inf))
    res = {}

    # ------------------------------------------------ 1. contexts
    groups = {"A, inside chunk": (col == 0) & reached, "A, beyond chunk end": (col == 0) & ~reached,
              "B, inside chunk": (col == 1) & reached, "B, beyond chunk end": (col == 1) & ~reached}
    res["contexts"] = {}
    for eps in [0.1, 0.4]:
        y = Sw[f"noise_{eps}"]; p = p_flip_noise(dp, dm, eps)
        res["contexts"][f"seed nudge {eps}"] = {g: dict(n=int(m.sum()), flips=int(y[m].sum()), observed=float(y[m].mean()),
                                                        predicted=float(p[m].mean()), auroc_lcs=auroc(p[m] + tie(sep)[m], y[m]),
                                                        auroc_ftle=auroc(f1[m], y[m])) for g, m in groups.items()}
    mg = np.fmin(np.where(np.isnan(dp), np.inf, dp / np.abs(vel[:, 0])), np.where(np.isnan(dm), np.inf, dm / np.abs(vel[:, 1])))
    for dg in [0.05, 0.2]:
        y = Sw[f"goal_{dg}"]; pr = mg <= dg
        res["contexts"][f"goal nudge {dg}"] = {g: dict(flips=int(y[m].sum()), precision=float((pr & y)[m].sum() / max(pr[m].sum(), 1)),
                                                       recall=float((pr & y)[m].sum() / max(y[m].sum(), 1))) for g, m in groups.items()}

    # ------------------------------------------------ 2. geometry: flatness and destination
    d, cross, ang = D["d"], D["cross"], D["angles"]
    flat = {}
    for a, th in enumerate(ang):
        c = np.cos(np.deg2rad(th)); pred = d / c; ok = np.isfinite(cross[:, a]) & (pred < 4)
        r = cross[ok, a] / pred[ok]
        flat[int(th)] = dict(n=int(ok.sum()), found=float(np.isfinite(cross[:, a]).mean()),
                             ratio_q=[float(v) for v in np.quantile(r, [.1, .25, .5, .75, .9])],
                             within_10pct=float(np.mean(np.abs(r - 1) < .1)),
                             destination_correct=float(np.mean(D["dest"][ok, a] == D["pred_dest"][ok])))
    res["flatness"] = dict(seeds=int(len(d)), by_angle=flat)
    pert = {}
    for e, ev in enumerate(D["eps"]):
        pred = (ev * D["proj"] > np.nan_to_num(D["dp"], nan=np.inf)[None]) | (-ev * D["proj"] > np.nan_to_num(D["dm"], nan=np.inf)[None])
        y = D["flip"][e]
        tp = int((pred & y).sum()); fp = int((pred & ~y).sum()); fn = int((~pred & y).sum())
        pert[str(float(ev))] = dict(perturbations=int(y.size), flips=int(y.sum()), tp=tp, fp=fp, fn=fn,
                                    precision=tp / max(tp + fp, 1), recall=tp / max(tp + fn, 1), accuracy=float(np.mean(pred == y)))
    res["per_perturbation"] = pert

    # ------------------------------------------------ 3. goal margins
    gm = {}
    for side, t, p in [("up", G["true_up"], G["pred_up"]), ("down", G["true_dn"], G["pred_dn"])]:
        both = np.isfinite(t) & np.isfinite(p); r = p[both] / t[both]; tb = t[both]
        gm[side] = {f"{lo}-{hi}": dict(n=int(((tb >= lo) & (tb < hi)).sum()),
                                       median_ratio=float(np.median(r[(tb >= lo) & (tb < hi)])),
                                       within_10pct=float(np.mean(np.abs(r[(tb >= lo) & (tb < hi)] - 1) < .1)))
                    for lo, hi in [(0, .05), (.05, .1), (.1, .2), (.2, .4), (.4, .6)]}
    tf = np.where(G["true_up"] < G["true_dn"], 1, -1); pf = np.where(G["pred_up"] < G["pred_dn"], 1, -1)
    m = np.isfinite(np.fmin(G["true_up"], G["true_dn"])) & np.isfinite(np.fmin(G["pred_up"], G["pred_dn"]))
    tm = np.fmin(G["true_up"], G["true_dn"]); pm = np.fmin(G["pred_up"], G["pred_dn"])
    miss = np.isfinite(tm) & (tm <= 0.2) & ~(pm <= 0.2)
    gm["direction_of_goal_change_correct"] = float(np.mean(tf[m] == pf[m]))
    gm["flips_within_0.2_missed"] = int(miss.sum())
    gm["missed_near_threshold_(pred<=0.25)"] = int((miss & (pm <= 0.25)).sum())
    gm["missed_no_LCS_along_normal"] = int((miss & ~np.isfinite(G["margin"])).sum())
    res["goal_margins"] = gm

    # ------------------------------------------------ 4. label-free LCS (chunk jump), tau chosen on even states
    train = grp % 2 == 0; test = ~train

    def lf(tau):
        return (np.where(L["jump_plus"] > tau, L["pos_plus"], np.nan), np.where(L["jump_minus"] > tau, L["pos_minus"], np.nan))

    y01 = Sw["noise_0.1"].astype(float)
    tr = {t: float(np.mean((p_flip_noise(*lf(t), 0.1)[train] - y01[train]) ** 2)) for t in TAU_GRID}
    tau = min(tr, key=tr.get); a, b = lf(tau); lfm = np.fmin(np.nan_to_num(a, nan=np.inf), np.nan_to_num(b, nan=np.inf))
    lab_free = dict(tau=tau, train_brier=tr, test={})
    for nm, m0 in [("all", np.ones(N, bool)), ("A pending", col == 0), ("B pending", col == 1)]:
        lab_free["test"][nm] = {}
        for eps in EPS:
            mm = m0 & test; y = Sw[f"noise_{eps}"][mm].astype(float)
            p = p_flip_noise(a, b, eps)[mm]; p2 = p_flip_noise(dp, dm, eps)[mm]
            lab_free["test"][nm][str(eps)] = dict(flips=int(y.sum()), observed=float(y.mean()),
                                                  label_free=dict(predicted=float(p.mean()), auroc=auroc(p + tie(lfm)[mm], y), brier_skill=skill(p, y)),
                                                  with_labels=dict(predicted=float(p2.mean()), auroc=auroc(p2 + tie(sep)[mm], y), brier_skill=skill(p2, y)))
    for side, dd in [("plus", dp), ("minus", dm)]:
        ok = ~np.isnan(dd) & ~np.isnan(L[f"pos_{side}"]) & (L[f"jump_{side}"] > tau)
        lab_free[f"locate_error_median_{side}"] = float(np.median(np.abs(L[f"pos_{side}"][ok] - dd[ok])))
    hard_det = int(((L["jump_plus"] > tau) & ~np.isnan(dp)).sum() + ((L["jump_minus"] > tau) & ~np.isnan(dm)).sum())
    spurious = int(((L["jump_plus"] > tau) & np.isnan(dp)).sum() + ((L["jump_minus"] > tau) & np.isnan(dm)).sum())
    lab_free.update(boundaries_detected_with_label_change=hard_det, detected_without_label_change=spurious)
    jm = np.r_[L["jump_plus"][~np.isnan(dp)], L["jump_minus"][~np.isnan(dm)]]
    cc = np.r_[col[~np.isnan(dp)], col[~np.isnan(dm)]]; rr = np.r_[reached[~np.isnan(dp)], reached[~np.isnan(dm)]]
    ctx = {"A": cc == 0, "B, inside chunk": (cc == 1) & rr, "B, beyond chunk end": (cc == 1) & ~rr}
    lab_free["soft_boundaries"] = {k: dict(separatrices=int(m.sum()), soft=int((jm[m] <= tau).sum())) for k, m in ctx.items()}
    res["label_free"] = lab_free
    json.dump(res, open("results_chunk.json", "w"), indent=1, default=float)
    figs(res, dict(D=D, G=G, jm=jm, ctx=ctx, J=np.load("chunk/lf_jumps.npz")["jumps"]))
    print(json.dumps({k: res[k] for k in ["per_perturbation", "goal_margins"]}, indent=1, default=float))
    return res


def figs(res, X):
    style()
    D, G = X["D"], X["G"]
    fig, axs = plt.subplots(1, 3, figsize=(16.5, 4.8), constrained_layout=True)
    # (a) flatness: actual / flat-predicted crossing distance by angle
    ax = axs[0]
    fl = res["flatness"]["by_angle"]; th = sorted(fl)
    q = np.array([fl[t]["ratio_q"] for t in th]); xs = np.arange(len(th))
    ax.vlines(xs, q[:, 0], q[:, 4], color=BLUE, lw=1.2)
    ax.vlines(xs, q[:, 1], q[:, 3], color=BLUE, lw=7)
    ax.scatter(xs, q[:, 2], color=SURF, edgecolor=BLUE, zorder=3, s=30, lw=1.5)
    ax.axhline(1, color=MUTED, lw=1, ls=":")
    ax.set_xticks(xs); ax.set_xticklabels([f"{t}°\n{100 * fl[t]['within_10pct']:.0f}%" for t in th], fontsize=8.5)
    ax.set_ylim(0.6, 1.4)
    ax.set_xlabel("angle of the nudge to the LCS normal\n(% of crossings within 10% of the flat prediction)")
    ax.set_ylabel("actual / predicted crossing distance")
    ax.set_title("The LCS is locally flat: crossing distance = d / cos θ\n(bars: 10–90% and 25–75%; destination mode correct in 100%)", fontsize=10.5)
    # (b) goal margins: predicted vs exact
    ax = axs[1]
    for t, p, c, lab in [(G["true_up"], G["pred_up"], BLUE, "goal raised"), (G["true_dn"], G["pred_dn"], ORANGE, "goal lowered")]:
        ok = np.isfinite(t) & np.isfinite(p)
        ax.scatter(t[ok], p[ok], s=7, color=c, alpha=.55, lw=0, label=lab)
    lim = [3e-3, 0.7]
    ax.plot(lim, lim, color=MUTED, lw=1, ls=":")
    ax.fill_between(lim, [v * 0.9 for v in lim], [v * 1.1 for v in lim], color=MUTED, alpha=.15, lw=0)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("exact goal change that switches the chunk (bisection)")
    ax.set_ylabel("predicted from the LCS (first order)")
    ax.set_title("Goal margins: exact up to ±0.2, drifting beyond\n(shaded ±10%)", fontsize=10.5)
    ax.legend(fontsize=8.5, loc="upper left", markerscale=2)
    # (c) context: AUROC LCS vs FTLE at eps 0.1 and 0.4
    ax = axs[2]
    groups = list(res["contexts"]["seed nudge 0.1"])
    w = 0.2
    for k, (eps, key, c, lab) in enumerate([(0.1, "auroc_lcs", BLUE, "LCS margin, ε = 0.1"), (0.1, "auroc_ftle", MUTED, "FTLE, ε = 0.1"),
                                            (0.4, "auroc_lcs", AQUA, "LCS margin, ε = 0.4"), (0.4, "auroc_ftle", "#c3c2b7", "FTLE, ε = 0.4")]):
        v = [res["contexts"][f"seed nudge {eps}"][g][key] for g in groups]
        ax.bar(np.arange(len(groups)) + (k - 1.5) * w, v, width=w - 0.03, color=c, label=lab)
    fl01 = [res["contexts"]["seed nudge 0.1"][g]["flips"] for g in groups]
    ax.set_xticks(range(len(groups))); ax.set_xticklabels([f"{g.replace(', ', chr(10))}\n{f} flips @0.1" for g, f in zip(groups, fl01)], fontsize=8.3)
    ax.set_ylim(0.75, 1.005); ax.set_ylabel("AUROC"); ax.axhline(1, color=MUTED, lw=.6)
    ax.set_title("Every context: the margin holds where FTLE fails\n(A = 3-way choice, B = 2-way)", fontsize=10.5)
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2)
    fig.suptitle("How well the LCS predicts a mode switch of its own chunk", fontweight="semibold")
    fig.savefig("figs/chunk_lcs.png", dpi=160); plt.close(fig)

    # label-free figure
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.7), constrained_layout=True)
    ax = axs[0]
    bins = np.logspace(-2.2, 1, 40)
    Jr = X["J"].ravel()
    ax.hist(Jr, bins=bins, weights=np.full(len(Jr), 1 / len(Jr)), color="#d6d4cd", label="every 0.1 step along the normal")
    for (k, m), c in zip(X["ctx"].items(), [BLUE, AQUA, ORANGE]):
        v = X["jm"][m]
        ax.hist(v, bins=bins, weights=np.full(len(v), 1 / len(v)), histtype="step", lw=2, color=c,
                label=f"across a separatrix: {k} ({m.sum()})")
    tau = res["label_free"]["tau"]
    ax.axvline(tau, color=INK2, lw=1, ls=":"); ax.text(tau * 1.08, 0.02, f"threshold τ = {tau}", fontsize=8.5, color=INK2, rotation=90, va="bottom")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(1e-3, 1)
    ax.set_xlabel("jump in the chunk across a 0.1 step (normalised action units)"); ax.set_ylabel("fraction within each group")
    ax.set_title("Hard vs soft boundaries: every 3-way (A) boundary makes the chunk jump;\nmost 2-way (B) boundaries change the label without one", fontsize=10.5)
    ax.legend(fontsize=8, loc="upper right")
    ax = axs[1]
    T = res["label_free"]["test"]
    E2 = [0.1, 0.2, 0.4]
    for nm, c in [("A pending", BLUE), ("B pending", ORANGE)]:
        ax.plot(E2, [T[nm][str(e)]["with_labels"]["auroc"] for e in E2], color=c, marker="o", ms=4, label=f"{nm}: separatrix (needs labels)")
        ax.plot(E2, [T[nm][str(e)]["label_free"]["auroc"] for e in E2], color=c, marker="x", ms=5, ls=(0, (4, 3)), label=f"{nm}: label-free LCS")
        for e in E2:
            ax.text(e, T[nm][str(e)]["label_free"]["auroc"] - 0.018, f"{T[nm][str(e)]['flips']}", fontsize=7.5, color=c, ha="center")
    ax.set_xscale("log"); ax.set_xticks(E2); ax.set_xticklabels([f"{e:g}" for e in E2]); ax.minorticks_off()
    ax.set_ylim(0.6, 1.01); ax.set_xlabel("size ε of the seed nudge (numbers: held-out flips)"); ax.set_ylabel("AUROC (held-out states)")
    ax.set_title("Without labels: as good as with them for the 3-way choice,\nweaker where boundaries are soft", fontsize=10.5)
    ax.legend(fontsize=8, loc="lower left")
    fig.savefig("figs/chunk_lcs_labelfree.png", dpi=160); plt.close(fig)


if __name__ == "__main__":
    analyze()
