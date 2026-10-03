"""Analysis + figure for the multi-replan (composed) FTLE study."""
import json
import numpy as np
import matplotlib.pyplot as plt

SURF, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
BLUE, ORANGE, AQUA, YELLOW, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#4a3aa7"
REGIONS = {"all states": (-9, 9), "x < −2.2\nA being chosen": (-9, -2.2),
           "−2.2…−1.5\nB beyond chunk": (-2.2, -1.5), "−1.5…0.5\nB inside chunk": (-1.5, 0.5)}


def auroc(score, y):
    score = np.asarray(score, float); y = np.asarray(y, bool)
    n1, n0 = y.sum(), (~y).sum()
    if n1 == 0 or n0 == 0:
        return np.nan
    _, inv, cnt = np.unique(score, return_inverse=True, return_counts=True)
    r = np.argsort(np.argsort(score, kind="mergesort"), kind="mergesort") + 1.0
    r = (np.bincount(inv, weights=r) / cnt)[inv]
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def within_auroc(score, y, grp):
    U = P = 0.0
    for g in np.unique(grp):
        m = grp == g
        if y[m].any() and (~y[m]).any():
            n = y[m].sum() * (~y[m]).sum(); U += auroc(score[m], y[m]) * n; P += n
    return U / P if P else np.nan


def boot(score, y, grp, n=300, seed=0):
    rng = np.random.default_rng(seed); ug = np.unique(grp)
    idx = {g: np.where(grp == g)[0] for g in ug}; v = []
    for _ in range(n):
        ii = np.concatenate([idx[g] for g in rng.choice(ug, len(ug))])
        v.append(auroc(score[ii], y[ii]))
    v = np.array(v); v = v[~np.isnan(v)]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if len(v) > 20 else [np.nan, np.nan]


def analyze():
    P = np.load("mc/prep.npz"); S3 = np.load("s3_rows.npz"); L = np.load("mc/labels.npz")
    x = P["Sp"][:, 0]; N = len(x); grp = np.repeat(np.arange(N // 24), 24)
    real = {k: np.load(f"mc/{k}.npz") for k in ["oracle", "zero", "rand1", "rand2"]}
    base_st = np.load("mc/base_states.npy")
    replay_err = float(np.abs(real["oracle"]["states"][:, :base_st.shape[1]] - base_st).max())
    b, gp_, gm_ = L["base"], L["gp"], L["gm"]
    cl_goal = (gp_[:, 2] != b[:, 2]) | (gm_[:, 2] != b[:, 2])
    goal_A = (gp_[:, 0] != b[:, 0]) | (gm_[:, 0] != b[:, 0])
    goal_Bonly = cl_goal & ~goal_A
    noise_Bonly = S3["cl_noise"] & ~S3["cl_noise_A"]
    targets = {"goal flip (any)": cl_goal, "goal flip, A changes": goal_A, "goal flip, only B changes": goal_Bonly,
               "seed flip (any)": S3["cl_noise"], "seed flip, only B changes": noise_Bonly}
    rand = {k: 0.5 * (real["rand1"][k] + real["rand2"][k]) for k in ["goal_chunkmax", "seed_chunkmax", "goal_cross", "seed_cross"]}
    def state_max(v):                      # one score per state: max over its 24 seeds x 2 futures
        out = np.empty_like(v)
        for g in np.unique(grp):
            m = grp == g; out[m] = v[m].max()
        return out
    rmax = {k: np.maximum(real["rand1"][k], real["rand2"][k]) for k in ["goal_chunkmax", "seed_chunkmax"]}
    # composed score: the largest total sensitivity of any chunk generated along the future path
    # while a decision is still pending (includes the unexecuted tail of each chunk)
    goal_scores = {"one chunk  log‖∂Φ/∂g‖": S3["gsens"],
                   "composed, zero future noise": real["zero"]["goal_chunkmax"],
                   "composed, random future noise (per seed)": rand["goal_chunkmax"],
                   "composed, random futures → state risk (max of 48 paths)": state_max(rmax["goal_chunkmax"]),
                   "composed, actual future noise": real["oracle"]["goal_chunkmax"],
                   "variant: y-at-column sensitivity (actual noise)": real["oracle"]["goal_cross"],
                   "variant: whole-trajectory norm (actual noise)": real["oracle"]["goal_stack"],
                   "baseline: one-chunk state entropy": S3["state_H"]}
    seed_scores = {"one chunk FTLE": P["ftle1"],
                   "composed, zero future noise": real["zero"]["seed_chunkmax"],
                   "composed, random future noise (per seed)": rand["seed_chunkmax"],
                   "composed, random futures → state risk (max of 48 paths)": state_max(rmax["seed_chunkmax"]),
                   "composed, actual future noise": real["oracle"]["seed_chunkmax"],
                   "variant: y-at-column sensitivity (actual noise)": real["oracle"]["seed_cross"],
                   "variant: whole-trajectory norm (actual noise)": real["oracle"]["seed_stack"],
                   "baseline: one-chunk state entropy": S3["state_H"]}
    res = dict(replay_max_state_diff=replay_err,
               goal_flip_labels_match_S3=float((cl_goal == S3["cl_goal"]).mean()), table={})
    for tn, y in targets.items():
        sc = goal_scores if tn.startswith("goal") else seed_scores
        res["table"][tn] = {}
        for rn, (lo, hi) in REGIONS.items():
            m = (x >= lo) & (x < hi)
            row = dict(positives=int(y[m].sum()), rows=int(m.sum()))
            for sn, s in sc.items():
                row[sn] = dict(auroc=auroc(s[m], y[m]), ci=boot(s[m], y[m], grp[m]),
                               within=within_auroc(s[m], y[m], grp[m]))
            res["table"][tn][rn] = row
    # which part of the horizon carries the goal sensitivity?  (median |d s_k/dg| by replan, actual noise)
    gk = real["oracle"]["goal_chunk_k"]
    res["argmax_replan_of_goal_score"] = np.bincount(np.argmax(gk, 1), minlength=gk.shape[1]).tolist()
    json.dump(res, open("results_mc.json", "w"), indent=1, default=float)
    for tn in targets:
        print(tn)
        for rn in REGIONS:
            row = res["table"][tn][rn]
            print(f"   {rn.splitlines()[0]:18s} n+={row['positives']:4d} " + "  ".join(
                f"{v['auroc']:.3f}/{v['within']:.3f}" for k, v in row.items() if isinstance(v, dict)))
    print("replay diff", replay_err, "goal labels match S3", res["goal_flip_labels_match_S3"])
    plot(res, x, cl_goal, goal_A, goal_Bonly)
    return res


def plot(res, x, cl_goal, goal_A, goal_Bonly, path="figs/mc_composed.png"):
    plt.rcParams.update({
        "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
        "text.color": INK, "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
        "axes.titlesize": 11, "axes.titleweight": "semibold", "legend.frameon": False, "lines.linewidth": 2})
    cols = [MUTED, AQUA, BLUE, ORANGE, VIOLET]
    fig = plt.figure(figsize=(17, 6.0), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 0.13], width_ratios=[1.35, 1.35, 0.9])
    axs = [fig.add_subplot(gs[0, k]) for k in range(3)]
    for ax, tn, ttl in [(axs[0], "goal flip (any)", "Goal nudged for the episode: does the route flip?"),
                        (axs[1], "goal flip, only B changes", "…only the later column-B choice flips")]:
        tab = res["table"][tn]; rns = list(REGIONS); w = 0.16
        names = [k for k, v in tab["all states"].items() if isinstance(v, dict) and not k.startswith(("variant", "baseline"))]
        for k, sn in enumerate(names):
            v = np.array([tab[r][sn]["auroc"] for r in rns]); ci = np.array([tab[r][sn]["ci"] for r in rns])
            xs = np.arange(len(rns)) + (k - 2) * w
            ax.bar(xs, v, width=w - 0.03, color=cols[k], label=sn)
            ok = ~np.isnan(ci[:, 0])
            ax.errorbar(xs[ok], v[ok], yerr=[v[ok] - ci[ok, 0], ci[ok, 1] - v[ok]], fmt="none", ecolor=INK2, lw=1, capsize=2)
        ax.axhline(0.5, color=MUTED, lw=1, ls=":")
        ax.set_xticks(range(len(rns)))
        ax.set_xticklabels([f"{r}\n{tab[r]['positives']} flips" for r in rns], fontsize=8.3)
        ax.set_ylim(0.3, 1.0); ax.set_ylabel("AUROC"); ax.set_title(ttl, fontsize=10.5)
    ax = axs[2]
    bins = np.linspace(-2.5, 0.5, 13); xc = 0.5 * (bins[1:] + bins[:-1]); bi = np.digitize(x, bins) - 1
    for y, c, lab in [(goal_A, BLUE, "A choice changes"), (goal_Bonly, ORANGE, "only B choice changes")]:
        ax.plot(xc, [100 * y[bi == k].mean() if (bi == k).any() else np.nan for k in range(len(xc))], color=c, marker="o", ms=4, label=lab)
    for c_ in [-2.2, -1.5]:
        ax.axvline(c_, color=MUTED, lw=1, ls=":")
    ax.set_xlabel("robot x at the evaluated state"); ax.set_ylabel("% of seeds whose route flips")
    ax.set_title("Which decision flips, by position", fontsize=10.5); ax.legend(fontsize=8.5)
    lax = fig.add_subplot(gs[1, :2]); lax.axis("off")
    h, l = axs[0].get_legend_handles_labels(); lax.legend(h, l, ncol=3, fontsize=8.8, loc="center")
    fig.suptitle("Composing flow-map Jacobians across replans: predicting route flips when the goal is nudged for a whole episode "
                 "(95% cluster-bootstrap CIs)", fontweight="semibold", fontsize=11)
    fig.savefig(path, dpi=160); plt.close(fig)


if __name__ == "__main__":
    analyze()
