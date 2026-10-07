"""
LCS of pi0's first-chunk flow in noise space: distances to the mode boundary from each
seed, and how well they predict mode switches (robot2/lcs_eval.py + chunk_eval.py,
ported to pi0).

    python lcs4.py <policy> scan     # FTLE, top right singular vector v1, line search along
                                     # z + s v1 (s in [-4, 4], step 0.1), bisection of the
                                     # nearest label change on each side       (~50 min)
    python lcs4.py <policy> nudge    # random nudges of several sizes: observed flips vs
                                     # the flat-LCS prediction (no fitting)     (~15 min)
    python lcs4.py <policy> plot     # figures + out/<policy>/lcs.json

Label = kind of the object the first chunk heads for (chunk_target), at the start state.
Margins d+ / d- are distances along v1 to the nearest label change on each side (nan if
none within 4). Label-free check: the largest chunk jump between grid points (the ridge)
vs the bisected separatrix. 30 scenes x 16 seeds per prompt, every prompt.
"""
import json, os, sys, time
import numpy as np
import torch
from scipy.stats import beta
from env4 import ALL_PROMPTS, TRAIN_PROMPTS, KIND_NAMES, H, make_scenes, chunk_target, kind_of
from pi0flow import load, ftle, D_ACT

NAME = sys.argv[1]
STEPS = sys.argv[2:]
OUT = os.path.join("out", NAME + os.environ.get("LCS_TAG", ""))      # LCS_TAG: smoke tests
os.makedirs(OUT, exist_ok=True); os.makedirs("figs", exist_ok=True)
pol = load(f"policies/{NAME}.pt") if any(s in STEPS for s in ("scan", "nudge")) else None
PROMPTS = os.environ["LCS_PROMPTS"].split(",") if "LCS_PROMPTS" in os.environ else list(ALL_PROMPTS)
A = H * D_ACT
N_SC, N_SEED = int(os.environ.get("LCS_NSC", 30)), 16
S_GRID = np.round(np.arange(-40, 41) * 0.1, 10)
I0 = 40
N_BISECT = 14
EPS = [0.5, 1.0, 2.0, 4.0, 8.0]
N_DIR = 8
ZERO = np.zeros((1, 2))
t00 = time.time()


def log(*a):
    print(f"[{time.time() - t00:6.0f}s]", *a, flush=True)


def kinds(z, c, objs1):
    """Kind (0..3) of the object each seed's first chunk heads for. objs1 [1, 4, 4]."""
    a = pol.flow_np(z, c, bs=2048)
    slot = chunk_target(np.zeros((len(z), 2)), a, np.repeat(objs1, len(z), 0))
    return kind_of(objs1[0][slot]), a


def scenes():
    return make_scenes(N_SC, np.random.default_rng(400))


# ----------------------------------------------------------------------------- scan
if "scan" in STEPS:
    objs, R = scenes()
    Z = np.random.default_rng(401).standard_normal((N_SC, N_SEED, A))
    res = {}
    for p in PROMPTS:
        F, V1, L, JUMP, DP, DM, DESTP, DESTM, HARDP, HARDM, SLIVER = ([] for _ in range(11))
        for i in range(N_SC):
            o1 = objs[i:i + 1]
            with torch.no_grad():
                c = pol.cond(ZERO, o1, p)
            _, J = pol.jac(Z[i], c)
            _, _, Vt = np.linalg.svd(J, full_matrices=False)
            v1 = Vt[:, 0]                                                      # [seed, A]
            zz = Z[i][:, None, :] + S_GRID[None, :, None] * v1[:, None, :]    # [seed, S, A]
            lab, a = kinds(zz.reshape(-1, A), c, o1)
            lab = lab.reshape(N_SEED, len(S_GRID)); a = a.reshape(N_SEED, len(S_GRID), -1)
            jump = np.linalg.norm(np.diff(a, axis=1), axis=-1)                # [seed, S-1]
            l0 = lab[:, I0]
            # nearest label change on each side -> bracket [inside, outside] in s
            br = np.full((N_SEED, 2, 2), np.nan)                              # [seed, side(+,-), (in, out)]
            for n in range(N_SEED):
                d = np.where(lab[n] != l0[n])[0]
                up, dn = d[d > I0], d[d < I0]
                if len(up):
                    br[n, 0] = S_GRID[up[0] - 1], S_GRID[up[0]]
                if len(dn):
                    br[n, 1] = S_GRID[dn[-1] + 1], S_GRID[dn[-1]]
            # bisection (both sides, all seeds at once)
            lo, hi = br[..., 0].copy(), br[..., 1].copy()
            ok = ~np.isnan(lo)
            idx = np.argwhere(ok)
            for _ in range(N_BISECT):
                if not len(idx):
                    break
                mid = 0.5 * (lo[ok] + hi[ok])
                lm, _ = kinds(Z[i][idx[:, 0]] + mid[:, None] * v1[idx[:, 0]], c, o1)
                same = lm == l0[idx[:, 0]]
                lo[ok] = np.where(same, mid, lo[ok]); hi[ok] = np.where(same, hi[ok], mid)
            dest = np.full((N_SEED, 2), -1); hard = np.full((N_SEED, 2), np.nan)
            if len(idx):
                # chunk jump across a 0.1 window centred on the boundary (hard vs soft, as robot2)
                sg = np.sign(hi[ok] - lo[ok])
                zin = Z[i][idx[:, 0]] + (lo[ok] - 0.05 * sg)[:, None] * v1[idx[:, 0]]
                zout = Z[i][idx[:, 0]] + (hi[ok] + 0.05 * sg)[:, None] * v1[idx[:, 0]]
                lout, aout = kinds(zout, c, o1); _, ain = kinds(zin, c, o1)
                dest[ok] = kinds(Z[i][idx[:, 0]] + hi[ok][:, None] * v1[idx[:, 0]], c, o1)[0]
                hard[ok] = np.linalg.norm(aout - ain, axis=1)                 # chunk jump across the boundary
            sep = 0.5 * (lo + hi)
            # destination basin: first label beyond the boundary that persists for >= 3 grid
            # points (0.3). Between two basins the chunk can sweep through the heading of an
            # object lying between them, leaving a thin sliver labelled with that object.
            stable = np.full((N_SEED, 2), -1)
            for n in range(N_SEED):
                for side, order in ((0, range(I0 + 1, len(S_GRID) - 2)), (1, range(I0 - 1, 1, -1))):
                    for j in order:
                        run = lab[n, j:j + 3] if side == 0 else lab[n, j - 2:j + 1]
                        if run[0] != l0[n] and np.all(run == run[0]):
                            stable[n, side] = run[0]; break
            F.append(ftle(J)); V1.append(v1.astype(np.float32)); L.append(l0); JUMP.append(jump.astype(np.float32))
            DP.append(sep[:, 0]); DM.append(-sep[:, 1]); DESTP.append(stable[:, 0]); DESTM.append(stable[:, 1])
            SLIVER.append(ok & (dest != stable))
            HARDP.append(hard[:, 0]); HARDM.append(hard[:, 1])
        st = lambda x: np.stack(x)
        res[p] = dict(ftle=st(F), v1=st(V1), label=st(L), jump=st(JUMP), d_plus=st(DP), d_minus=st(DM),
                      dest_plus=st(DESTP), dest_minus=st(DESTM), hard_plus=st(HARDP), hard_minus=st(HARDM),
                      sliver=st(SLIVER))
        m = np.fmin(res[p]["d_plus"], res[p]["d_minus"])
        log(f"scan {p!r:18} boundary within 4: {np.mean(~np.isnan(m)):.2f} | median margin "
            f"{np.nanmedian(m) if np.any(~np.isnan(m)) else np.nan:.2f}")
        np.savez_compressed(os.path.join(OUT, "lcs_scan.npz"), prompts=np.array([q for q in res]), objs=objs, Z=Z,
                            s_grid=S_GRID, **{f"{k}__{q}": v for q, d in res.items() for k, v in d.items()})


def load_scan():
    D = np.load(os.path.join(OUT, "lcs_scan.npz"))
    ps = D["prompts"].tolist()
    keys = ["ftle", "v1", "label", "jump", "d_plus", "d_minus", "dest_plus", "dest_minus", "hard_plus", "hard_minus",
            "sliver"]
    return D, ps, {p: {k: D[f"{k}__{p}"] for k in keys} for p in ps}


def p_cross(margin, eps, D=A):
    """P(eps * (u . n) > margin) for u uniform on the unit sphere in R^D."""
    m = np.asarray(margin, float)
    a = np.clip(m / eps, 0, 1)
    return np.where(np.isnan(m), 0.0, np.where(m <= 0, 0.5, 0.5 * beta.sf(a ** 2, 0.5, (D - 1) / 2)))


# ----------------------------------------------------------------------------- nudge
if "nudge" in STEPS:
    D, ps, S = load_scan()
    objs, Z = D["objs"], D["Z"]
    rng = np.random.default_rng(402)
    out = {}
    for p in ps:
        s = S[p]
        flip = np.zeros((len(EPS), N_SC, N_SEED, N_DIR), bool); comp = np.zeros_like(flip, dtype=float)
        dest = np.full(flip.shape, -1)
        for i in range(N_SC):
            with torch.no_grad():
                c = pol.cond(ZERO, objs[i:i + 1], p)
            U = rng.standard_normal((len(EPS), N_SEED, N_DIR, A)); U /= np.linalg.norm(U, axis=-1, keepdims=True)
            zz = Z[i][None, :, None, :] + np.array(EPS)[:, None, None, None] * U
            lab, _ = kinds(zz.reshape(-1, A), c, objs[i:i + 1])
            lab = lab.reshape(len(EPS), N_SEED, N_DIR)
            flip[:, i] = lab != s["label"][i][None, :, None]
            dest[:, i] = lab
            comp[:, i] = np.array(EPS)[:, None, None] * np.einsum("esda,sa->esd", U, s["v1"][i].astype(float))
        out[p] = dict(flip=flip, comp=comp, dest=dest)
        log(f"nudge {p!r:18} flip rate by eps: " + " ".join(f"{e}:{flip[k].mean():.3f}" for k, e in enumerate(EPS)))
    np.savez_compressed(os.path.join(OUT, "lcs_nudge.npz"), prompts=np.array(ps), eps=np.array(EPS),
                        **{f"{k}__{q}": v for q, d in out.items() for k, v in d.items()})


# ----------------------------------------------------------------------------- analysis + plot
def auroc(score, y):
    from scipy.stats import rankdata
    y = np.asarray(y, bool)
    if y.all() or (~y).all():
        return np.nan
    r = rankdata(score)
    return (r[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum())


def auroc_ci(score, y, cluster, n_boot=1000, seed=0):
    """AUROC overall, its 95% bootstrap CI clustered by scene, and the mean within-scene AUROC."""
    rng = np.random.default_rng(seed)
    cl = np.unique(cluster); groups = [np.where(cluster == g)[0] for g in cl]
    boots = []
    for _ in range(n_boot):
        ii = np.concatenate([groups[j] for j in rng.integers(len(groups), size=len(groups))])
        boots.append(auroc(score[ii], y[ii]))
    within = [auroc(score[g], y[g]) for g in groups]
    return (auroc(score, y), *np.nanquantile(boots, [0.025, 0.975]), np.nanmean(within))


if "plot" in STEPS:
    import matplotlib.pyplot as plt
    D, ps, S = load_scan()
    N = np.load(os.path.join(OUT, "lcs_nudge.npz")) if os.path.exists(os.path.join(OUT, "lcs_nudge.npz")) else None
    SURF, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
    CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#c4359b"]
    plt.rcParams.update({
        "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
        "text.color": INK, "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
        "axes.titlesize": 11, "axes.titleweight": "semibold", "legend.frameon": False})
    summ = {"per_prompt": {}, "nudge": {}}
    for p in ps:
        s = S[p]; m = np.fmin(s["d_plus"], s["d_minus"])
        hard = np.concatenate([s["hard_plus"].ravel(), s["hard_minus"].ravel()])
        summ["per_prompt"][p] = dict(
            n_match=len(ALL_PROMPTS[p]), trained=p in TRAIN_PROMPTS,
            boundary_within_4=float(np.mean(~np.isnan(m))), boundary_within_1=float(np.mean(m <= 1)),
            median_margin=float(np.nanmedian(m)) if np.any(~np.isnan(m)) else None,
            frac_ftle_pos=float(np.mean(s["ftle"] > 0)),
            sliver_frac=float(s["sliver"].sum() / max((~np.isnan(np.stack([s["d_plus"], s["d_minus"]], -1))).sum(), 1)),
            hard_jump_median=float(np.nanmedian(hard)) if np.any(~np.isnan(hard)) else None)

    # label-free ridge (largest chunk jump on that side) vs separatrix, for boundaries within 0.5
    gaps = []
    for p in ps:
        s = S[p]
        for side, key in ((0, "d_plus"), (1, "d_minus")):
            d = s[key]
            sel = np.argwhere(d <= 0.5)
            for i, n in sel:
                j = s["jump"][i, n]
                rng_ = np.arange(I0, len(S_GRID) - 1) if side == 0 else np.arange(0, I0)
                k = rng_[np.argmax(j[rng_])]
                ridge = 0.5 * (S_GRID[k] + S_GRID[k + 1])
                gaps.append(abs(abs(ridge) - d[i, n]))
    gaps = np.array(gaps)
    summ["ridge_vs_separatrix"] = dict(n=int(len(gaps)), within_0p1=float(np.mean(gaps <= 0.1 + 1e-9)),
                                      median_gap=float(np.median(gaps)) if len(gaps) else None)

    # destination named by the boundary: the label beyond it vs the label a nudge lands in
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4), layout="constrained")
    # (a) margin distribution by prompt group
    groups = [("1 match (trained)", [p for p in ps if len(ALL_PROMPTS[p]) == 1 and p in TRAIN_PROMPTS]),
              ("2 matches (trained)", [p for p in ps if len(ALL_PROMPTS[p]) == 2 and p in TRAIN_PROMPTS]),
              ("\"shape\"", ["shape"]),
              ("paraphrases", [p for p in ps if p not in TRAIN_PROMPTS])]
    bins = np.linspace(0, 4, 41)
    for g, (name, pl) in enumerate(groups):
        pl = [q for q in pl if q in ps]
        if not pl:
            continue
        m = np.concatenate([np.fmin(S[p]["d_plus"], S[p]["d_minus"]).ravel() for p in pl])
        h = np.histogram(m[~np.isnan(m)], bins)[0] / len(m)
        ax[0].step(bins[:-1], np.cumsum(h), where="post", color=CAT[g], lw=2,
                   label=f"{name}: {np.mean(~np.isnan(m)):.0%} within 4")
    ax[0].set_xlabel("margin to nearest LCS along v1 (noise units)"); ax[0].set_ylabel("fraction of seeds (cumulative)")
    ax[0].set_title("distance from each seed to the mode boundary"); ax[0].legend(fontsize=8.5)

    if N is not None:
        eps = N["eps"]
        rows = []
        for k, e in enumerate(eps):
            y, pred_rule, p_an, ft, cl = [], [], [], [], []
            n_dest, n_dest_ok = 0, 0
            for p in ps:
                s = S[p]; fl = N[f"flip__{p}"][k]; cp = N[f"comp__{p}"][k]
                dp = np.nan_to_num(s["d_plus"], nan=np.inf)[..., None]; dm = np.nan_to_num(s["d_minus"], nan=np.inf)[..., None]
                y.append(fl.ravel()); pred_rule.append(((cp > dp) | (cp < -dm)).ravel())
                # destination: the label beyond the boundary on the side the nudge crossed
                cross = fl & ((cp > dp) | (cp < -dm))
                want = np.where(cp > 0, s["dest_plus"][..., None], s["dest_minus"][..., None])
                n_dest += cross.sum(); n_dest_ok += (cross & (N[f"dest__{p}"][k] == want)).sum()
                pa = np.clip(p_cross(s["d_plus"], e) + p_cross(s["d_minus"], e), 0, 1)
                p_an.append(np.repeat(pa[..., None], N_DIR, -1).ravel())
                ft.append(np.repeat(s["ftle"][..., None], N_DIR, -1).ravel())
                cl.append(np.repeat(np.repeat(np.arange(N_SC)[:, None], N_SEED, 1)[..., None], N_DIR, -1).ravel())
            y, pr, pa, ft, cl = map(np.concatenate, (y, pred_rule, p_an, ft, cl))
            tp = (pr & y).sum()
            a_l = auroc_ci(pa, y, cl); a_f = auroc_ci(ft, y, cl)
            rows.append(dict(eps=float(e), n=int(len(y)), flips=int(y.sum()), observed=float(y.mean()),
                             predicted=float(pa.mean()), rule_precision=float(tp / max(pr.sum(), 1)),
                             rule_recall=float(tp / max(y.sum(), 1)), rule_accuracy=float((pr == y).mean()),
                             dest_correct=float(n_dest_ok / max(n_dest, 1)), n_dest=int(n_dest),
                             auroc_lcs=a_l, auroc_ftle=a_f,
                             brier_skill=float(1 - np.mean((pa - y) ** 2) / np.mean((y.mean() - y) ** 2))))
        summ["nudge"] = rows
        x = np.arange(len(eps))
        ax[1].plot(x, [r["observed"] for r in rows], "o-", color=CAT[0], lw=2, label="observed")
        ax[1].plot(x, [r["predicted"] for r in rows], "s--", color=CAT[1], lw=2, label="LCS, analytic (no fit)")
        ax[1].set_xticks(x); ax[1].set_xticklabels([f"{e:g}" for e in eps]); ax[1].set_yscale("log")
        ax[1].set_xlabel("nudge size ε (384-D, random direction)"); ax[1].set_ylabel("flip rate")
        ax[1].set_title("flip rate: observed vs predicted"); ax[1].legend()
        for key, col, lab in (("auroc_lcs", CAT[0], "LCS margin"), ("auroc_ftle", CAT[1], "FTLE")):
            v = np.array([r[key] for r in rows])
            ax[2].errorbar(x, v[:, 0], yerr=[v[:, 0] - v[:, 1], v[:, 2] - v[:, 0]], fmt="o-", color=col, lw=2,
                           capsize=3, label=lab)
        ax[2].set_xticks(x); ax[2].set_xticklabels([f"{e:g}" for e in eps]); ax[2].set_ylim(0.5, 1.0)
        ax[2].set_xlabel("nudge size ε"); ax[2].set_ylabel("AUROC (95% CI, clustered by scene)")
        ax[2].set_title("which perturbations flip the mode"); ax[2].legend()
    fig.savefig(f"figs/lcs_{NAME}.png", dpi=150)
    json.dump(summ, open(os.path.join(OUT, "lcs.json"), "w"), indent=1)
    log(f"figs/lcs_{NAME}.png, {OUT}/lcs.json")
    print(json.dumps({k: summ[k] for k in ("ridge_vs_separatrix", "nudge")}, indent=1))
