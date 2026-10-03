"""
How well does the generative-flow FTLE predict mode changes in a policy with
spatially and temporally structured multimodality?

    python eval2.py            # S0 S1 S2 S3
    python eval2.py S2 S3      # subset; results cached in results2.json

S0  imitation: route distribution vs expert, collisions
S1  spatial: 3-way basins and triple junctions vs FTLE ridges (precision / recall)
S2  temporal: FTLE and ridge height along rollouts vs entropy of each pending decision
S3  per-seed prediction of mode flips (open loop and closed loop, noise and goal
    perturbations): AUROC of FTLE and baselines, cluster-bootstrap CIs
"""
import json, os, sys, time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle
from scipy.ndimage import binary_dilation
from scipy.stats import spearmanr
from policy2 import Policy, A, ftle
from env2 import (norm_obs, route_of_traj, collisions, chunk_pending_label, pending_column,
                  route_probs_expert, probs_A, COLS, OBSTACLES, GAPS_A, GAPS_B, H, EXEC,
                  ACT_SCALE, X_START, X_GOAL, CORR, ROUTE_NAMES)

ONLY = set(sys.argv[1:]) or {"S0", "S1", "S2", "S3"}
os.makedirs("figs", exist_ok=True)
R = json.load(open("results2.json")) if os.path.exists("results2.json") else {}
pol = Policy.load("policy2.npz")
NS = 20                      # RK4 steps for every sample / Jacobian
HORIZON = H * 0.101          # x-reach of one chunk (mean expert step 0.101)

SURF, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN = CAT
VIOLET = "#4a3aa7"
SEQ = LinearSegmentedColormap.from_list("seq", ["#f0efec", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "axes.titlesize": 11, "axes.titleweight": "semibold", "legend.frameon": False, "lines.linewidth": 2,
})


def save():
    json.dump(R, open("results2.json", "w"), indent=1, default=float)


def obsn(pos, g):
    return norm_obs(np.column_stack([pos, np.broadcast_to(np.asarray(g, float), (len(pos),))]))


def entropy_bits(labels, K):
    p = np.bincount(labels[labels >= 0], minlength=K) / max((labels >= 0).sum(), 1)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def rollout(pos0, g, seed, first_noise=None, R_=None, bs=20000):
    """Closed-loop receding-horizon rollouts (batched). Rows sharing `seed` get the
    same noise stream, so variants differ only where we perturb them."""
    pos = np.asarray(pos0, float).copy(); N = len(pos)
    g = np.broadcast_to(np.asarray(g, float), (N,)).copy()
    rng = np.random.default_rng(seed)
    if R_ is None:
        R_ = int(np.ceil((X_GOAL - pos[:, 0].min()) / (EXEC * 0.1))) + 2
    traj, states = [pos.copy()], []
    for r in range(R_):
        z = rng.standard_normal((N, A)).astype(np.float32)
        if r == 0 and first_noise is not None:
            z = first_noise.astype(np.float32)
        states.append(pos.copy())
        ch = np.concatenate([pol.sample(z[i:i + bs], obsn(pos[i:i + bs], g[i:i + bs]), NS)
                             for i in range(0, N, bs)])
        d = ch.reshape(N, H, 2)[:, :EXEC].astype(float) * ACT_SCALE
        for k in range(EXEC):
            pos = pos + d[:, k]
            traj.append(pos.copy())
    return np.stack(traj, 1), np.stack(states, 1)


def jac(z, pos, g, with_g=False, record=False, bs=1500):
    outs = []
    for i in range(0, len(z), bs):
        outs.append(pol.sample_jac(z[i:i + bs].astype(np.float32), obsn(pos[i:i + bs], g[i:i + bs]),
                                   NS, with_g=with_g, record=record))
    return [None if outs[0][k] is None else np.concatenate([o[k] for o in outs]) for k in range(len(outs[0]))]


def pend_label(z, pos, g, bs=10000):
    """Sample chunks and label the pending gap, in memory-bounded batches."""
    outs = []
    for i in range(0, len(z), bs):
        ch = pol.sample(z[i:i + bs].astype(np.float32), obsn(pos[i:i + bs], g[i:i + bs]), NS).astype(float)
        outs.append(chunk_pending_label(pos[i:i + bs], ch))
    return tuple(np.concatenate([o[k] for o in outs]) for k in range(3))


def auroc(score, y):
    score = np.asarray(score, float); y = np.asarray(y, bool)
    n1, n0 = y.sum(), (~y).sum()
    if n1 == 0 or n0 == 0:
        return np.nan
    r = np.argsort(np.argsort(score, kind="mergesort"), kind="mergesort") + 1.0
    # average ranks for ties
    _, inv, cnt = np.unique(score, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=r); r = (sums / cnt)[inv]
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def within_auroc(score, y, grp):
    U = P = 0.0
    for gi in np.unique(grp):
        m = grp == gi
        if y[m].any() and (~y[m]).any():
            a = auroc(score[m], y[m]); n = y[m].sum() * (~y[m]).sum()
            U += a * n; P += n
    return U / P if P else np.nan


def boot_ci(fn, grp, n_boot=300, seed=0):
    rng = np.random.default_rng(seed)
    ug = np.unique(grp); idx_by = {g: np.where(grp == g)[0] for g in ug}
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(ug, len(ug))
        idx = np.concatenate([idx_by[g] for g in pick])
        newgrp = np.concatenate([np.full(len(idx_by[g]), k) for k, g in enumerate(pick)])
        vals.append(fn(idx, newgrp))
    v = np.array(vals); v = v[~np.isnan(v)]
    return (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) if len(v) else (np.nan, np.nan)


def draw_env(ax):
    for cx, cy, r in OBSTACLES:
        ax.add_patch(Circle((cx, cy), r, color=INK2, alpha=.22, lw=0))
    ax.set_aspect("equal"); ax.set_xlim(-2.7, 2.7); ax.set_ylim(-2.3, 2.3)


t00 = time.time()
# ============================================================================ S0
if "S0" in ONLY:
    print("S0", flush=True)
    gs = np.linspace(-0.9, 0.9, 10); n = 400
    rows = []
    for j, g in enumerate(gs):
        rng = np.random.default_rng(100 + j)
        pos0 = np.column_stack([np.full(n, X_START), rng.uniform(-0.3, 0.3, n)])
        traj, _ = rollout(pos0, g, seed=200 + j)
        iA, iB, r = route_of_traj(traj)
        pol_r = np.bincount(r[r >= 0], minlength=6) / n
        exp_r = route_probs_expert(g).ravel()
        goal_err = np.linalg.norm(traj[:, -1] - np.array([X_GOAL, g]), axis=1)
        rows.append(dict(g=float(g), policy=pol_r.tolist(), expert=exp_r.tolist(),
                         tv=float(0.5 * np.abs(pol_r - exp_r).sum() + 0.5 * (r < 0).mean()),
                         unlabelled=float((r < 0).mean()), collision=float(collisions(traj).mean()),
                         goal_err_median=float(np.median(goal_err))))
        print("  ", {k: (np.round(v, 3).tolist() if isinstance(v, list) else round(v, 3)) for k, v in rows[-1].items()}, flush=True)
    R["S0"] = rows; save()
    rng = np.random.default_rng(7)
    pos0 = np.column_stack([np.full(120, X_START), rng.uniform(-0.3, 0.3, 120)])
    traj, _ = rollout(pos0, 0.1, seed=8)
    _, _, rr = route_of_traj(traj)
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.3), constrained_layout=True,
                            gridspec_kw=dict(width_ratios=[1.3, 1, 1]))
    ax = axs[0]; draw_env(ax)
    for k in range(6):
        for tr in traj[rr == k]:
            ax.plot(tr[:, 0], tr[:, 1], color=CAT[k], lw=0.9, alpha=.85)
        ax.plot([], [], color=CAT[k], label=ROUTE_NAMES[k])
    ax.set_title("120 closed-loop rollouts, goal g = 0.1"); ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.legend(fontsize=8, ncol=3, loc="lower center", bbox_to_anchor=(.5, -.02))
    P = np.array([r_["policy"] for r_ in rows]).reshape(-1, 3, 2); E = np.array([r_["expert"] for r_ in rows]).reshape(-1, 3, 2)
    gf = np.linspace(-1, 1, 200); EA = probs_A(gf)
    ax = axs[1]
    for i, nm in enumerate(COLS[0]["labels"]):
        ax.plot(gf, EA[:, i], color=CAT[i], lw=1.3, ls=(0, (4, 3)))
        ax.plot(gs, P[:, i].sum(1), color=CAT[i], marker="o", ms=4, label=f"A = {nm}")
    ax.plot([], [], color=MUTED, ls=(0, (4, 3)), lw=1.3, label="expert")
    ax.set_xlabel("goal height g"); ax.set_ylabel("probability"); ax.set_title("First decision: gap in column A")
    ax.legend(fontsize=8.5)
    ax = axs[2]
    for i, nm in enumerate(COLS[0]["labels"]):
        pa = P[:, i].sum(1); ea = E[:, i].sum(1)
        ok = pa > 0.03
        ax.plot(gs[ok], P[ok, i, 0] / pa[ok], color=CAT[i], marker="o", ms=4, label=f"after A = {nm}")
        ax.plot(gs, E[:, i, 0] / ea, color=CAT[i], lw=1.3, ls=(0, (4, 3)))
    ax.set_xlabel("goal height g"); ax.set_ylabel("P(B = upper | A gap)")
    ax.set_title("Second decision: gap in column B"); ax.legend(fontsize=8.5)
    fig.savefig("figs/s0_imitation.png", dpi=160); plt.close(fig)

# ============================================================================ S1
if "S1" in ONLY:
    print("S1", flush=True)
    cases = [("start, g = 0.0", np.array([X_START, 0.0]), 0.0),
             ("start, g = 0.5", np.array([X_START, 0.0]), 0.5),
             ("after mid gap of A (x = 0), g = 0.0", np.array([0.0, 0.0]), 0.0)]
    grid = np.linspace(-3, 3, 81); G1, G2 = np.meshgrid(grid, grid)
    out, fields = [], []
    rng = np.random.default_rng(31)
    for name, pos, g in cases:
        Z = rng.standard_normal((20000, A))
        col, lab, reached = pend_label(Z, np.tile(pos, (len(Z), 1)), np.full(len(Z), g))
        K = 3 if col[0] == 0 else 3          # B labels: 0 upper, 1 lower, 2 = outside
        present = [k for k in range(K) if (lab == k).mean() > 0.005]
        mu = np.stack([Z[lab == k].mean(0) for k in present]); m0 = Z.mean(0)
        Sw = sum(np.cov(Z[lab == k].T) * (lab == k).sum() for k in present) / len(Z)
        Sb = sum((lab == k).mean() * np.outer(mu[i] - m0, mu[i] - m0) for i, k in enumerate(present))
        w, V = np.linalg.eig(np.linalg.solve(Sw, Sb)); V = np.real(V[:, np.argsort(-np.real(w))])
        u1 = V[:, 0] / np.linalg.norm(V[:, 0])
        u2 = V[:, 1] if len(present) > 2 else rng.standard_normal(A)
        u2 = u2 - (u2 @ u1) * u1; u2 /= np.linalg.norm(u2)
        seeds = G1.ravel()[:, None] * u1 + G2.ravel()[:, None] * u2
        P_ = np.tile(pos, (len(seeds), 1)); g_ = np.full(len(seeds), g)
        a1, F, _ = jac(seeds, P_, g_)
        f = ftle(F).reshape(G1.shape)
        _, lb, _ = chunk_pending_label(P_, a1.astype(float)); lb = lb.reshape(G1.shape)
        edge = np.zeros_like(lb, bool)
        edge[:-1] |= lb[:-1] != lb[1:]; edge[1:] |= lb[:-1] != lb[1:]
        edge[:, :-1] |= lb[:, :-1] != lb[:, 1:]; edge[:, 1:] |= lb[:, :-1] != lb[:, 1:]
        q = max(edge.mean(), 0.005)
        high = f >= np.quantile(f, 1 - q)                 # as many high-FTLE pixels as edge pixels
        prec = float((high & binary_dilation(edge, iterations=2)).sum() / high.sum())
        rec = float((edge & binary_dilation(high, iterations=2)).sum() / max(edge.sum(), 1))
        out.append(dict(case=name, label_mass={COLS[col[0]]["labels"][k] if k < len(COLS[col[0]]["labels"]) else "outside":
                                               float((lab == k).mean()) for k in range(K)},
                        n_modes_in_slice=int(len(np.unique(lb))), precision=prec, recall=rec,
                        ftle_edge_median=float(np.median(f[edge])) if edge.any() else np.nan,
                        ftle_off_edge_p99=float(np.quantile(f[~binary_dilation(edge, iterations=3)], 0.99))))
        fields.append((name, f, lb, col[0]))
        print("  ", out[-1], flush=True)
    R["S1"] = out; save()
    vmin = min(f.min() for _, f, _, _ in fields); vmax = max(np.quantile(f, 0.995) for _, f, _, _ in fields)
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
    for ax, (name, f, lb, c), o in zip(axs, fields, out):
        im = ax.imshow(f, extent=[-3, 3, -3, 3], origin="lower", cmap=SEQ, vmin=vmin, vmax=vmax)
        for k in np.unique(lb):
            ax.contour(G1, G2, (lb == k).astype(float), levels=[.5], colors=ORANGE, linewidths=1.4, linestyles="dashed")
        ax.set_title(f"{name}\nprecision {o['precision']:.2f} · recall {o['recall']:.2f}")
        ax.set_xlabel("noise along discriminant 1"); ax.set_ylabel("noise along discriminant 2" if c == 0 else "noise along random ⊥ direction")
    fig.colorbar(im, ax=axs, shrink=.85).set_label("FTLE")
    axs[0].plot([], [], color=ORANGE, ls="dashed", label="gap boundaries (endpoint labels)"); axs[0].legend(fontsize=8.5, loc="lower left")
    fig.suptitle("Three-way and two-way choices: FTLE ridges on 2-D slices of the 24-D noise space", fontweight="semibold")
    fig.savefig("figs/s1_spatial.png", dpi=160); plt.close(fig)

# ============================================================================ S2
if "S2" in ONLY:
    print("S2", flush=True)
    g0 = 0.1; n_ep = 40
    rng = np.random.default_rng(41)
    pos0 = np.column_stack([np.full(n_ep, X_START), rng.uniform(-0.3, 0.3, n_ep)])
    _, states = rollout(pos0, g0, seed=42)
    S = states.reshape(-1, 2); S = S[S[:, 0] < COLS[1]["x"] + CORR]          # keep states before B is passed
    nS = len(S); print("   states", nS, flush=True)
    # FTLE quantiles over 32 seeds per state
    nz = 32
    Z = np.random.default_rng(43).standard_normal((nS * nz, A))
    Sp = np.repeat(S, nz, 0); gp = np.full(len(Sp), g0)
    _, F, _ = jac(Z, Sp, gp)
    fl = ftle(F).reshape(nS, nz)
    # chunk-level pending-decision entropy from 512 samples, + ridge height by bisection
    nl = 512
    Zl = np.random.default_rng(44).standard_normal((nS * nl, A))
    col, lab, reached = pend_label(Zl, np.repeat(S, nl, 0), np.full(nS * nl, g0))
    lab = lab.reshape(nS, nl); col = col.reshape(nS, nl)[:, 0]; reached = reached.reshape(nS, nl).mean(1)
    Hc = np.array([entropy_bits(lab[i], 3) for i in range(nS)])
    pairs_a, pairs_b, owner = [], [], []
    Zl = Zl.reshape(nS, nl, A)
    for i in range(nS):
        ks = np.unique(lab[i])
        if len(ks) < 2:
            continue
        minority = min(ks, key=lambda k: (lab[i] == k).sum())
        ia = np.where(lab[i] == minority)[0]; ib = np.where(lab[i] != minority)[0]
        D = np.linalg.norm(Zl[i, ia][:, None] - Zl[i, ib][None], axis=2)
        for _ in range(min(4, len(ia))):
            p, q = np.unravel_index(np.argmin(D), D.shape)
            pairs_a.append(Zl[i, ia[p]]); pairs_b.append(Zl[i, ib[q]]); owner.append(i)
            D[p, :] = np.inf
    pairs_a = np.array(pairs_a); pairs_b = np.array(pairs_b); owner = np.array(owner)
    la = pend_label(pairs_a, S[owner], np.full(len(owner), g0))[1]
    lo, hi = np.zeros(len(owner)), np.ones(len(owner))
    for _ in range(14):
        mid = 0.5 * (lo + hi)
        lm = pend_label(pairs_a + mid[:, None] * (pairs_b - pairs_a), S[owner], np.full(len(owner), g0))[1]
        same = lm == la
        lo = np.where(same, mid, lo); hi = np.where(same, hi, mid)
    zb = pairs_a + (0.5 * (lo + hi))[:, None] * (pairs_b - pairs_a)
    _, Fb, _ = jac(zb, S[owner], np.full(len(owner), g0))
    fb = ftle(Fb)
    ridge = np.full(nS, np.nan)
    for i in np.unique(owner):
        ridge[i] = np.median(fb[owner == i])
    # closed-loop entropy of each still-pending decision, from 48 continuations per state
    nc = 48
    traj, _ = rollout(np.repeat(S, nc, 0), g0, seed=45)
    iA, iB, _ = route_of_traj(traj)
    iA = iA.reshape(nS, nc); iB = iB.reshape(nS, nc)
    HA = np.array([entropy_bits(iA[i], 3) if S[i, 0] < COLS[0]["x"] else 0.0 for i in range(nS)])
    HB = np.array([entropy_bits(iB[i], 2) if S[i, 0] < COLS[1]["x"] else 0.0 for i in range(nS)])
    Hpend = np.where(col == 0, HA, np.where(col == 1, HB, 0.0))
    ok = ~np.isnan(ridge)
    stats = dict(
        n_states=int(nS),
        spearman_ftle_max_vs_chunk_entropy=float(spearmanr(fl.max(1), Hc)[0]),
        spearman_ftle_p90_vs_chunk_entropy=float(spearmanr(np.quantile(fl, .9, 1), Hc)[0]),
        spearman_ftle_max_vs_pending_closed_loop_entropy=float(spearmanr(fl.max(1), Hpend)[0]),
        spearman_ridge_vs_chunk_entropy=float(spearmanr(ridge[ok], Hc[ok])[0]),
        spearman_ridge_vs_pending_closed_loop_entropy=float(spearmanr(ridge[ok], Hpend[ok])[0]),
        frac_states_with_ridge=float(ok.mean()),
        B_closed_loop_entropy_at_start_mean=float(HB[S[:, 0] < X_START + 0.2].mean()),
        chunk_entropy_when_B_pending_but_out_of_horizon=float(Hc[(col == 1) & (S[:, 0] < COLS[1]["x"] - CORR - HORIZON)].mean())
        if ((col == 1) & (S[:, 0] < COLS[1]["x"] - CORR - HORIZON)).any() else np.nan)
    R["S2"] = dict(stats=stats, x=S[:, 0].tolist(), y=S[:, 1].tolist(), HA=HA.tolist(), HB=HB.tolist(),
                   Hchunk=Hc.tolist(), col=col.tolist(), reached=reached.tolist(),
                   ftle_max=fl.max(1).tolist(), ftle_med=np.median(fl, 1).tolist(),
                   ftle_p90=np.quantile(fl, .9, 1).tolist(), ridge=ridge.tolist())
    save(); print("  ", stats, flush=True)

    import plot_s2
    plot_s2.plot(R["S2"])

# ============================================================================ S3
if "S3" in ONLY:
    print("S3", flush=True)
    rng = np.random.default_rng(51)
    n_ep = 150
    gE = rng.uniform(-0.9, 0.9, n_ep)
    pos0 = np.column_stack([np.full(n_ep, X_START), rng.uniform(-0.3, 0.3, n_ep)])
    _, states = rollout(pos0, gE, seed=52)
    S = states.reshape(-1, 2); gS = np.repeat(gE, states.shape[1])
    keep = pending_column(S) >= 0
    S, gS = S[keep], gS[keep]
    pick = rng.choice(len(S), 360, replace=False); S, gS = S[pick], gS[pick]
    nS, nz = len(S), 24
    grp = np.repeat(np.arange(nS), nz)
    Sp, gp = np.repeat(S, nz, 0), np.repeat(gS, nz)
    Z = rng.standard_normal((nS * nz, A))
    t0 = time.time()
    a1, F, fg, eul, lnorm = jac(Z, Sp, gp, with_g=True, record=True)
    sc_ftle = ftle(F)
    Aug = np.zeros((len(F), A + 1, A + 1)); Aug[:, :A, :A] = F; Aug[:, :A, A] = fg; Aug[:, A, A] = 1
    sc_joint = ftle(Aug)
    sc_g = np.log(np.linalg.norm(fg.astype(float), axis=1) + 1e-12)
    print(f"   jacobians {time.time() - t0:.0f}s", flush=True)
    # state-level baseline: entropy of pending gap from 256 samples
    Zs = rng.standard_normal((nS * 256, A))
    _, ls, _ = pend_label(Zs, np.repeat(S, 256, 0), np.repeat(gS, 256))
    Hs = np.array([entropy_bits(l, 3) for l in ls.reshape(nS, 256)])
    sc_state = np.repeat(Hs, nz)
    # open-loop flips
    eps, dg, ndir = 0.1, 0.05, 4
    dirs = rng.standard_normal((ndir, len(Z), A)); dirs /= np.linalg.norm(dirs, axis=2, keepdims=True)
    l0 = np.concatenate([chunk_pending_label(Sp[i:i + 10000], a1[i:i + 10000].astype(float))[1] for i in range(0, len(Sp), 10000)])
    ol_noise = np.zeros(len(Z), bool)
    for d in dirs:
        ol_noise |= pend_label(Z + eps * d, Sp, gp)[1] != l0
    ol_goal = (pend_label(Z, Sp, gp + dg)[1] != l0) | (pend_label(Z, Sp, gp - dg)[1] != l0)
    # closed-loop flips: perturb only the first chunk's noise (later noise shared), or the goal
    t0 = time.time()
    variants = [("base", Z, gp)] + [(f"n{k}", Z + eps * dirs[k], gp) for k in range(ndir)] + \
               [("g+", Z, gp + dg), ("g-", Z, gp - dg)]
    routes = {}
    for nm, zz, gg in variants:
        traj, _ = rollout(Sp, gg, seed=53, first_noise=zz)
        routes[nm] = route_of_traj(traj)
        print(f"   rollout {nm} {time.time() - t0:.0f}s", flush=True)
    bA, bB, bR = routes["base"]
    cl_noise = np.zeros(len(Z), bool); cl_noise_A = np.zeros(len(Z), bool); cl_noise_B = np.zeros(len(Z), bool)
    for k in range(ndir):
        a_, b_, r_ = routes[f"n{k}"]
        cl_noise |= r_ != bR; cl_noise_A |= a_ != bA; cl_noise_B |= b_ != bB
    cl_goal = (routes["g+"][2] != bR) | (routes["g-"][2] != bR)
    pendA = pending_column(Sp) == 0
    scores = {"FTLE (noise)": sc_ftle, "goal sensitivity log‖∂Φ/∂g‖": sc_g, "joint FTLE (noise, goal)": sc_joint,
              "Eulerian max σ(∂v/∂a)": eul, "log-norm bound ∫μ(∂v/∂a)": lnorm,
              "state entropy (no seed info)": sc_state}
    targets = {"open loop, noise": ol_noise, "open loop, goal": ol_goal,
               "closed loop, noise": cl_noise, "closed loop, goal": cl_goal}
    res = {}
    for tn, y in targets.items():
        res[tn] = dict(base_rate=float(y.mean()))
        for sn, s in scores.items():
            ov = auroc(s, y); wi = within_auroc(s, y, grp)
            ci_ov = boot_ci(lambda idx, gr: auroc(s[idx], y[idx]), grp)
            ci_wi = boot_ci(lambda idx, gr: within_auroc(s[idx], y[idx], gr), grp, n_boot=150)
            res[tn][sn] = dict(overall=ov, overall_ci=ci_ov, within=wi, within_ci=ci_wi)
        print("  ", tn, round(y.mean(), 3), {sn: round(v["overall"], 3) for sn, v in res[tn].items() if sn != "base_rate"}, flush=True)
    # temporal split: does the first chunk's FTLE predict the A flip or the later B flip?
    m = pendA
    res["temporal split (A pending)"] = dict(
        n=int(m.sum()), rate_A_flip=float(cl_noise_A[m].mean()), rate_B_only_flip=float((cl_noise_B & ~cl_noise_A)[m].mean()),
        auroc_ftle_A_flip=auroc(sc_ftle[m], cl_noise_A[m]),
        auroc_ftle_B_only_flip=auroc(sc_ftle[m], (cl_noise_B & ~cl_noise_A)[m]))
    # how much of the closed-loop noise effect is the open-loop flip?
    res["closed vs open loop (noise)"] = dict(
        P_cl_given_ol=float(cl_noise[ol_noise].mean()) if ol_noise.any() else np.nan,
        P_cl_given_not_ol=float(cl_noise[~ol_noise].mean()))
    print("  ", res["temporal split (A pending)"], res["closed vs open loop (noise)"], flush=True)
    R["S3"] = res; save()
    np.savez_compressed("s3_rows.npz", ftle=sc_ftle, joint=sc_joint, gsens=sc_g, eul=eul, lnorm=lnorm,
                        state_H=sc_state, grp=grp, x=Sp[:, 0], g=gp, ol_noise=ol_noise, ol_goal=ol_goal,
                        cl_noise=cl_noise, cl_goal=cl_goal, cl_noise_A=cl_noise_A, cl_noise_B=cl_noise_B)

    import plot_s3
    plot_s3.plot(res, dict(np.load("s3_rows.npz")))

print(f"done in {time.time() - t00:.0f}s")
