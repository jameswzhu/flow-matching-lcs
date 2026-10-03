"""
Re-test the 2D-GMM findings on the learned, 16-D, goal-conditioned robot policy.

    python validate.py            # all
    python validate.py 1 4        # subset

V0  imitation check (route probability vs expert; closed-loop rollouts)
V1  FTLE ridge = basin boundary; boundary location vs Phi^-1(route probability)
V2  joint (noise, goal) field; rare-route fragility under small goal changes
V3  is FTLE a route-count detector?  (minority-route mass vs max FTLE)
V4  few-step Euler bias, open loop and closed loop
"""
import json, os, sys, time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import norm
from scipy.ndimage import binary_dilation
from policy import Policy, A, ftle
from robot_task import (norm_obs, chunk_route_label, p_top_expert, make_demos,
                        OBST_C, OBST_R, T_EP)
from evalutil import p_top_open_loop, rollout, START

ONLY = set(sys.argv[1:]) or {"0", "1", "2", "2b", "3", "4"}
os.makedirs("figs", exist_ok=True)
R = json.load(open("results.json")) if os.path.exists("results.json") else {}
pol = Policy.load("policy.npz", dtype=np.float32)   # f32: FTLE agrees with f64 to 6e-5
N_STEPS = 30          # reference RK4 steps

SURF, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7"
SEQ = LinearSegmentedColormap.from_list(
    "seq", ["#f0efec", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": MUTED,
    "ytick.color": MUTED, "text.color": INK, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 10, "axes.titlesize": 11,
    "axes.titleweight": "semibold", "legend.frameon": False, "lines.linewidth": 2,
})


def save():
    json.dump(R, open("results.json", "w"), indent=1, default=float)


def logistic_fit(Z, y, iters=25, l2=1e-3):
    """Newton logistic regression; returns unit normal u (pointing to y=1) and
    the hyperplane offset along u, plus training accuracy."""
    X = np.column_stack([Z, np.ones(len(Z))])
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ w))
        g = X.T @ (p - y) + l2 * np.r_[w[:-1], 0]
        Hm = (X * (p * (1 - p))[:, None]).T @ X + l2 * np.eye(X.shape[1])
        w -= np.linalg.solve(Hm, g)
    acc = float(((X @ w > 0) == (y > 0.5)).mean())
    nrm = np.linalg.norm(w[:-1])
    return w[:-1] / nrm, -w[-1] / nrm, acc


def ridge_on_line(s, f):
    i = int(np.argmax(f)); i = min(max(i, 1), len(s) - 2)
    y0, y1, y2 = f[i - 1:i + 2]
    den = y0 - 2 * y1 + y2
    return s[i] + (0.5 * (y0 - y2) / den if den != 0 else 0) * (s[1] - s[0]), f[i]


def label_flip_on_line(s, lab):
    k = np.where(np.diff(lab) != 0)[0]
    return [0.5 * (s[i] + s[i + 1]) for i in k]


def jac_batched(a0, ob, with_g=False, bs=2000):
    outs = [pol.sample_jac(a0[i:i + bs], ob, N_STEPS, with_g) for i in range(0, len(a0), bs)]
    a1 = np.concatenate([o[0] for o in outs]); F = np.concatenate([o[1] for o in outs])
    fg = np.concatenate([o[2] for o in outs]) if with_g else None
    return a1, F, fg


t00 = time.time()
# =============================================================================
# V0: imitation
# =============================================================================
if "0" in ONLY:
    print("V0")
    gs = np.linspace(-0.9, 0.9, 13)
    ol = [p_top_open_loop(pol, g, n=5000) for g in gs]
    cl, col, gerr = [], [], []
    for g in gs:
        r, c, e, _ = rollout(pol, g, n_ep=300)
        cl.append(float((r > 0).mean())); col.append(float(c.mean())); gerr.append(float(np.median(e)))
    _, _, _, trajs = rollout(pol, 0.0, n_ep=60, rng=np.random.default_rng(5))
    _, _, info = make_demos(40, np.random.default_rng(7))
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.0), constrained_layout=True)
    ax = axs[0]
    for (g, top, p) in info:
        ax.plot(p[:, 0], p[:, 1], color="#c3c2b7", lw=0.8)
    for tr in trajs:
        ax.plot(tr[:, 0], tr[:, 1], color=BLUE, lw=0.9, alpha=.8)
    ax.add_patch(plt.Circle(OBST_C, OBST_R, color=INK2, alpha=.25, lw=0))
    ax.plot([], [], color="#c3c2b7", label="expert demos (random goals)")
    ax.plot([], [], color=BLUE, label="policy rollouts, goal g = 0")
    ax.set_aspect("equal"); ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title("Task and closed-loop behaviour"); ax.legend(fontsize=8.5, loc="lower left")
    ax = axs[1]
    gf = np.linspace(-1, 1, 200)
    ax.plot(gf, p_top_expert(gf), color=MUTED, lw=1.5, ls=(0, (4, 3)), label="expert  σ(3g)")
    ax.plot(gs, ol, color=BLUE, marker="o", ms=4, label="policy, open loop (first chunk)")
    ax.plot(gs, cl, color=ORANGE, marker="o", ms=4, label="policy, closed loop (route taken)")
    ax.set_xlabel("goal height g (the conditioning)"); ax.set_ylabel("P(route over the obstacle)")
    ax.set_title("The policy imitates the expert's route preference"); ax.legend(fontsize=8.5)
    fig.savefig("figs/v0_imitation.png", dpi=160); plt.close(fig)
    R["v0"] = dict(g=gs.tolist(), expert=p_top_expert(gs).tolist(), open_loop=ol,
                   closed_loop=cl, collision_rate=col, median_goal_err=gerr)
    print("   max |open-loop - expert|", np.abs(np.array(ol) - p_top_expert(gs)).max(),
          " max collision", max(col)); save()

# =============================================================================
# V1: ridge = boundary; boundary location vs route probability
# =============================================================================
if "1" in ONLY:
    print("V1")
    rng = np.random.default_rng(11)
    gs1 = [-0.6, -0.3, 0.0, 0.3, 0.6]
    s = np.linspace(-3.5, 3.5, 701)
    rows = [] if "v1_line" not in R else None
    for g in (gs1 if rows is not None else []):
        ob = norm_obs(np.r_[START, g])
        Z = rng.standard_normal((20000, A))
        lab = (chunk_route_label(pol.sample(Z, ob, N_STEPS)) == 1).astype(float)
        p_top = lab.mean()
        u, off, acc = logistic_fit(Z, lab)
        a1, F, _ = jac_batched(s[:, None] * u[None], ob)
        f = ftle(F)
        r_loc, r_h = ridge_on_line(s, f)
        flips = label_flip_on_line(s, chunk_route_label(a1))
        rows.append(dict(g=g, p_top=float(p_top), linear_acc=acc, hyperplane_offset=float(off),
                         pred_from_prob=float(norm.ppf(1 - p_top)), ridge=float(r_loc),
                         ridge_ftle=float(r_h), label_flips=[float(x) for x in flips]))
        print("  ", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in rows[-1].items()})
    if rows is not None: R["v1_line"] = rows; save()

    # 2D slices through the origin spanned by u (separating direction) and a random u2
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)
    gsl = np.linspace(-3, 3, 81); G1, G2 = np.meshgrid(gsl, gsl)
    R["v1_slice"] = []
    fields = []
    for g in [0.0, 0.6]:
        ob = norm_obs(np.r_[START, g])
        Z = rng.standard_normal((20000, A))
        lab = (chunk_route_label(pol.sample(Z, ob, N_STEPS)) == 1).astype(float)
        u, off, acc = logistic_fit(Z, lab)
        u2 = rng.standard_normal(A); u2 -= (u2 @ u) * u; u2 /= np.linalg.norm(u2)
        seeds = G1.ravel()[:, None] * u[None] + G2.ravel()[:, None] * u2[None]
        a1, F, _ = jac_batched(seeds, ob)
        fields.append((g, ftle(F).reshape(G1.shape), chunk_route_label(a1).reshape(G1.shape), lab.mean()))
    vmax = max(np.percentile(f, 99.5) for _, f, _, _ in fields)
    vmin = min(f.min() for _, f, _, _ in fields)
    for ax, (g, fld, lb, pt) in zip(axs, fields):
        im = ax.imshow(fld, extent=[-3, 3, -3, 3], origin="lower", cmap=SEQ, vmin=vmin, vmax=vmax)
        ax.contour(G1, G2, (lb == 1).astype(float), levels=[.5], colors=ORANGE,
                   linewidths=1.6, linestyles="dashed")
        ax.axvline(norm.ppf(1 - pt), color=INK2, lw=1, ls=":")
        ax.set_title(f"goal g = {g}   (P(over) = {pt:.2f})")
        ax.set_xlabel("noise along separating direction u"); ax.set_ylabel("noise along random direction ⊥ u")
        edge = np.zeros_like(lb, bool)
        edge[:-1] |= lb[:-1] != lb[1:]; edge[1:] |= lb[:-1] != lb[1:]
        edge[:, :-1] |= lb[:, :-1] != lb[:, 1:]; edge[:, 1:] |= lb[:, :-1] != lb[:, 1:]
        near = binary_dilation(edge, iterations=2)
        top = fld >= np.percentile(fld, 99)
        R["v1_slice"].append(dict(g=g, frac_top1pct_ftle_on_label_edge=float((top & near).sum() / top.sum())))
    fig.colorbar(im, ax=axs, shrink=.85).set_label("FTLE")
    axs[0].plot([], [], color=ORANGE, ls="dashed", label="route boundary (from endpoint labels)")
    axs[0].plot([], [], color=INK2, ls=":", label=r"$\Phi^{-1}(1-P(\mathrm{over}))$")
    axs[0].legend(fontsize=8.5, loc="lower left")
    fig.suptitle("Learned 16-D policy: FTLE ridge on a 2-D slice of noise space vs the route boundary",
                 fontweight="semibold")
    fig.savefig("figs/v1_slices.png", dpi=160); plt.close(fig)
    print("  ", R["v1_slice"]); save()

# =============================================================================
# V2: joint (noise, goal) field and rare-route fragility
# =============================================================================
if "2" in ONLY:
    print("V2")
    rng = np.random.default_rng(21)
    ob0 = norm_obs(np.r_[START, 0.0])
    Z = rng.standard_normal((20000, A))
    lab = (chunk_route_label(pol.sample(Z, ob0, N_STEPS)) == 1).astype(float)
    u, _, _ = logistic_fit(Z, lab)                       # fixed direction from g = 0
    s = np.linspace(-3, 3, 181)
    gg = np.linspace(-0.9, 0.9, 37)
    Fj = np.empty((len(gg), len(s))); Fn = np.empty_like(Fj); flipline = []; pred = []
    for j, g in enumerate(gg):
        ob = norm_obs(np.r_[START, g])
        a1, F, fg = pol.sample_jac(s[:, None] * u[None], ob, N_STEPS, with_g=True)
        Aug = np.zeros((len(s), A + 1, A + 1)); Aug[:, :A, :A] = F; Aug[:, :A, A] = fg; Aug[:, A, A] = 1
        Fj[j] = ftle(Aug); Fn[j] = ftle(F)
        fl = label_flip_on_line(s, chunk_route_label(a1)); flipline.append(fl[0] if len(fl) else np.nan)
        pt = p_top_open_loop(pol, g, n=4000, rng=np.random.default_rng(j))
        pred.append(norm.ppf(1 - pt) if 0 < pt < 1 else np.nan)
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    for ax, fld, ttl in [(axs[0], Fn, "FTLE w.r.t. noise"), (axs[1], Fj, "FTLE w.r.t. (noise, goal g)")]:
        im = ax.imshow(fld, extent=[s[0], s[-1], gg[0], gg[-1]], origin="lower", aspect="auto",
                       cmap=SEQ, vmax=np.percentile(fld, 99.5))
        ax.plot(flipline, gg, color=ORANGE, lw=1.8, ls="dashed", label="route boundary on this line")
        ax.plot(pred, gg, color=INK2, lw=1.2, ls=":", label=r"$\Phi^{-1}(1-P(\mathrm{over}\,|\,g))$")
        ax.set_xlabel("noise along u (fixed, from g = 0)"); ax.set_ylabel("goal height g"); ax.set_title(ttl)
        fig.colorbar(im, ax=ax, shrink=.9).set_label("FTLE")
    axs[0].legend(fontsize=8.5, loc="upper left")
    fig.suptitle("Learned policy: the route boundary sweeps through noise space as the goal moves",
                 fontweight="semibold")
    fig.savefig("figs/v2_joint_noise_goal.png", dpi=160); plt.close(fig)

    R["v2"] = dict(joint_boundary_vs_pred=dict(g=gg.tolist(), flip=list(map(float, flipline)),
                                               pred=list(map(float, pred))))
    save()

# =============================================================================
# V2b: rare-route fragility -- push each seed's goal TOWARD the other route
# =============================================================================
if "2b" in ONLY:
    print("V2b")
    dg = 0.05
    gq = np.linspace(-0.8, 0.8, 13)
    rows = []
    for g in gq:
        Zq = np.random.default_rng(int(1000 * (g + 1))).standard_normal((10000, A))
        lab = lambda gg_: chunk_route_label(pol.sample(Zq, norm_obs(np.r_[START, gg_]), N_STEPS)) == 1
        l0, lup, ldn = lab(g), lab(g + dg), lab(g - dg)
        p0 = l0.mean()
        # UNDER seeds pushed toward OVER (g up); OVER seeds pushed toward UNDER (g down)
        rows.append(dict(g=float(g), p_over=float(p0),
                         flip_under_to_over=float((~l0 & lup).sum() / max((~l0).sum(), 1)),
                         flip_over_to_under=float((l0 & ~ldn).sum() / max(l0.sum(), 1)),
                         wrong_way_flips=float(((l0 & ~lup) | (~l0 & ldn)).mean()),
                         dp_up=float(lup.mean() - p0), dp_dn=float(p0 - ldn.mean())))
        print("  ", {k: round(v, 4) for k, v in rows[-1].items()}, flush=True)
    R["v2b"] = rows; save()
    po = np.array([r["p_over"] for r in rows])
    fig, ax = plt.subplots(figsize=(6.8, 4.1), constrained_layout=True)
    ax.plot(po, [r["flip_over_to_under"] for r in rows], color=BLUE, marker="o", ms=4,
            label="OVER seeds, goal lowered by 0.05")
    ax.plot(po, [r["flip_under_to_over"] for r in rows], color=ORANGE, marker="o", ms=4,
            label="UNDER seeds, goal raised by 0.05")
    ax.plot(po, [r["dp_dn"] / max(p, 1e-9) for r, p in zip(rows, po)], color=BLUE, lw=1.2,
            ls=(0, (1, 2)), label="Δp / P(route)  (monotone sweep)")
    ax.plot(po, [r["dp_up"] / max(1 - p, 1e-9) for r, p in zip(rows, po)], color=ORANGE, lw=1.2, ls=(0, (1, 2)))
    ax.set_xlabel("P(over) at the current goal")
    ax.set_ylabel("fraction of seeds that switch route")
    ax.set_title("Whichever route is rarer is the more fragile one")
    ax.legend(fontsize=8.5, loc="upper center")
    fig.savefig("figs/v2_rare_route_fragility.png", dpi=160); plt.close(fig)
    print("   max wrong-way flip fraction", max(r["wrong_way_flips"] for r in rows))

# =============================================================================
# V3: FTLE vs route count
# =============================================================================
if "3" in ONLY:
    print("V3")
    rng = np.random.default_rng(31)
    pts = [] if "v3" not in R else None
    for x in ([-2.0, -1.5, -1.0] if pts is not None else []):
        for y in np.linspace(-0.6, 0.6, 13):
            ob = norm_obs(np.array([x, y, 0.0]))
            Z = rng.standard_normal((2000, A))
            a1, F, _ = jac_batched(Z, ob)
            l = chunk_route_label(a1) == 1
            fl = ftle(F)
            pts.append(dict(x=x, y=float(y), minority=float(min(l.mean(), 1 - l.mean())),
                            ftle_max=float(fl.max()), ftle_p99=float(np.percentile(fl, 99)),
                            ftle_median=float(np.median(fl))))
    if pts is not None: R["v3"] = pts; save()
    pts = R["v3"]
    fig, ax = plt.subplots(figsize=(6.8, 4.1), constrained_layout=True)
    for x, col in zip([-2.0, -1.5, -1.0], [BLUE, ORANGE, AQUA]):
        P = [p for p in pts if p["x"] == x]
        ax.scatter([max(p["minority"], 2e-4) for p in P], [p["ftle_max"] for p in P], s=28, color=col,
                   edgecolor=SURF, lw=1, label=f"state x = {x}", zorder=3)
    ax.set_xscale("log")
    ax.axvline(1 / 2000, color=MUTED, lw=1, ls=":")
    ax.text(1 / 2000 * 1.1, ax.get_ylim()[0] + .2, "minority route\nnot observed\n(<1 in 2000)", fontsize=8.5, color=INK2)
    ax.set_xlabel("mass of the minority route at this state (log scale)")
    ax.set_ylabel("max FTLE over 2000 noise seeds")
    ax.set_title("Max FTLE falls smoothly as the minority route vanishes")
    ax.legend(fontsize=8.5, loc="upper left")
    fig.savefig("figs/v3_ftle_vs_route_count.png", dpi=160); plt.close(fig)

# =============================================================================
# V4: few-step Euler bias, open and closed loop
# =============================================================================
if "4" in ONLY:
    print("V4")
    Ns = [2, 3, 5, 10, 20]
    cols = [MAGENTA, ORANGE, YELLOW, AQUA, BLUE]
    gs = np.linspace(-0.9, 0.9, 13)
    ref_ol = np.array([p_top_open_loop(pol, g, n=8000) for g in gs])
    gcl = np.array([-0.6, -0.3, 0.0, 0.3, 0.6])
    ref_cl = []
    for g in gcl:
        r, c, _, _ = rollout(pol, g, n_ep=400)
        ref_cl.append((r > 0).mean())
    ref_cl = np.array(ref_cl)
    R["v4"] = dict(g=gs.tolist(), ref_open=ref_ol.tolist(), g_cl=gcl.tolist(), ref_closed=ref_cl.tolist(), euler={})
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.1), constrained_layout=True)
    for N, col in zip(Ns, cols):
        ol = np.array([p_top_open_loop(pol, g, n=8000, n_steps=N, method="euler") for g in gs])
        cl, colr = [], []
        for g in gcl:
            r, c, _, _ = rollout(pol, g, n_ep=400, n_steps=N, method="euler")
            cl.append((r > 0).mean()); colr.append(c.mean())
        cl = np.array(cl)
        axs[0].plot(gs, ol - ref_ol, color=col, label=f"Euler, N = {N}")
        axs[1].plot(gcl, cl - ref_cl, color=col, marker="o", ms=4, label=f"Euler, N = {N}")
        R["v4"]["euler"][N] = dict(open=ol.tolist(), closed=cl.tolist(), collision=list(map(float, colr)))
        print(f"   N={N}: open-loop err range [{(ol-ref_ol).min():+.3f},{(ol-ref_ol).max():+.3f}]"
              f"  closed-loop err range [{(cl-ref_cl).min():+.3f},{(cl-ref_cl).max():+.3f}]  collisions {np.round(colr,3)}")
    for ax in axs:
        ax.axhline(0, color=MUTED, lw=.8); ax.set_xlabel("goal height g")
    axs[0].set_ylabel("P(over) error vs RK4-30"); axs[0].set_title("Open loop: first action chunk")
    axs[1].set_ylabel("P(over) error vs RK4-30"); axs[1].set_title("Closed loop: route actually taken (400 episodes)")
    axs[0].legend(fontsize=8.5, ncol=2)
    fig.suptitle("Learned policy: few-step Euler sampling shifts the route preference", fontweight="semibold")
    fig.savefig("figs/v4_euler.png", dpi=160); plt.close(fig)
    save()

print(f"done in {time.time() - t00:.0f}s")
