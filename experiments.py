"""
FTLE / Lagrangian-coherent-structure experiments on a closed-form 2D
conditional GMM flow.  Run:  python experiments.py   (writes figs/ and results.json)

Exp 1  FTLE ridge vs the analytic basin boundary (2 modes, weight = c)
Exp 2  Ridges in the joint (noise, conditioning) space; 3-mode non-separable case
Exp 3  Mode-merge bifurcation: ridge strength vs mode separation
Exp 4  Eulerian fixed points vs the Lagrangian separatrix in flow time
Exp 5  Discretization: few-step Euler distorts basins and mode probabilities
"""
import json, os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import norm
from gmmflow import velocity, flow, ftle_from_jac

import sys
os.makedirs("figs", exist_ok=True)
ONLY = set(sys.argv[1:]) or {"1", "2", "3", "4", "5"}
R = json.load(open("results.json")) if os.path.exists("results.json") else {}

# ---- style (light surface, recessive axes, single-hue sequential ramp) ------
SURF, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
SEQ = LinearSegmentedColormap.from_list(
    "seq", ["#f0efec", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": MUTED,
    "ytick.color": MUTED, "text.color": INK, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 10, "axes.titlesize": 11,
    "axes.titleweight": "semibold", "legend.frameon": False, "lines.linewidth": 2,
})

SIG = 0.3


def two_mode(c, m=2.0, sig=SIG):
    """Left mode at -m with weight c, right mode at +m with weight 1-c."""
    return (np.array([c, 1 - c]), np.array([[-m, 0.], [m, 0.]]),
            np.array([sig, sig]))


def analytic_boundary(c, m=2.0, sig=SIG):
    """Exact noise-space basin boundary for the two-mode target.

    With isotropic, equal-width modes on the x-axis, the y-dynamics decouple
    and the x-dynamics are a 1D ODE, so the flow map in x is the unique
    monotone map F_target^{-1} o Phi.  The trajectory that ends at the
    density saddle x=0 therefore starts at Phi^{-1}(F_target(0))."""
    Ft0 = c * norm.cdf(m / sig) + (1 - c) * norm.cdf(-m / sig)
    return norm.ppf(Ft0)


def ftle_1d(xg, P, c, **kw):
    """FTLE along y=0 (sufficient for the separable two-mode case)."""
    X = np.stack([xg, np.zeros_like(xg)], 1)
    _, F = flow(X, P, c, jac=True, **kw)
    return ftle_from_jac(F)


def ridge_1d(xg, P, c, **kw):
    """Ridge location from the FTLE maximum, refined by a parabola fit."""
    f = ftle_1d(xg, P, c, **kw)
    i = int(np.argmax(f))
    i = min(max(i, 1), len(xg) - 2)
    y0, y1, y2 = f[i - 1:i + 2]
    dx = xg[1] - xg[0]
    off = 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2) if (y0 - 2 * y1 + y2) != 0 else 0
    return xg[i] + off * dx, f[i], f


# =============================================================================
# Exp 1: FTLE ridge vs analytic boundary
# =============================================================================
if "1" in ONLY:
    print("Exp 1")
    g = np.linspace(-3, 3, 241)
    GX, GY = np.meshgrid(g, g)
    X0 = np.stack([GX.ravel(), GY.ravel()], 1)
    cs1 = [0.5, 0.3, 0.1]
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.9), constrained_layout=True)
    fields = []
    for c in cs1:
        _, F = flow(X0, two_mode, c, jac=True, n_steps=200)
        fields.append(ftle_from_jac(F).reshape(GX.shape))
    vmin = min(f.min() for f in fields); vmax = max(np.percentile(f, 99.5) for f in fields)
    xf = np.linspace(-3, 3, 4001)
    R["exp1"] = []
    for ax, c, fld in zip(axs, cs1, fields):
        im = ax.imshow(fld, extent=[-3, 3, -3, 3], origin="lower", cmap=SEQ,
                       vmin=vmin, vmax=vmax, aspect="equal")
        b = analytic_boundary(c)
        xr, peak, _ = ridge_1d(xf, two_mode, c, n_steps=200)
        ax.axvline(b, color=ORANGE, lw=2, ls=(0, (4, 3)), label="analytic boundary")
        ax.set_title(f"left-mode weight c = {c}")
        ax.set_xlabel("noise $a_0^{(1)}$"); ax.set_ylabel("noise $a_0^{(2)}$")
        ax.text(-2.85, 2.6, f"ridge {xr:+.4f}\nanalytic {b:+.4f}", fontsize=9,
                color=INK, va="top", bbox=dict(fc=SURF, ec="none", alpha=.85))
        R["exp1"].append(dict(c=c, ridge=xr, analytic=b, err=xr - b, peak_ftle=peak))
    axs[0].legend(loc="lower left", fontsize=9)
    cb = fig.colorbar(im, ax=axs, shrink=.85); cb.set_label("FTLE (flow time 0→1)")
    fig.suptitle("FTLE of the generative flow map over noise space: the ridge is the basin boundary",
                 fontweight="semibold")
    fig.savefig("figs/fig1_ftle_vs_analytic.png", dpi=160)
    plt.close(fig)
    for r in R["exp1"]:
        print(f"  c={r['c']}: ridge {r['ridge']:+.5f} analytic {r['analytic']:+.5f} err {r['err']:.1e}")

# =============================================================================
# Exp 2a: joint (noise, conditioning) FTLE for the two-mode case
# =============================================================================
if "2" in ONLY:
    print("Exp 2a")
    xg = np.linspace(-3, 3, 301)
    cg = np.linspace(0.02, 0.98, 241)
    dc = 1e-4
    ftle_joint = np.empty((len(cg), len(xg)))
    ftle_noise = np.empty_like(ftle_joint)
    for j, c in enumerate(cg):
        X = np.stack([xg, np.zeros_like(xg)], 1)
        xT, F = flow(X, two_mode, c, jac=True, n_steps=150)
        xp = flow(X, two_mode, c + dc, n_steps=150)
        xm = flow(X, two_mode, c - dc, n_steps=150)
        dPhidc = (xp - xm) / (2 * dc)                         # [N, 2]
        # augmented state (a0, c) with dc/dtau = 0:  [[F, dPhi/dc], [0, 1]]
        A = np.zeros((len(xg), 3, 3))
        A[:, :2, :2] = F; A[:, :2, 2] = dPhidc; A[:, 2, 2] = 1
        ftle_joint[j] = ftle_from_jac(A)
        ftle_noise[j] = ftle_from_jac(F)

    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    for ax, fld, ttl in [(axs[0], ftle_noise, "FTLE w.r.t. noise only"),
                         (axs[1], ftle_joint, "FTLE w.r.t. (noise, conditioning)")]:
        im = ax.imshow(fld, extent=[xg[0], xg[-1], cg[0], cg[-1]], origin="lower",
                       cmap=SEQ, aspect="auto", vmax=np.percentile(fld, 99.5))
        ax.plot([analytic_boundary(c) for c in cg], cg, color=ORANGE, lw=2,
                ls=(0, (4, 3)), label=r"analytic $b(c)=\Phi^{-1}(F_c(0))$")
        ax.set_xlabel("noise $a_0^{(1)}$"); ax.set_ylabel("conditioning c (left-mode weight)")
        ax.set_title(ttl)
        fig.colorbar(im, ax=ax, shrink=.9).set_label("FTLE")
    axs[0].legend(loc="upper left", fontsize=9)
    fig.suptitle("A fixed noise seed near b(c) flips mode under an infinitesimal change in c",
                 fontweight="semibold")
    fig.savefig("figs/fig2a_joint_noise_conditioning.png", dpi=160)
    plt.close(fig)

# =============================================================================
# Exp 2b: three modes (non-separable) - ridges vs endpoint-label boundaries
# =============================================================================
if "2" in ONLY:
    print("Exp 2b")
    ang = np.deg2rad([90, 210, 330])
    MU3 = 2.2 * np.stack([np.cos(ang), np.sin(ang)], 1)


    def three_mode(c):
        """c in [0,1] moves weight from mode A (top) to mode C (lower right)."""
        w = np.array([0.5 - 0.4 * c, 0.3, 0.2 + 0.4 * c])
        return w, MU3, np.array([0.3, 0.3, 0.3])


    g3 = np.linspace(-3, 3, 241)
    G3X, G3Y = np.meshgrid(g3, g3)
    X3 = np.stack([G3X.ravel(), G3Y.ravel()], 1)
    cs3 = [0.0, 0.5, 1.0]
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.9), constrained_layout=True)
    R["exp2b"] = []
    flds = []
    for c in cs3:
        xT, F = flow(X3, three_mode, c, jac=True, n_steps=200)
        lab = np.argmin(((xT[:, None] - MU3[None]) ** 2).sum(-1), 1).reshape(G3X.shape)
        flds.append((ftle_from_jac(F).reshape(G3X.shape), lab))
    vmax3 = max(np.percentile(f, 99.5) for f, _ in flds)
    for ax, c, (fld, lab) in zip(axs, cs3, flds):
        im = ax.imshow(fld, extent=[-3, 3, -3, 3], origin="lower", cmap=SEQ,
                       vmax=vmax3, aspect="equal")
        # endpoint-label boundaries (independent of the FTLE computation)
        for k in range(3):
            ax.contour(G3X, G3Y, (lab == k).astype(float), levels=[0.5],
                       colors=ORANGE, linewidths=1.5, linestyles="dashed")
        w = three_mode(c)[0]
        ax.set_title(f"c = {c}:  w = ({w[0]:.1f}, {w[1]:.1f}, {w[2]:.1f})")
        ax.set_xlabel("noise $a_0^{(1)}$"); ax.set_ylabel("noise $a_0^{(2)}$")
        # agreement: fraction of top-1% FTLE pixels lying within 2 px of a label edge
        edge = np.zeros_like(lab, bool)
        edge[:-1] |= lab[:-1] != lab[1:]; edge[1:] |= lab[:-1] != lab[1:]
        edge[:, :-1] |= lab[:, :-1] != lab[:, 1:]; edge[:, 1:] |= lab[:, :-1] != lab[:, 1:]
        from scipy.ndimage import binary_dilation
        near = binary_dilation(edge, iterations=2)
        top = fld >= np.percentile(fld, 99)
        frac = float((top & near).sum() / top.sum())
        mass = np.bincount(lab.ravel(), weights=np.exp(-0.5 * (X3 ** 2).sum(1)), minlength=3)
        R["exp2b"].append(dict(c=c, frac_top1pct_ftle_on_label_edge=frac,
                               weights=w.tolist(), noise_mass_per_basin=(mass / mass.sum()).tolist()))
    cb = fig.colorbar(im, ax=axs, shrink=.85); cb.set_label("FTLE")
    fig.suptitle("Three modes: FTLE ridges (blue) coincide with basin edges from endpoint labels (dashed)",
                 fontweight="semibold")
    fig.savefig("figs/fig2b_three_modes.png", dpi=160)
    plt.close(fig)
    for r in R["exp2b"]:
        print("  ", r)

# =============================================================================
# Exp 3: mode-merge bifurcation, separation m as the parameter
# =============================================================================
if "3" in ONLY:
    print("Exp 3")
    # Equal weights, so the boundary sits at a0 = 0 by symmetry. In this 1D
    # (separable) setting the x-stretch of the flow map is a density ratio:
    #   dPhi/da0 (a0) = phi(a0) / p_target(Phi(a0))  (monotone transport),
    # so at the boundary  log-stretch = log phi(0) - log p_target(0).
    # A ridge exists when a0 = 0 is a local MAX of the log-stretch field;
    # its birth is where the curvature at 0 changes sign.
    ms = np.linspace(0.0, 2.0, 81)
    xf = np.linspace(-0.6, 0.6, 241)
    i0 = len(xf) // 2; dx = xf[1] - xf[0]
    peak, curv, analytic = [], [], []
    for m in ms:
        P = lambda c, m=m: two_mode(c, m=m)
        X = np.stack([xf, np.zeros_like(xf)], 1)
        _, F = flow(X, P, 0.5, jac=True, n_steps=200)
        ell = np.log(np.abs(F[:, 0, 0]))                 # log-stretch along x
        peak.append(ell[i0])
        curv.append((ell[i0 + 1] - 2 * ell[i0] + ell[i0 - 1]) / dx ** 2)
        pT0 = 2 * 0.5 * norm.pdf(0, loc=m, scale=SIG)
        analytic.append(norm.logpdf(0) - np.log(pT0))
    peak, curv, analytic = map(np.array, (peak, curv, analytic))
    # exact curvature for the symmetric case (Phi''(0)=0, (log p_T)'(0)=0):
    #   l''(0) = -1 + (1 - m^2/sig^2) exp(m^2/sig^2)   ~  -(m/sig)^4 / 2  as m -> 0
    curv_an = -1 + (1 - ms ** 2 / SIG ** 2) * np.exp(ms ** 2 / SIG ** 2)
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 3.9), constrained_layout=True)
    ax = axs[0]
    ax.plot(ms, peak, color=BLUE, label="computed (variational eq.)")
    ax.plot(ms, analytic, color=ORANGE, ls=(0, (4, 3)), label=r"analytic $\log\varphi(0)-\log p_{\rm target}(0)$")
    ax.set_xlabel("half-separation m of the two modes (σ = 0.3)")
    ax.set_ylabel("log-stretch at the boundary")
    ax.set_title("Ridge height = log density ratio (grows ~ m²/2σ²)")
    ax.legend(fontsize=9, loc="upper left")
    ax = axs[1]
    mm = ms <= 0.75
    ax.plot(ms[mm], curv[mm], color=BLUE, label="computed (finite diff. of log-stretch)")
    ax.plot(ms[mm], curv_an[mm], color=ORANGE, ls=(0, (4, 3)),
            label=r"analytic $-1+(1-m^2/\sigma^2)\,e^{m^2/\sigma^2}$")
    ax.axhline(0, color=MUTED, lw=.8)
    ax.axvline(SIG, color=MUTED, lw=1, ls=":")
    ax.text(SIG + 0.02, -0.003, "target turns\nbimodal (m = σ)", fontsize=9, color=INK2)
    ax.set_xlabel("half-separation m of the two modes (σ = 0.3)")
    ax.set_ylabel("curvature of log-stretch at $a_0=0$")
    ax.set_yscale("symlog", linthresh=0.01)
    ax.set_title("Ridge exists for every m > 0: no threshold at bimodality")
    ax.legend(fontsize=9, loc="lower left")
    fig.suptitle("Merging two modes: the FTLE ridge varies smoothly and does not mark the bimodality transition", fontweight="semibold")
    fig.savefig("figs/fig3_merge_bifurcation.png", dpi=160)
    plt.close(fig)
    R["exp3"] = dict(m=ms.tolist(), log_stretch_at_boundary=peak.tolist(),
                     analytic=analytic.tolist(), curvature=curv.tolist(),
                     curvature_analytic=curv_an.tolist(),
                     max_rel_curv_err_0p05_le_m_le_0p6=float((np.abs(curv - curv_an) / np.abs(curv_an))[(ms >= 0.05) & (ms <= 0.6)].max()),
                     max_abs_err_vs_analytic=float(np.abs(peak - analytic).max()))
    print("   height err", R["exp3"]["max_abs_err_vs_analytic"], "curv err", R["exp3"]["max_rel_curv_err_0p05_le_m_le_0p6"])

# =============================================================================
# Exp 4: Eulerian fixed points vs the Lagrangian separatrix (flow time)
# =============================================================================
if "4" in ONLY:
    print("Exp 4")
    c4 = 0.2
    w4, mu4, s4 = two_mode(c4)
    taus = np.linspace(0, 1, 401)
    xs = np.linspace(-3.5, 3.5, 7001)
    Xs = np.stack([xs, np.zeros_like(xs)], 1)
    stable_pts, unstable_pts = [], []
    for t in taus:
        v = velocity(Xs, t, w4, mu4, s4)[:, 0]
        sgn = np.sign(v)
        idx = np.where(sgn[:-1] * sgn[1:] < 0)[0]
        for i in idx:
            x0 = xs[i] - v[i] * (xs[i + 1] - xs[i]) / (v[i + 1] - v[i])
            (unstable_pts if v[i + 1] > v[i] else stable_pts).append((t, x0))
    stable_pts = np.array(stable_pts); unstable_pts = np.array(unstable_pts)
    b4 = analytic_boundary(c4)
    _, path_b = flow(np.array([[b4, 0.]]), two_mode, c4, n_steps=400, return_path=True)
    rng = np.random.default_rng(1)
    starts = np.sort(rng.standard_normal(28))
    _, paths = flow(np.stack([starts, np.zeros_like(starts)], 1), two_mode, c4,
                       n_steps=400, return_path=True)
    tpath = np.linspace(0, 1, 401)
    fig, ax = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
    for p in paths:
        ax.plot(tpath, p[:, 0], color="#c3c2b7", lw=0.8, zorder=1)
    ax.scatter(stable_pts[:, 0], stable_pts[:, 1], s=3, color=BLUE, zorder=2,
               label="Eulerian: stable zeros of v(·, τ)")
    ax.scatter(unstable_pts[:, 0], unstable_pts[:, 1], s=3, color=YELLOW, zorder=2,
               label="Eulerian: unstable zero of v(·, τ)")
    ax.plot(tpath, path_b[0, :, 0], color=ORANGE, lw=2.2, zorder=3,
            label=f"Lagrangian separatrix (starts at b = {b4:+.3f})")
    t_sn = unstable_pts[:, 0].min()
    ax.axvline(t_sn, color=MUTED, lw=1, ls=":")
    ax.text(t_sn + 0.01, -3.3, f"saddle-node in v at τ ≈ {t_sn:.2f}", fontsize=9, color=INK2)
    ax.set_xlabel("flow time τ"); ax.set_ylabel("$a^{(1)}$")
    ax.set_title(f"Eulerian vs Lagrangian structure (c = {c4})")
    ax.legend(loc="upper left", fontsize=8.5, markerscale=4)
    fig.savefig("figs/fig4_eulerian_vs_lagrangian.png", dpi=160)
    plt.close(fig)
    # where would an "Eulerian" reading place the boundary? Trace the unstable
    # zero at its birth back to tau=0 along the true flow (backward integration).
    x_birth = unstable_pts[unstable_pts[:, 0] == t_sn][0, 1]
    xx = np.array([[x_birth, 0.]]); n = 400; h = t_sn / n
    for i in range(n):                            # RK4 backward from t_sn to 0
        t = t_sn - i * h
        f = lambda x, tt: velocity(x, tt, w4, mu4, s4)
        k1 = f(xx, t); k2 = f(xx - .5 * h * k1, t - .5 * h)
        k3 = f(xx - .5 * h * k2, t - .5 * h); k4 = f(xx - h * k3, t - h)
        xx = xx - h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    # separatrix position at t_sn vs unstable zero at t_sn
    i_sn = int(round(t_sn * 400))
    R["exp4"] = dict(c=c4, tau_saddle_node=float(t_sn), unstable_zero_at_birth=float(x_birth),
                     separatrix_at_birth=float(path_b[0, i_sn, 0]),
                     eulerian_boundary_pulled_back_to_tau0=float(xx[0, 0]),
                     true_boundary_tau0=float(b4),
                     mode_mass_implied_by_eulerian=float(norm.cdf(xx[0, 0])),
                     true_left_mass=c4)
    print("  ", R["exp4"])

# =============================================================================
# Exp 5: few-step Euler distorts basin boundaries and mode probabilities
# =============================================================================
if "5" in ONLY:
    print("Exp 5")
    Ns = [2, 3, 5, 10, 20, 50]
    cg5 = np.linspace(0.05, 0.95, 37)
    xq = np.linspace(-3.5, 3.5, 7001)
    Xq = np.stack([xq, np.zeros_like(xq)], 1)
    R["exp5"] = {}
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.1), constrained_layout=True)
    cols = ["#e87ba4", ORANGE, YELLOW, AQUA, BLUE, "#4a3aa7"]
    for N, col in zip(Ns, cols):
        bN, massN = [], []
        for c in cg5:
            xT = flow(Xq, two_mode, c, n_steps=N, method="euler")[:, 0]
            cross = np.where(np.diff(np.sign(xT)) != 0)[0]
            b = xq[cross[0]] if len(cross) else np.nan
            bN.append(b); massN.append(norm.cdf(b))
        bN = np.array(bN); massN = np.array(massN)
        axs[0].plot(cg5, bN - np.array([analytic_boundary(c) for c in cg5]), color=col,
                    label=f"Euler, N = {N}")
        axs[1].plot(cg5, massN - cg5, color=col, label=f"Euler, N = {N}")
        R["exp5"][N] = dict(max_abs_boundary_shift=float(np.nanmax(np.abs(
            bN - np.array([analytic_boundary(c) for c in cg5])))),
            max_abs_mass_error=float(np.nanmax(np.abs(massN - cg5))))
    for ax in axs:
        ax.axhline(0, color=MUTED, lw=.8)
        ax.set_xlabel("conditioning c (true left-mode weight)")
    axs[0].set_ylabel("boundary shift  $b_N(c) - b(c)$")
    axs[0].set_title("Basin boundary displacement in noise space")
    axs[1].set_ylabel("left-mode probability error")
    axs[1].set_title("Induced error in mode probabilities")
    axs[1].legend(fontsize=8.5, ncol=2)
    fig.suptitle("Few-step Euler sampling moves the separatrix, so it changes which behaviour a prompt produces",
                 fontweight="semibold")
    fig.savefig("figs/fig5_euler_discretization.png", dpi=160)
    plt.close(fig)
    for N in Ns:
        print("  N", N, R["exp5"][N])

with open("results.json", "w") as f:
    json.dump(R, f, indent=1, default=float)
print("done")
