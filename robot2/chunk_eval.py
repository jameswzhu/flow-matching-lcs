"""
How well does the LCS predict a mode switch of its OWN chunk (open loop, the chunk the
LCS was computed for; no closed loop, no later replans)?

Reuses the 8,640 cases of S3 (mc/prep.npz) and the LCS located in lcs_eval.py
(lcs/scan.npz, lcs/refine.npz, lcs/sweep.npz).

    python chunk_eval.py dirs      # per-perturbation tests near the LCS: flatness, destination (~10 min)
    python chunk_eval.py goal      # exact goal margins by bisection vs linear prediction       (~4 min)
    python chunk_eval.py analyze   # breakdowns, label-free predictor, figures -> results_chunk.json
"""
import json, os, sys, time
import numpy as np
from policy2 import Policy, A
from env2 import chunk_pending_label
from lcs import labels, p_flip_noise

STEPS = sys.argv[1:]
pol = Policy.load("policy2.npz")
os.makedirs("chunk", exist_ok=True)
S_GRID = np.linspace(-3, 3, 61); I0 = 30
ANGLES = [0, 30, 45, 60, 75, 85]
t00 = time.time()


def log(m):
    print(f"{m}  [{time.time() - t00:.0f}s]", flush=True)


def load_lcs():
    P = np.load("mc/prep.npz"); Sc = np.load("lcs/scan.npz"); Rf = np.load("lcs/refine.npz")
    dp, dm = Rf["d_plus"], Rf["d_minus"]
    margin = np.fmin(np.nan_to_num(dp, nan=np.inf), np.nan_to_num(dm, nan=np.inf))
    return P, Sc, Rf, dp, dm, margin


def dest_along_normal(lab_row_scan):
    """Label just beyond the first label change on each side of s = 0 along the normal."""
    l0 = lab_row_scan[:, I0]
    plus = np.full(len(l0), -1); minus = np.full(len(l0), -1)
    for i, row in enumerate(lab_row_scan):
        r = np.where(row[I0 + 1:] != l0[i])[0]
        if len(r):
            plus[i] = row[I0 + 1 + r[0]]
        r = np.where(row[:I0][::-1] != l0[i])[0]
        if len(r):
            minus[i] = row[I0 - 1 - r[0]]
    return plus, minus


def line_cross(z, u, pos, g, l0, s_max, K=33, iters=10):
    """First label change along z + s u for s in (0, s_max] (per row). Returns crossing
    distance (nan if none) and the label just beyond it."""
    N = len(z)
    S = np.linspace(0, 1, K)[None, 1:] * s_max[:, None]                     # [N, K-1]
    labs = np.stack([labels(pol, z + S[:, j:j + 1] * u, pos, g) for j in range(K - 1)], 1)
    diff = labs != l0[:, None]
    has = diff.any(1)
    j = np.argmax(diff, 1)
    hi = S[np.arange(N), j]; lo = np.where(j > 0, S[np.arange(N), np.maximum(j - 1, 0)], 0.0)
    dest = labs[np.arange(N), j]
    if has.any():
        ii = np.where(has)[0]
        lo_, hi_ = lo[ii].copy(), hi[ii].copy()
        for _ in range(iters):
            mid = 0.5 * (lo_ + hi_)
            lm = labels(pol, z[ii] + mid[:, None] * u[ii], pos[ii], g[ii])
            same = lm == l0[ii]
            lo_ = np.where(same, mid, lo_); hi_ = np.where(same, hi_, mid)
        hi[ii] = 0.5 * (lo_ + hi_)
    return np.where(has, hi, np.nan), np.where(has, dest, -1)


if "dirs" in STEPS:
    P, Sc, Rf, dp, dm, margin = load_lcs()
    sel = np.where(margin < 1.0)[0]
    Z, Sp, gp = P["Z"][sel], P["Sp"][sel], P["gp"][sel]
    n = Sc["n"][sel].astype(float); lab_scan = Sc["lab"][sel].astype(int)
    l0 = lab_scan[:, I0]
    near_plus = np.nan_to_num(dp[sel], nan=np.inf) <= np.nan_to_num(dm[sel], nan=np.inf)
    sgn = np.where(near_plus, 1.0, -1.0); d = margin[sel]
    pdest_p, pdest_m = dest_along_normal(lab_scan)
    pred_dest = np.where(near_plus, pdest_p, pdest_m)
    rng = np.random.default_rng(101)
    # (1) controlled angles to the LCS normal, pointing at the nearest LCS
    cross = np.full((len(sel), len(ANGLES)), np.nan); dest = np.full((len(sel), len(ANGLES)), -1)
    for a, th in enumerate(ANGLES):
        t = rng.standard_normal(Z.shape); t -= (t * n).sum(1, keepdims=True) * n
        t /= np.linalg.norm(t, axis=1, keepdims=True)
        c = np.cos(np.deg2rad(th)); s = np.sin(np.deg2rad(th))
        u = sgn[:, None] * (c * n + s * t)
        s_pred = d / c
        s_max = np.minimum(4.0, np.maximum(2.0 * s_pred, 0.5))
        cross[:, a], dest[:, a] = line_cross(Z, u, Sp, gp, l0, s_max)
        log(f"angle {th}: found {np.isfinite(cross[:, a]).mean():.3f}")
    # (2) per-perturbation test with random directions (8 per seed per size)
    eps_list = [0.1, 0.2, 0.4]; ndir = 8
    U = rng.standard_normal((ndir, len(sel), A)); U /= np.linalg.norm(U, axis=2, keepdims=True)
    proj = (U * n[None]).sum(2)                                                 # u . n
    flip = np.zeros((len(eps_list), ndir, len(sel)), bool)
    for e, eps in enumerate(eps_list):
        for k in range(ndir):
            flip[e, k] = labels(pol, Z + eps * U[k], Sp, gp) != l0
        log(f"random eps {eps}")
    np.savez_compressed("chunk/dirs.npz", sel=sel, d=d, sgn=sgn, cross=cross, dest=dest, pred_dest=pred_dest,
                        l0=l0, proj=proj, flip=flip, eps=np.array(eps_list), angles=np.array(ANGLES),
                        dp=dp[sel], dm=dm[sel])

if "goal" in STEPS:
    P, Sc, Rf, dp, dm, margin = load_lcs()
    Sw = np.load("lcs/sweep.npz"); vel = Rf["vel"]
    # predicted signed goal change that brings each boundary to the seed (first order)
    dg_plus = -dp / vel[:, 0]                  # boundary at +d_plus moves with ds*/dg = vel+
    dg_minus = dm / vel[:, 1]                  # boundary at -d_minus moves with ds*/dg = vel-
    cand = np.stack([dg_plus, dg_minus], 1)
    up = np.nanmin(np.where(cand > 0, cand, np.nan), 1); dn = -np.nanmax(np.where(cand < 0, cand, np.nan), 1)
    up = np.nan_to_num(up, nan=np.inf); dn = np.nan_to_num(dn, nan=np.inf)
    sel = np.where((np.fmin(up, dn) < 0.6) | Sw["goal_0.2"])[0]
    Z, Sp, gp = P["Z"][sel], P["Sp"][sel], P["gp"][sel]
    l0 = labels(pol, Z, Sp, gp)
    G = np.linspace(0, 0.6, 31)[1:]
    true = {}
    for side, sg in [("up", 1.0), ("down", -1.0)]:
        labs = np.stack([labels(pol, Z, Sp, gp + sg * gv) for gv in G], 1)
        diff = labs != l0[:, None]; has = diff.any(1); j = np.argmax(diff, 1)
        hi = G[j]; lo = np.where(j > 0, G[np.maximum(j - 1, 0)], 0.0)
        ii = np.where(has)[0]; lo_, hi_ = lo[ii].copy(), hi[ii].copy()
        for _ in range(10):
            mid = 0.5 * (lo_ + hi_)
            same = labels(pol, Z[ii], Sp[ii], gp[ii] + sg * mid) == l0[ii]
            lo_ = np.where(same, mid, lo_); hi_ = np.where(same, hi_, mid)
        tr = np.full(len(sel), np.inf); tr[ii] = 0.5 * (lo_ + hi_)
        true[side] = tr
        log(f"goal {side}: flips within 0.6 for {has.mean():.3f}")
    np.savez_compressed("chunk/goal.npz", sel=sel, true_up=true["up"], true_dn=true["down"],
                        pred_up=up[sel], pred_dn=dn[sel], margin=margin[sel])

if "lf" in STEPS:
    # Label-free LCS: along the LCS normal, find where the chunk itself jumps. A ridge
    # narrower than the grid can hide its peak stretch, but not the jump it causes.
    from eval2_rollout import obsn
    P, Sc, Rf, dp, dm, margin = load_lcs()
    Z, Sp, gp = P["Z"], P["Sp"], P["gp"]; n = Sc["n"].astype(float); N = len(Z)
    ob = obsn(Sp, gp)

    def chunks(zz, bs=10000):
        return np.concatenate([pol.sample(zz[i:i + bs].astype(np.float32), ob[i:i + bs], 20)
                               for i in range(0, len(zz), bs)]).astype(np.float32)

    jumps = np.empty((N, len(S_GRID) - 1), np.float32)
    prev = chunks(Z + S_GRID[0] * n)
    for j in range(1, len(S_GRID)):
        cur = chunks(Z + S_GRID[j] * n)
        jumps[:, j - 1] = np.linalg.norm(cur - prev, axis=1); prev = cur
        if j % 10 == 0:
            log(f"lf scan {j}/{len(S_GRID) - 1}")
    np.savez_compressed("chunk/lf_jumps.npz", jumps=jumps)
    log("lf scan saved")

if "lf_refine" in STEPS:
    # refine the largest jump on each side by label-free bisection (keep the half that
    # changes the chunk more), for every row where that jump exceeds a low floor
    from eval2_rollout import obsn
    P, Sc, Rf, dp, dm, margin = load_lcs()
    Z, Sp, gp = P["Z"], P["Sp"], P["gp"]; n = Sc["n"].astype(float)
    J = np.load("chunk/lf_jumps.npz")["jumps"].astype(float)
    out = {}
    for side in ["plus", "minus"]:
        Js = J[:, I0:] if side == "plus" else J[:, :I0][:, ::-1]          # intervals moving away from s=0
        k = np.argmax(Js, 1); jmax = Js[np.arange(len(J)), k]
        sg = 1.0 if side == "plus" else -1.0
        lo = sg * (S_GRID[1] - S_GRID[0]) * k; hi = lo + sg * (S_GRID[1] - S_GRID[0])
        ii = np.where(jmax > 0.05)[0]
        ob = obsn(Sp[ii], gp[ii])
        lo_, hi_ = lo[ii].copy(), hi[ii].copy()
        f = lambda s: pol.sample((Z[ii] + s[:, None] * n[ii]).astype(np.float32), ob, 20).astype(float)
        a_lo, a_hi = f(lo_), f(hi_)
        for _ in range(12):
            mid = 0.5 * (lo_ + hi_); a_mid = f(mid)
            left = np.linalg.norm(a_mid - a_lo, axis=1) >= np.linalg.norm(a_hi - a_mid, axis=1)
            hi_ = np.where(left, mid, hi_); a_hi = np.where(left[:, None], a_mid, a_hi)
            lo_ = np.where(left, lo_, mid); a_lo = np.where(left[:, None], a_lo, a_mid)
        pos = np.full(len(J), np.nan); pos[ii] = np.abs(0.5 * (lo_ + hi_))
        out[f"pos_{side}"] = pos; out[f"jump_{side}"] = jmax
        log(f"lf refine {side}: {len(ii)} rows")
    np.savez_compressed("chunk/lf.npz", **out)

if "analyze" in STEPS:
    import plot_chunk
    plot_chunk.analyze()
log("done")
