"""
Locate the LCS of the generative flow from each seed and use it to predict mode
switches. Reuses the S3 cases (mc/prep.npz: 360 states x 24 seeds, first-chunk Jacobians).

    python lcs_eval.py scan      # line search along the LCS normal          (~25 min)
    python lcs_eval.py refine    # bisect separatrices; LCS velocity in g     (~12 min)
    python lcs_eval.py sweep     # fresh flips for several nudge sizes         (~6 min)
    python lcs_eval.py timing    # mid-course switches along rollouts          (~15 min)
    python lcs_eval.py analyze   # metrics + figures -> results_lcs.json
"""
import json, os, sys, time
import numpy as np
from policy2 import Policy, A, ftle
from env2 import norm_obs, chunk_pending_label, pending_column, X_START, H, EXEC, ACT_SCALE
from lcs import scan, crossings, bisect, ridges, labels, p_flip_noise, mass_beyond, flow_dir
from multichunk import flow_obs_jac
from eval2_rollout import obsn

STEPS = sys.argv[1:]
pol = Policy.load("policy2.npz")
os.makedirs("lcs", exist_ok=True)
S_GRID = np.linspace(-3, 3, 61); I0 = 30
t00 = time.time()


def log(m):
    print(f"{m}  [{time.time() - t00:.0f}s]", flush=True)


def top_right_sv(F):
    _, _, Vt = np.linalg.svd(F.astype(np.float64))
    return Vt[:, 0, :]


if "scan" in STEPS:
    P = np.load("mc/prep.npz")
    n = top_right_sv(P["F0"])
    lab, st = scan(pol, P["Z"], n, P["Sp"], P["gp"], S_GRID, log=log)
    l0 = chunk_pending_label(P["Sp"], P["a0"].astype(float))[1]
    log(f"scan done; label at s=0 matches first chunk: {(lab[:, I0] == l0).mean():.4f}")
    np.savez_compressed("lcs/scan.npz", n=n.astype(np.float32), lab=lab.astype(np.int8), st=st.astype(np.float32))

if "refine" in STEPS:
    P = np.load("mc/prep.npz"); Sc = np.load("lcs/scan.npz")
    n = Sc["n"].astype(float); lab = Sc["lab"]; st = Sc["st"].astype(float)
    Z, Sp, gp = P["Z"], P["Sp"], P["gp"]
    l0 = lab[:, I0]
    pp, mm = crossings(lab, S_GRID, I0)
    d_plus = bisect(pol, Z, n, Sp, gp, pp, l0)
    d_minus = bisect(pol, Z, -n, Sp, gp, -mm, l0)         # along -n: a positive distance
    rid = ridges(st, S_GRID, I0)
    log("separatrices refined")
    # LCS velocity in g at each crossing: ds*/dg = -(w . dPhi/dg) / |dPhi/ds|, w = unit(dPhi/ds)
    vel = np.full((len(Z), 2), np.nan)
    for side, d, sgn in [(0, d_plus, 1.0), (1, d_minus, -1.0)]:
        ok = ~np.isnan(d)
        zs = Z[ok] + (sgn * d[ok])[:, None] * n[ok]
        outs = [pol.sample_jac(zs[i:i + 1500].astype(np.float32), obsn(Sp[ok][i:i + 1500], gp[ok][i:i + 1500]), 20, with_g=True)
                for i in range(0, ok.sum(), 1500)]
        F = np.concatenate([o[1] for o in outs]).astype(float); fg = np.concatenate([o[2] for o in outs]).astype(float)
        Fn = np.einsum("nij,nj->ni", F, n[ok])
        w = Fn / np.linalg.norm(Fn, axis=1, keepdims=True)
        vel[ok, side] = -(w * fg).sum(1) / np.linalg.norm(Fn, axis=1)       # ds*/dg along +n
        log(f"LCS velocity side {side}")
    np.savez_compressed("lcs/refine.npz", d_plus=d_plus, d_minus=d_minus, ridge=rid, vel=vel)

if "sweep" in STEPS:
    P = np.load("mc/prep.npz"); Z, Sp, gp = P["Z"], P["Sp"], P["gp"]
    l0 = np.load("lcs/scan.npz")["lab"][:, I0]
    rng = np.random.default_rng(71)
    out = {}
    for eps in [0.03, 0.05, 0.1, 0.2, 0.4]:
        f = np.zeros(len(Z), bool)
        for _ in range(4):
            u = rng.standard_normal(Z.shape); u /= np.linalg.norm(u, axis=1, keepdims=True)
            f |= labels(pol, Z + eps * u, Sp, gp) != l0
        out[f"noise_{eps}"] = f; log(f"eps {eps}: flip rate {f.mean():.4f}")
    for dg in [0.02, 0.05, 0.1, 0.2]:
        f = (labels(pol, Z, Sp, gp + dg) != l0) | (labels(pol, Z, Sp, gp - dg) != l0)
        out[f"goal_{dg}"] = f; log(f"dg {dg}: flip rate {f.mean():.4f}")
    np.savez_compressed("lcs/sweep.npz", **out)

if "timing" in STEPS:
    # rollouts at g = 0.1; at every replan, locate the LCS from the seed actually drawn
    g0, n_ep, R_ = 0.1, 60, 15
    rng = np.random.default_rng(81)
    pos = np.column_stack([np.full(n_ep, X_START), rng.uniform(-0.3, 0.3, n_ep)])
    g = np.full(n_ep, g0)
    rows = []
    for k in range(R_):
        z = rng.standard_normal((n_ep, A)).astype(np.float32)
        a1, F, _ = pol.sample_jac(z, obsn(pos, g), 20)
        col, lab0, _ = chunk_pending_label(pos, a1.astype(float))
        rows.append(dict(k=k, pos=pos.copy(), z=z.astype(float), F=F.astype(float), col=col, lab=lab0))
        d = a1.reshape(n_ep, H, 2)[:, :EXEC].astype(float) * ACT_SCALE
        for j in range(EXEC):
            pos = pos + d[:, j]
    rows.append(dict(k=R_, pos=pos.copy(), col=pending_column(pos)))
    # flatten replans where a decision is pending
    idx = [(k, i) for k in range(R_) for i in range(n_ep) if rows[k]["col"][i] >= 0]
    K = np.array([k for k, _ in idx]); I = np.array([i for _, i in idx])
    pos_k = np.stack([rows[k]["pos"][i] for k, i in idx]); z_k = np.stack([rows[k]["z"][i] for k, i in idx])
    F_k = np.stack([rows[k]["F"][i] for k, i in idx]); col_k = np.array([rows[k]["col"][i] for k, i in idx])
    lab_k = np.array([rows[k]["lab"][i] for k, i in idx])
    pos_next = np.stack([rows[k + 1]["pos"][i] for k, i in idx])
    gk = np.full(len(idx), g0)
    log(f"timing: {len(idx)} pending replans")
    n = top_right_sv(F_k)
    lab, st = scan(pol, z_k, n, pos_k, gk, S_GRID)
    pp, mm = crossings(lab, S_GRID, I0)
    d_plus = bisect(pol, z_k, n, pos_k, gk, pp, lab[:, I0])
    d_minus = bisect(pol, z_k, -n, pos_k, gk, -mm, lab[:, I0])
    nz = (n * z_k).sum(1)
    # Gaussian mass on the far side of each LCS plane (plane through z + d n, normal n)
    m_plus = np.where(np.isnan(d_plus), 0, mass_beyond(nz + np.nan_to_num(d_plus)))
    m_minus = np.where(np.isnan(d_minus), 0, mass_beyond(-(nz - np.nan_to_num(d_minus))))
    lcs_mass = np.clip(m_plus + m_minus, 0, 1)
    log("timing: LCS located")
    # truth: fraction of fresh seeds choosing a different gap, at the current and at the next state
    nm = 256
    zz = np.random.default_rng(82).standard_normal((len(idx) * nm, A))
    lc = labels(pol, zz, np.repeat(pos_k, nm, 0), np.repeat(gk, nm)).reshape(-1, nm)
    mc_now = (lc != lab_k[:, None]).mean(1)
    still = pending_column(pos_next) == col_k
    ln = labels(pol, zz, np.repeat(pos_next, nm, 0), np.repeat(gk, nm)).reshape(-1, nm)
    mc_next = np.where(still, (ln != lab_k[:, None]).mean(1), np.nan)
    # realized: does the next replan in the actual rollout head for a different gap?
    nxt = {(k, i): j for j, (k, i) in enumerate(idx)}
    realized = np.array([lab_k[nxt[(k + 1, i)]] != lab_k[j] if (k + 1, i) in nxt and col_k[nxt[(k + 1, i)]] == col_k[j] else -1
                         for j, (k, i) in enumerate(idx)], dtype=float)
    realized[realized < 0] = np.nan
    np.savez_compressed("lcs/timing.npz", K=K, I=I, pos=pos_k, col=col_k, lab=lab_k, d_plus=d_plus, d_minus=d_minus,
                        lcs_mass=lcs_mass, mc_now=mc_now, mc_next=mc_next, realized=realized, ftle=ftle(F_k))
    log("timing done")

if "persist" in STEPS:
    # Rollouts that reuse ONE seed at every replan (a fixed noise schedule). The LCS moves
    # with the state; a mode switch happens when it sweeps past the fixed seed. Track the
    # seed's margin to the LCS at every replan and ask whether it forecasts the switch.
    n_ep, R_ = 240, 15
    rng = np.random.default_rng(91)
    g = rng.uniform(-0.6, 0.6, n_ep)
    pos = np.column_stack([np.full(n_ep, X_START), rng.uniform(-0.3, 0.3, n_ep)])
    z = rng.standard_normal((n_ep, A))
    rec = []
    for k in range(R_):
        a1, F, _ = pol.sample_jac(z.astype(np.float32), obsn(pos, g), 20)
        col, lab0, _ = chunk_pending_label(pos, a1.astype(float))
        live = col >= 0
        d_plus = np.full(n_ep, np.nan); d_minus = np.full(n_ep, np.nan)
        m_next = np.full(n_ep, np.inf)
        d = a1.reshape(n_ep, H, 2)[:, :EXEC].astype(float) * ACT_SCALE
        ds_exec = d.sum(1)                                   # state change before the next replan (known now)
        if live.any():
            n = top_right_sv(F[live])
            labs, _ = scan(pol, z[live], n, pos[live], g[live], S_GRID)
            pp, mm = crossings(labs, S_GRID, I0)
            d_plus[live] = bisect(pol, z[live], n, pos[live], g[live], pp, labs[:, I0])
            d_minus[live] = bisect(pol, z[live], -n, pos[live], g[live], -mm, labs[:, I0])
            # first-order motion of the LCS as the state moves: ds*/d(x,y) = -(w . dPhi/d(x,y)) / |dPhi/ds|
            li = np.where(live)[0]
            for dd, sgn in [(d_plus[live], 1.0), (d_minus[live], -1.0)]:
                ok = ~np.isnan(dd)
                if not ok.any():
                    continue
                rows_ = li[ok]
                zs = z[rows_] + (sgn * dd[ok])[:, None] * n[ok]
                _, Fn = flow_dir(pol, zs, n[ok], pos[rows_], g[rows_])
                _, Pobs = flow_obs_jac(pol, zs, pos[rows_], g[rows_])
                w_ = Fn / np.linalg.norm(Fn, axis=1, keepdims=True)
                v_xy = -np.einsum("na,nak->nk", w_, Pobs[:, :, :2].astype(float)) / np.linalg.norm(Fn, axis=1)[:, None]
                shift = (v_xy * ds_exec[rows_]).sum(1)          # boundary displacement along +n
                nxt = dd[ok] + sgn * shift                       # distance left after the move
                m_next[rows_] = np.fmin(m_next[rows_], nxt)
        rec.append(dict(pos=pos.copy(), col=col, lab=lab0, margin=np.fmin(np.nan_to_num(d_plus, nan=np.inf),
                                                                           np.nan_to_num(d_minus, nan=np.inf)),
                        margin_next=m_next, ftle=ftle(F)))
        for j in range(EXEC):
            pos = pos + d[:, j]
        log(f"persist replan {k}")
    np.savez_compressed("lcs/persist.npz", g=g, **{f"{key}": np.stack([r[key] for r in rec], 1)
                                                  for key in ["pos", "col", "lab", "margin", "margin_next", "ftle"]})

if "analyze" in STEPS:
    import plot_lcs
    plot_lcs.analyze()
log("done")
