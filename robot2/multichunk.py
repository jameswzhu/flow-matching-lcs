"""
Multi-replan (composed) sensitivity of the closed loop.

The receding-horizon closed loop is a map  s_{k+1} = s_k + E Phi(z_k; s_k, g):
Phi is the generative flow map (noise -> 24-D chunk) and E sums the first EXEC
displacements. Chaining Jacobians gives the sensitivity of every future replanning
state to the first noise seed z_0 and to the goal g:

  D_{k+1} = D_k + E ( [k=0] dPhi_0/dz_0  +  dPhi_k/ds  D_k )        (seed,  D: 2 x 24)
  d_{k+1} = d_k + E ( dPhi_k/dg          +  dPhi_k/ds  d_k )        (goal,  d: 2)

dPhi_k/d(x, y, g) comes from forward-mode tangents of the flow ODE (3 tangents per
row). The later seeds z_1, z_2, ... must be fixed to compose; we use
  "oracle": the actual stream used by the flip evaluation,
  "zero":   all-zero seeds (noise-free lookahead),
  "random": independent streams (what a deployed monitor could use).
"""
import numpy as np
from policy2 import A, D_IN, ftle
from mlp import silu, tau_features
from env2 import norm_obs, OBS_STD, EXEC, ACT_SCALE, COLS

XC = [c["x"] for c in COLS]          # column centres: where each route decision is read off

E = np.zeros((2, A))
for j in range(EXEC):
    E[0, 2 * j] = E[1, 2 * j + 1] = ACT_SCALE


def jvp_rows(net, x, Tin):
    """Network output and per-row JVPs. x: [N, d_in]; Tin: [N, k, d_in] -> y, [N, k, d_out]."""
    h, Th = x, Tin
    L = len(net.params)
    for i, (W, b) in enumerate(net.params):
        z = h @ W + b
        Tz = Th @ W
        if i < L - 1:
            h, d = silu(z)
            Th = Tz * d[:, None, :]
        else:
            h, Th = z, Tz
    return h, Th


def flow_obs_jac(pol, z, pos, g, n_steps=20):
    """Flow map a1 and its derivative w.r.t. raw (x, y, g): [N, A, 3]
    (forward-mode on the ODE: delta' = J_a delta + J_obs)."""
    N = len(z); dt = pol.dt
    obs_n = norm_obs(np.column_stack([pos, g])).astype(dt)
    a = z.astype(dt)
    dl = np.zeros((N, 3, A), dt)                       # tangent of a for each obs direction
    eo = np.zeros((3, 3 + 8), dt)
    for j in range(3):
        eo[j, j] = 1.0 / OBS_STD[j]
    h = 1.0 / n_steps

    def rhs(a, dl, t):
        x = np.concatenate([a, obs_n, tau_features(np.full(N, t))], 1).astype(dt)
        Tin = np.concatenate([dl, np.broadcast_to(eo, (N, 3, 11))], 2)
        return jvp_rows(pol.net, x, Tin)

    for i in range(n_steps):
        t = i * h
        k1 = rhs(a, dl, t); k2 = rhs(a + .5 * h * k1[0], dl + .5 * h * k1[1], t + .5 * h)
        k3 = rhs(a + .5 * h * k2[0], dl + .5 * h * k2[1], t + .5 * h); k4 = rhs(a + h * k3[0], dl + h * k3[1], t + h)
        a = a + h / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        dl = dl + h / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
    return a, np.transpose(dl, (0, 2, 1))


def composed(pol, pos0, g, z0, F0, P0, a0_chunk, stream, R, bs=3000):
    """Roll the closed loop forward R replans, composing sensitivities.

    pos0 [N,2], g [N], z0 [N,A] first seed, F0 = dPhi_0/dz0 [N,A,A], P0 = dPhi_0/d(x,y,g)
    [N,A,3] and a0_chunk the first chunk (all at the start state). `stream(k, N)` gives the
    seeds for replans k >= 1. Returns replan states [N,R+1,2], seed sensitivities
    D [N,R+1,2,A], goal sensitivities d [N,R+1,2], and the sensitivities of the
    route-deciding coordinate (y where the path crosses each column):
    cross_D [N,2,A], cross_d [N,2] (nan if that column was already passed or never reached)."""
    N = len(pos0)
    s = pos0.astype(float).copy()
    D = np.zeros((N, 2, A)); d = np.zeros((N, 2))
    S_, D_, d_ = [s.copy()], [D.copy()], [d.copy()]
    cross_D = np.full((N, len(XC), A), np.nan); cross_d = np.full((N, len(XC)), np.nan)
    done = np.stack([pos0[:, 0] >= xc for xc in XC], 1)
    # sensitivity of each future chunk (all 12 steps, executed or not) while a decision is pending
    ch_seed = np.full((N, R), -np.inf); ch_goal = np.full((N, R), -np.inf)
    for k in range(R):
        s_prev, D_prev, d_prev = s.copy(), D.copy(), d.copy()
        if k == 0:
            a1, Pk = a0_chunk.astype(float), P0.astype(float)
            dchunk_D = F0.astype(float) + Pk[:, :, :2] @ D
        else:
            zk = stream(k, N)
            outs = [flow_obs_jac(pol, zk[i:i + bs], s[i:i + bs], g[i:i + bs]) for i in range(0, N, bs)]
            a1 = np.concatenate([o[0] for o in outs]).astype(float)
            Pk = np.concatenate([o[1] for o in outs]).astype(float)
            dchunk_D = Pk[:, :, :2] @ D
        dchunk_d = Pk[:, :, 2] + np.einsum("nij,nj->ni", Pk[:, :, :2], d)
        live = s_prev[:, 0] < XC[-1]
        if live.any():
            ch_seed[live, k] = np.log(np.linalg.norm(dchunk_D[live], ord=2, axis=(1, 2)) + 1e-12)
            ch_goal[live, k] = np.log(np.linalg.norm(dchunk_d[live], axis=1) + 1e-12)
        D = D + E @ dchunk_D
        d = d + dchunk_d @ E.T
        disp = a1.reshape(N, -1, 2)[:, :EXEC] * ACT_SCALE        # same arithmetic as the rollout
        for j in range(EXEC):
            s = s + disp[:, j]
        for c, xc in enumerate(XC):
            hit = ~done[:, c] & (s[:, 0] >= xc)
            if hit.any():
                cross_D[hit, c], cross_d[hit, c] = _crossing_grad(xc, s_prev[hit], s[hit], D_prev[hit], D[hit],
                                                                  d_prev[hit], d[hit])
                done[hit, c] = True
        S_.append(s.copy()); D_.append(D.copy()); d_.append(d.copy())
    return np.stack(S_, 1), np.stack(D_, 1), np.stack(d_, 1), cross_D, cross_d, ch_seed, ch_goal


def _crossing_grad(xc, s0, s1, D0, D1, d0, d1):
    """Gradient of y_c = y0 + t (y1 - y0), t = (xc - x0) / (x1 - x0), w.r.t. seed (D) and goal (d)."""
    dx = s1[:, 0] - s0[:, 0]; dy = s1[:, 1] - s0[:, 1]
    t = (xc - s0[:, 0]) / dx

    def grad(J0x, J0y, J1x, J1y):
        sh = (slice(None),) + (None,) * (J0x.ndim - 1)
        dt = (-J0x * dx[sh] - (xc - s0[:, 0])[sh] * (J1x - J0x)) / (dx ** 2)[sh]
        return J0y + t[sh] * (J1y - J0y) + dy[sh] * dt

    return grad(D0[:, 0], D0[:, 1], D1[:, 0], D1[:, 1]), grad(d0[:, 0], d0[:, 1], d1[:, 0], d1[:, 1])


def scores_from(D, d, cross_D, cross_d, ch_seed, ch_goal):
    """Closed-loop sensitivity scores from composed Jacobians."""
    N, K = D.shape[:2]
    chunk = dict(seed_chunkmax=ch_seed.max(1), goal_chunkmax=ch_goal.max(1),
                 seed_chunkmax_future=ch_seed[:, 1:].max(1), goal_chunkmax_future=ch_goal[:, 1:].max(1),
                 seed_chunk_k=ch_seed.astype(np.float32), goal_chunk_k=ch_goal.astype(np.float32))
    cD = np.nan_to_num(np.linalg.norm(cross_D, axis=2)); cd = np.nan_to_num(np.abs(cross_d))
    cross = dict(seed_cross=np.log(np.sqrt((cD ** 2).sum(1)) + 1e-12), goal_cross=np.log(np.sqrt((cd ** 2).sum(1)) + 1e-12),
                 seed_cross_A=np.log(cD[:, 0] + 1e-12), seed_cross_B=np.log(cD[:, 1] + 1e-12),
                 goal_cross_A=np.log(cd[:, 0] + 1e-12), goal_cross_B=np.log(cd[:, 1] + 1e-12))
    stacked = D[:, 1:].reshape(N, -1, A)                                   # [N, 2K, A]
    s_stack = np.log(np.linalg.svd(stacked, compute_uv=False)[:, 0] + 1e-12)
    s_max = np.log(np.linalg.svd(D[:, 1:], compute_uv=False)[..., 0].max(1) + 1e-12)
    g_stack = np.log(np.sqrt((d[:, 1:] ** 2).sum((1, 2))) + 1e-12)
    g_max = np.log(np.linalg.norm(d[:, 1:], axis=2).max(1) + 1e-12)
    return dict(seed_stack=s_stack, seed_max=s_max, goal_stack=g_stack, goal_max=g_max, **cross, **chunk)
