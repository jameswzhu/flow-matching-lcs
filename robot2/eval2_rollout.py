"""Closed-loop rollout shared by mc_eval.py (same arithmetic as eval2.rollout)."""
import numpy as np
from policy2 import A
from env2 import norm_obs, H, EXEC, ACT_SCALE, X_GOAL


def obsn(pos, g):
    return norm_obs(np.column_stack([pos, np.broadcast_to(np.asarray(g, float), (len(pos),))]))


def rollout(pol, pos0, g, seed, first_noise=None, R_=None, bs=20000, NS=20):
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
    states.append(pos.copy())
    return np.stack(traj, 1), np.stack(states, 1)
