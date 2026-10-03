"""Shared helpers for evaluating the trained policy."""
import numpy as np
from policy import Policy, A
from robot_task import (norm_obs, chunk_route_label, p_top_expert, ACT_SCALE, EXEC,
                        T_EP, H, OBST_C, OBST_R)

START = np.array([-2.0, 0.0])


def p_top_open_loop(pol, g, n=10000, rng=None, n_steps=30, method="rk4", pos=START):
    rng = rng or np.random.default_rng(0)
    a0 = rng.standard_normal((n, A))
    a1 = pol.sample(a0, norm_obs(np.r_[pos, g]), n_steps, method)
    return float((chunk_route_label(a1) == 1).mean())


def rollout(pol, g, n_ep=300, rng=None, n_steps=30, method="rk4", ys_range=0.25):
    """Closed-loop receding-horizon rollouts. Returns route (+1 over / -1 under),
    collision flag, final distance to goal, and the trajectories."""
    rng = rng or np.random.default_rng(1)
    g = np.broadcast_to(np.asarray(g, float), (n_ep,))
    pos = np.stack([np.full(n_ep, -2.0), rng.uniform(-ys_range, ys_range, n_ep)], 1)
    traj = [pos.copy()]
    for _ in range(T_EP // EXEC + 2):
        ob = norm_obs(np.column_stack([pos, g]))
        ch = pol.sample(rng.standard_normal((n_ep, A)), ob, n_steps, method)
        disp = ch.reshape(n_ep, H, 2)[:, :EXEC] * ACT_SCALE
        for k in range(EXEC):
            pos = pos + disp[:, k]
            traj.append(pos.copy())
    traj = np.stack(traj, 1)                                   # [n, T, 2]
    dist = np.linalg.norm(traj - OBST_C, axis=2)
    collide = dist.min(1) < OBST_R
    i0 = np.argmin(np.abs(traj[:, :, 0]), axis=1)            # where x is closest to 0
    route = np.sign(traj[np.arange(n_ep), i0, 1])
    goal = np.column_stack([np.full(n_ep, 2.0), g])
    final_err = np.linalg.norm(traj[:, -1] - goal, axis=1)
    return route, collide, final_err, traj
