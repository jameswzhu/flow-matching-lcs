"""
Toy robot task: a 2D point robot must get from the left to a goal on the right,
passing a disk obstacle either OVER or UNDER it.

Conditioning ("prompt") = goal height g in [-1, 1].
Expert: picks the top route with probability sigmoid(K * g), then follows a smooth
quadratic-Bezier path through a waypoint above/below the obstacle.

Policy output = action chunk of H future displacement vectors (2H = 16 dims),
normalised by ACT_SCALE so actions are O(1), like the N(0, I) noise.
"""
import numpy as np

OBST_C = np.array([0.0, 0.0])
OBST_R = 0.6
WAY_Y = 1.1          # waypoint height above / below the obstacle
K_PREF = 3.0         # P(top | g) = sigmoid(K_PREF * g)
T_EP = 40            # steps per expert episode
H = 8                # action-chunk length
EXEC = 4             # actions executed per replan (receding horizon)
ACT_SCALE = 0.125    # typical displacement per step
OBS_MEAN = np.array([0.0, 0.0, 0.0])
OBS_STD = np.array([1.2, 0.8, 0.6])


def p_top_expert(g):
    return 1.0 / (1.0 + np.exp(-K_PREF * np.asarray(g)))


def bezier_path(start, goal, top, n=T_EP + 1, dense=2000):
    w = np.array([0.0, WAY_Y if top else -WAY_Y])
    P1 = 2 * w - 0.5 * (start + goal)          # curve passes through w at t=0.5
    t = np.linspace(0, 1, dense)[:, None]
    pts = (1 - t) ** 2 * start + 2 * (1 - t) * t * P1 + t ** 2 * goal
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    si = np.linspace(0, s[-1], n)              # constant-speed resampling
    return np.stack([np.interp(si, s, pts[:, 0]), np.interp(si, s, pts[:, 1])], 1)


def make_demos(n_ep, rng, noise=0.01):
    """Returns obs [N,3] = (x, y, g) and chunks [N, 2H] (normalised), plus episode info."""
    obs, chunks, info = [], [], []
    for _ in range(n_ep):
        start = np.array([-2.0, rng.uniform(-0.25, 0.25)])
        g = rng.uniform(-1, 1)
        top = rng.random() < p_top_expert(g)
        path = bezier_path(start, np.array([2.0, g]), top)
        path = path + noise * rng.standard_normal(path.shape) * np.r_[0, np.ones(T_EP - 1), 0][:, None]
        disp = np.diff(path, axis=0)                               # [T, 2]
        disp = np.vstack([disp, np.zeros((H, 2))])                 # stay at goal after the end
        for t in range(T_EP):
            obs.append([path[t, 0], path[t, 1], g])
            chunks.append(disp[t:t + H].ravel() / ACT_SCALE)
        info.append((g, top, path))
    return np.array(obs), np.array(chunks), info


def norm_obs(obs):
    return (obs - OBS_MEAN) / OBS_STD


def chunk_route_label(chunks):
    """+1 = heading over the obstacle, -1 = under, judged from the first half of
    the chunk (only meaningful at states left of the obstacle)."""
    dy = chunks.reshape(-1, H, 2)[:, :H // 2, 1].sum(1)
    return np.where(dy > 0, 1, -1)


def min_clearance(path):
    return np.linalg.norm(path - OBST_C, axis=1).min() - OBST_R


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    obs, ch, info = make_demos(400, rng)
    clear = np.array([min_clearance(p) for _, _, p in info])
    tops = np.array([t for _, t, _ in info])
    print("demos", obs.shape, ch.shape, "min clearance", clear.min().round(3),
          "chunk |a| mean", np.abs(ch).mean().round(3))
    starts = obs[::T_EP]; lab = chunk_route_label(ch[::T_EP])
    print("label agrees with route at start:", np.mean((lab == 1) == tops))
