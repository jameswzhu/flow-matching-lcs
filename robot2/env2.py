"""
Multi-obstacle task: two columns of vertically stacked obstacles.

  Column A (x = -1): 2 stacked disks -> 3 gaps (top / mid / bottom)
  Column B (x = +1): 3 stacked disks -> 2 inner gaps used (upper / lower)

A route is a pair (gap in A, gap in B): 6 routes, chosen by two decisions made at
different times. Conditioning ("prompt") = goal height g in [-1, 1].

Expert:  P(A = i | g)       = softmax_i( -(yA_i - 1.6 g)^2 / (2 * 1.0^2) )
         P(B = j | g, yA)   = softmax_j( -(yB_j - (0.6 g + 0.35 yA))^2 / (2 * 0.5^2) )
Path: monotone PCHIP y(x) through flat corridors at the chosen gap heights
(collision-free by construction), resampled at constant speed.

Policy output: action chunk of H = 12 displacements (24-D), / ACT_SCALE.
"""
import numpy as np
from scipy.interpolate import PchipInterpolator

X_START, X_GOAL = -2.5, 2.5
CORR = 0.5                                     # half-length of the flat corridor through a column
COLS = [
    dict(name="A", x=-1.0, obst=[(0.85, 0.40), (-0.85, 0.40)],
         gaps=np.array([1.65, 0.0, -1.65]), labels=["top", "mid", "bot"]),
    dict(name="B", x=1.0, obst=[(1.40, 0.35), (0.0, 0.35), (-1.40, 0.35)],
         gaps=np.array([0.70, -0.70]), labels=["upper", "lower"]),
]
GAPS_A, GAPS_B = COLS[0]["gaps"], COLS[1]["gaps"]
OBSTACLES = np.array([(c["x"], cy, r) for c in COLS for cy, r in c["obst"]])   # (cx, cy, r)
SIG_A, SIG_B = 1.0, 0.5
T_EP, H, EXEC = 60, 12, 4
ACT_SCALE = 0.125
OBS_MEAN = np.zeros(3)
OBS_STD = np.array([1.5, 1.0, 0.6])
ROUTE_NAMES = [f"{a}/{b}" for a in COLS[0]["labels"] for b in COLS[1]["labels"]]


def _softmax(z):
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def probs_A(g):
    g = np.atleast_1d(np.asarray(g, float))
    return _softmax(-(GAPS_A[None] - 1.6 * g[:, None]) ** 2 / (2 * SIG_A ** 2))


def probs_B(g, yA):
    g, yA = np.broadcast_arrays(np.atleast_1d(np.asarray(g, float)), np.atleast_1d(np.asarray(yA, float)))
    t = 0.6 * g + 0.35 * yA
    return _softmax(-(GAPS_B[None] - t[:, None]) ** 2 / (2 * SIG_B ** 2))


def route_probs_expert(g):
    """[3, 2] matrix of P(A = i, B = j | g)."""
    pA = probs_A(g)[0]
    return np.array([pA[i] * probs_B(g, GAPS_A[i])[0] for i in range(3)])


def expert_path(y0, g, iA, iB, n=T_EP + 1):
    xa, xb = COLS[0]["x"], COLS[1]["x"]
    kx = [X_START, xa - CORR, xa + CORR, xb - CORR, xb + CORR, X_GOAL]
    ky = [y0, GAPS_A[iA], GAPS_A[iA], GAPS_B[iB], GAPS_B[iB], g]
    f = PchipInterpolator(kx, ky)
    xs = np.linspace(X_START, X_GOAL, 6000)
    pts = np.stack([xs, f(xs)], 1)
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    si = np.linspace(0, s[-1], n)
    return np.stack([np.interp(si, s, pts[:, 0]), np.interp(si, s, pts[:, 1])], 1)


def make_demos(n_ep, rng, noise=0.01):
    obs, chunks, info = [], [], []
    for _ in range(n_ep):
        y0 = rng.uniform(-0.3, 0.3); g = rng.uniform(-1, 1)
        iA = rng.choice(3, p=probs_A(g)[0])
        iB = rng.choice(2, p=probs_B(g, GAPS_A[iA])[0])
        path = expert_path(y0, g, iA, iB)
        path[1:-1] += noise * rng.standard_normal(path[1:-1].shape)
        disp = np.vstack([np.diff(path, axis=0), np.zeros((H, 2))])
        for t in range(T_EP):
            obs.append([path[t, 0], path[t, 1], g])
            chunks.append(disp[t:t + H].ravel() / ACT_SCALE)
        info.append((g, iA, iB, path))
    return np.array(obs), np.array(chunks), info


def norm_obs(obs):
    return (np.asarray(obs) - OBS_MEAN) / OBS_STD


# ----------------------------------------------------------------------------- labels
def _y_at_x(traj, xq):
    """y where each trajectory first crosses x = xq (linear interp); nan if never."""
    x = traj[..., 0]
    hit = x >= xq
    any_ = hit.any(1)
    k = np.argmax(hit, 1)
    k = np.clip(k, 1, traj.shape[1] - 1)
    i = np.arange(len(traj))
    x0, x1 = x[i, k - 1], x[i, k]; y0, y1 = traj[i, k - 1, 1], traj[i, k, 1]
    w = np.clip((xq - x0) / np.where(np.abs(x1 - x0) < 1e-9, 1e-9, x1 - x0), 0, 1)
    y = y0 + w * (y1 - y0)
    y[~any_] = np.nan
    first_already = x[:, 0] >= xq
    y[first_already] = traj[first_already, 0, 1]
    return y


def route_of_traj(traj):
    """Returns (iA, iB, route index 0..5 or -1 if a column was never reached)."""
    yA = _y_at_x(traj, COLS[0]["x"]); yB = _y_at_x(traj, COLS[1]["x"])
    iA = np.where(np.isnan(yA), -1, np.argmin(np.abs(np.nan_to_num(yA)[:, None] - GAPS_A[None]), 1))
    iB = np.where(np.isnan(yB), -1, np.argmin(np.abs(np.nan_to_num(yB)[:, None] - GAPS_B[None]), 1))
    r = np.where((iA >= 0) & (iB >= 0), 2 * iA + iB, -1)
    return iA, iB, r


def collisions(traj, sub=6):
    """Densify each trajectory and test against every disk."""
    t = np.linspace(0, 1, sub, endpoint=False)
    seg = traj[:, :-1, None, :] + t[None, None, :, None] * np.diff(traj, axis=1)[:, :, None, :]
    pts = seg.reshape(len(traj), -1, 2)
    d = np.linalg.norm(pts[:, :, None, :] - OBSTACLES[None, None, :, :2], axis=-1) - OBSTACLES[None, None, :, 2]
    return d.min((1, 2)) < 0


def pending_column(pos):
    """0 if the A decision is still ahead, 1 if B is ahead, -1 if both are passed."""
    x = pos[:, 0]
    return np.where(x < COLS[0]["x"] - CORR, 0, np.where(x < COLS[1]["x"] - CORR, 1, -1))


def _prototypes(pos, col, step=0.101, ngrid=240):
    """Expert-like chunks from the current state to each gap of the pending column:
    hold y, smoothstep to the gap height before the column entry, then stay flat;
    resampled at constant speed. Returns [N, K, H, 2] positions (K = gaps)."""
    N = len(pos)
    gaps = np.where(col[:, None] == 0, np.r_[GAPS_A][None], np.r_[GAPS_B, np.nan][None])   # [N, 3]
    xa_e, xb_s, xb_e = COLS[0]["x"] - CORR, COLS[0]["x"] + CORR, COLS[1]["x"] - CORR
    x_s = np.where(col == 0, pos[:, 0], np.maximum(pos[:, 0], xb_s))
    x_e = np.where(col == 0, xa_e, xb_e)
    span = np.maximum(x_e - x_s, 1e-3)
    xg = pos[:, 0:1] + np.linspace(0, 1, ngrid)[None] * (H * step * 1.2)                     # [N, G]
    u = np.clip((xg - x_s[:, None]) / span[:, None], 0, 1)
    s = 3 * u ** 2 - 2 * u ** 3
    out = np.full((N, 3, H, 2), np.nan)
    for k in range(3):
        tgt = gaps[:, k]
        if np.all(np.isnan(tgt)):
            continue
        yg = pos[:, 1:2] + (np.nan_to_num(tgt)[:, None] - pos[:, 1:2]) * s
        seg = np.sqrt(np.diff(xg, axis=1) ** 2 + np.diff(yg, axis=1) ** 2)
        arc = np.concatenate([np.zeros((N, 1)), np.cumsum(seg, 1)], 1)
        tq = step * np.arange(1, H + 1)
        for n in range(N):
            out[n, k, :, 0] = np.interp(tq, arc[n], xg[n]); out[n, k, :, 1] = np.interp(tq, arc[n], yg[n])
        out[np.isnan(tgt), k] = np.nan
    return out


def chunk_pending_label(pos, chunk):
    """Which gap of the next column the chunk heads for: the nearest expert-like
    prototype (in chunk-position space). Returns (column, gap index, reached-entry)."""
    N = len(pos)
    pts = pos[:, None, :] + np.cumsum(chunk.reshape(N, H, 2) * ACT_SCALE, 1)
    col = pending_column(pos)
    lab = np.full(N, -1)
    reached = np.zeros(N, bool)
    m = col >= 0
    if m.any():
        P = _prototypes(pos[m], col[m])
        d = np.nansum((P - pts[m][:, None]) ** 2, axis=(2, 3))
        d[np.isnan(P[:, :, 0, 0])] = np.inf
        lab[m] = np.argmin(d, 1)
        xe = np.where(col[m] == 0, COLS[0]["x"] - CORR, COLS[1]["x"] - CORR)
        reached[m] = pts[m][:, -1, 0] >= xe
    return col, lab, reached


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    obs, ch, info = make_demos(600, rng)
    paths = np.stack([p for *_, p in info])
    iA, iB, r = route_of_traj(paths)
    true_r = np.array([2 * a + b for _, a, b, _ in info])
    d = np.linalg.norm(paths[:, :, None, :] - OBSTACLES[None, None, :, :2], axis=-1) - OBSTACLES[None, None, :, 2]
    print("demos", obs.shape, ch.shape, "| route label accuracy", (r == true_r).mean(),
          "| collisions", collisions(paths).sum(), "| min clearance", d.min().round(3),
          "| step len", np.linalg.norm(np.diff(paths, axis=1), axis=2).mean().round(3))
    for g in [-0.5, 0.0, 0.1, 0.5]:
        print(f"g={g:+.1f} expert P(route)", np.round(route_probs_expert(g).ravel(), 3))
