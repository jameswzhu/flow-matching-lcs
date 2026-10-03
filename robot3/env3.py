"""
Language-conditioned shapes task.

A point robot starts near the origin. Four goal objects, one of each kind
(red/blue x square/circle), sit at random angles on a ring of radius R around the
origin (minimum angular separation SEP_MIN). Their slot order in the observation is
shuffled every episode, so the policy must read the attributes, not the slot.

Prompt -> set of matching kinds. The expert picks a matching object uniformly and
drives straight to it at constant speed, then stays put:
    "blue square"       1 match
    "square" / "red"    2 matches
    "shape"             4 matches

R is sampled per episode from R_RANGE so that the object distance relative to the
chunk reach (H * STEP) is a deliberate variable: FTLE of one chunk only sees
decisions whose consequences land inside the chunk.

Observation (per object, in slot order): position relative to the robot / POS_SCALE,
colour (+1 red, -1 blue), shape (+1 square, -1 circle)  ->  4 x 4 = 16 numbers.
Policy output: action chunk of H = 12 displacements (24-D), / ACT_SCALE.
"""
import numpy as np

H, EXEC = 12, 4
STEP = 0.1                       # expert speed per step
ACT_SCALE = 0.1
REACH = H * STEP                 # distance covered by one chunk
R_RANGE = (0.8, 2.4)             # ring radius, in [0.67, 2] x chunk reach
SEP_MIN = np.deg2rad(50)
START_JIT = 0.3
T_EP = int(np.ceil((R_RANGE[1] + START_JIT) / STEP)) + 2
OBJ_R = 0.15                     # "reached" radius
POS_SCALE = 1.5
N_OBJ = 4
D_OBS = 4 * N_OBJ

# kind k = 2 * is_blue + is_circle  ->  0 red square, 1 red circle, 2 blue square, 3 blue circle
KINDS = [("red", "square"), ("red", "circle"), ("blue", "square"), ("blue", "circle")]
KIND_NAMES = [f"{c} {s}" for c, s in KINDS]


def _match(colours=("red", "blue"), shapes=("square", "circle")):
    return frozenset(k for k, (c, s) in enumerate(KINDS) if c in colours and s in shapes)


ALL = _match()
# canonical prompts (training set)
TRAIN_PROMPTS = {
    "red square": _match(["red"], ["square"]), "red circle": _match(["red"], ["circle"]),
    "blue square": _match(["blue"], ["square"]), "blue circle": _match(["blue"], ["circle"]),
    "square": _match(shapes=["square"]), "circle": _match(shapes=["circle"]),
    "red": _match(["red"]), "blue": _match(["blue"]),
    "shape": ALL,
}
HOLDOUT = "blue circle"
# zero-shot paraphrases and synonyms (never trained on), with their intended matches
TEST_PROMPTS = {
    "box": _match(shapes=["square"]), "red box": _match(["red"], ["square"]),
    "blue box": _match(["blue"], ["square"]), "round object": _match(shapes=["circle"]),
    "red disk": _match(["red"], ["circle"]), "blue disk": _match(["blue"], ["circle"]),
    "crimson square": _match(["red"], ["square"]), "navy circle": _match(["blue"], ["circle"]),
    "a red thing": _match(["red"]), "something blue": _match(["blue"]),
    "any object": ALL, "object": ALL,
}
ALL_PROMPTS = {**TRAIN_PROMPTS, **TEST_PROMPTS}


def train_prompts(holdout=False):
    return [p for p in TRAIN_PROMPTS if not (holdout and p == HOLDOUT)]


def specificity(prompt):
    return len(ALL_PROMPTS[prompt])


# ----------------------------------------------------------------------------- scenes
def make_scenes(n, rng, R=None):
    """objs [n, 4, 4] = (x, y, is_red, is_square) in shuffled slot order; R [n]."""
    R = rng.uniform(*R_RANGE, n) if R is None else np.broadcast_to(np.asarray(R, float), (n,)).copy()
    ang = np.empty((n, N_OBJ))
    for i in range(n):
        while True:
            a = np.sort(rng.uniform(0, 2 * np.pi, N_OBJ))
            gaps = np.diff(np.r_[a, a[0] + 2 * np.pi])
            if gaps.min() >= SEP_MIN:
                break
        ang[i] = rng.permutation(a)          # random angle for each kind
    kinds = np.stack([rng.permutation(N_OBJ) for _ in range(n)])    # slot -> kind
    a = np.take_along_axis(ang, kinds, 1)
    objs = np.stack([R[:, None] * np.cos(a), R[:, None] * np.sin(a),
                     (kinds < 2).astype(float), (kinds % 2 == 0).astype(float)], -1)
    return objs, R


def kind_of(objs):
    """[..., 4] slot attributes -> kind index."""
    return (2 * (1 - objs[..., 2]) + (1 - objs[..., 3])).astype(int)


def matches(objs, prompt):
    """[n, 4] bool: which slots match the prompt."""
    return np.isin(kind_of(objs), list(ALL_PROMPTS[prompt]))


def obs_features(pos, objs):
    """pos [n, 2], objs [n, 4, 4] -> [n, 16]."""
    rel = (objs[..., :2] - pos[:, None, :]) / POS_SCALE
    return np.concatenate([rel, 2 * objs[..., 2:4] - 1], -1).reshape(len(pos), -1)


# ----------------------------------------------------------------------------- expert
def straight_path(p0, p1, n=T_EP + 1, noise=0.0, rng=None):
    """Constant-speed (STEP) straight line from p0 to p1, then stay. [n, 2]."""
    d = p1 - p0; L = np.linalg.norm(d)
    s = np.minimum(STEP * np.arange(n), L)
    path = p0 + (s / max(L, 1e-9))[:, None] * d
    if noise > 0:
        moving = (s > 0) & (s < L)
        path[moving] += noise * rng.standard_normal((moving.sum(), 2))
    return path


def make_demos(n_ep, rng, prompts, noise=0.01):
    """Returns obs [N, 16], prompt index [N] (into `prompts`), chunks [N, 2H], and
    per-episode info (objs, prompt idx, target slot, path)."""
    objs, R = make_scenes(n_ep, rng)
    obs, pid, chunks, info = [], [], [], []
    for i in range(n_ep):
        p = rng.integers(len(prompts))
        slots = np.where(matches(objs[i:i + 1], prompts[p])[0])[0]
        tgt = rng.choice(slots)
        p0 = rng.uniform(-1, 1, 2) * START_JIT
        path = straight_path(p0, objs[i, tgt, :2], noise=noise, rng=rng)
        disp = np.vstack([np.diff(path, axis=0), np.zeros((H, 2))])
        n_move = int(np.ceil(np.linalg.norm(objs[i, tgt, :2] - p0) / STEP))
        for t in range(min(n_move + 2, T_EP)):               # a couple of "arrived" states
            obs.append(obs_features(path[t:t + 1], objs[i:i + 1])[0])
            chunks.append(disp[t:t + H].ravel() / ACT_SCALE)
            pid.append(p)
        info.append((objs[i], p, tgt, path))
    return np.array(obs), np.array(pid), np.array(chunks), info


# ----------------------------------------------------------------------------- labels
def chunk_positions(pos, chunk):
    return pos[:, None, :] + np.cumsum(chunk.reshape(len(pos), H, 2) * ACT_SCALE, 1)


def chunk_target(pos, chunk, objs):
    """Which slot the chunk heads for: nearest straight-line prototype (in chunk
    position space). Returns slot index [n]."""
    pts = chunk_positions(pos, chunk)                                         # [n, H, 2]
    dvec = objs[..., :2] - pos[:, None, :]                                    # [n, 4, 2]
    L = np.linalg.norm(dvec, axis=-1)
    s = np.minimum(STEP * np.arange(1, H + 1)[None, None, :], L[..., None])  # [n, 4, H]
    P = pos[:, None, None, :] + (s / np.maximum(L, 1e-9)[..., None])[..., None] * dvec[:, :, None, :]
    return np.argmin(((P - pts[:, None]) ** 2).sum((2, 3)), 1)


def first_reached(traj, objs):
    """traj [n, T, 2] -> first slot whose disk the trajectory enters (-1 if none),
    and the step index."""
    dist = np.linalg.norm(traj[:, :, None, :] - objs[:, None, :, :2], axis=-1)    # [n, T, 4]
    hit = dist < OBJ_R
    any_t = hit.any(2)
    t = np.where(any_t.any(1), np.argmax(any_t, 1), -1)
    slot = np.where(t >= 0, np.argmax(hit[np.arange(len(traj)), np.maximum(t, 0)], 1), -1)
    return slot, t


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    P = train_prompts()
    obs, pid, ch, info = make_demos(500, rng, P)
    print("demos", obs.shape, ch.shape, "| |a| mean", np.abs(ch).mean().round(3),
          "| T_EP", T_EP, "| chunk reach", REACH)
    paths = np.stack([p for *_, p in info]); objs = np.stack([o for o, *_ in info])
    tgt = np.array([t for _, _, t, _ in info])
    slot, _ = first_reached(paths, objs)
    print("first object reached == target:", (slot == tgt).mean())
    pos0 = paths[:, 0]
    lab = chunk_target(pos0, ((paths[:, 1:H + 1] - paths[:, :H]) / ACT_SCALE).reshape(len(paths), -1), objs)
    print("chunk label at start == target:", (lab == tgt).mean())
    for p in ["blue square", "square", "shape", "box"]:
        print(f"{p!r}: {specificity(p)} matches")
