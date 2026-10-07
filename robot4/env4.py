"""
Language-conditioned shapes task for pi0: robot3's task, seen through a camera.

The scene, prompts, match sets, expert and chunk / closed-loop labels are robot3's
(imported from ../robot3/env3.py, so the two studies stay comparable). What changes is
the observation: instead of 16 object features, pi0 gets

    image   224 x 224 top-down RGB render of the table (objects + robot), its only
            source of object positions, colours and shapes
    state   robot position / POS_SCALE, zero-padded to 32
    prompt  an instruction built from the robot3 prompt ("reach the blue square"),
            tokenised and encoded by pi0's own PaliGemma

so binding the words to the pixels is pi0's job. Actions are robot3's: a chunk of H = 12
displacements / ACT_SCALE (24 numbers), zero-padded to pi0's 32 action dims.
"""
import os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "robot3"))
from env3 import (H, EXEC, STEP, ACT_SCALE, REACH, R_RANGE, START_JIT, T_EP, OBJ_R, POS_SCALE,   # noqa: E402,F401
                  N_OBJ, KINDS, KIND_NAMES, TRAIN_PROMPTS, TEST_PROMPTS, ALL_PROMPTS, HOLDOUT,
                  train_prompts, specificity, make_scenes, kind_of, matches, straight_path,
                  chunk_positions, chunk_target, first_reached)

# ----------------------------------------------------------------------------- language
# robot3 prompts are bare noun phrases; pi0 was pre-trained on instructions, so wrap them.
_TEXT = {"red": "reach the red object", "blue": "reach the blue object",
         "a red thing": "reach a red thing", "something blue": "reach something blue",
         "any object": "reach any object"}


def instruction(prompt):
    return _TEXT.get(prompt, f"reach the {prompt}")


# ----------------------------------------------------------------------------- camera
IMG = 224
VIEW = 2.8                       # image spans [-VIEW, VIEW]^2 in world units (y up)
OBJ_HALF = 0.2                   # drawn half-size of an object (square half-side / circle radius)
ROBOT_R = 0.11
TABLE = (0.86, 0.84, 0.80)
RED, BLUE = (0.85, 0.16, 0.14), (0.13, 0.32, 0.86)
ROBOT = (0.15, 0.15, 0.15)


def render(pos, objs, device="cpu"):
    """pos [N, 2], objs [N, 4, 4] (x, y, is_red, is_square) -> images [N, 3, IMG, IMG] in
    [0, 1], float32 on `device`. Anti-aliased with a one-pixel linear ramp. Vectorised, so
    a batch of 256 renders in a few ms on the GPU."""
    pos = torch.as_tensor(np.asarray(pos), dtype=torch.float32, device=device)
    objs = torch.as_tensor(np.asarray(objs), dtype=torch.float32, device=device)
    px = 2 * VIEW / IMG
    c = (torch.arange(IMG, device=device, dtype=torch.float32) + 0.5) * px - VIEW
    X, Y = c[None, :], c.flip(0)[:, None]                          # row 0 = top = +y
    img = torch.tensor(TABLE, device=device)[None, :, None, None].expand(len(pos), 3, IMG, IMG).clone()
    red, blue = torch.tensor(RED, device=device), torch.tensor(BLUE, device=device)

    def paint(alpha, col):                                         # alpha [N, IMG, IMG], col [N, 3]
        nonlocal img
        img = img * (1 - alpha[:, None]) + col[:, :, None, None] * alpha[:, None]

    for j in range(objs.shape[1]):
        o = objs[:, j]
        dx = X[None] - o[:, 0, None, None]; dy = Y[None] - o[:, 1, None, None]
        d_sq = torch.maximum(dx.abs(), dy.abs())                    # Chebyshev: square
        d_ci = torch.sqrt(dx ** 2 + dy ** 2)                        # Euclidean: circle
        d = torch.where(o[:, 3, None, None] > 0.5, d_sq, d_ci)
        alpha = ((OBJ_HALF - d) / px + 0.5).clamp(0, 1)
        col = torch.where(o[:, 2, None] > 0.5, red[None], blue[None])
        paint(alpha, col)
    d = torch.sqrt((X[None] - pos[:, 0, None, None]) ** 2 + (Y[None] - pos[:, 1, None, None]) ** 2)
    paint(((ROBOT_R - d) / px + 0.5).clamp(0, 1), torch.tensor(ROBOT, device=device).expand(len(pos), 3))
    return img


def state_vec(pos):
    """Robot position -> pi0 state (before zero-padding to 32)."""
    return np.asarray(pos, float) / POS_SCALE


# ----------------------------------------------------------------------------- demos
def make_demos(n_ep, rng, prompts, noise=0.01, n_arrive=2, arrive_jit=0.0):
    """Like env3.make_demos, but keeps the absolute robot position and the scene (the
    image is rendered at batch time, on the GPU). Returns pos [N, 2], scene index [N]
    into objs, objs [n_ep, 4, 4], prompt index [N] into `prompts`, chunks [N, 2H].

    n_arrive "arrived" states per episode follow the moving ones. With arrive_jit = 0
    they sit on the object centre (robot3's demos: n_arrive = 2); with arrive_jit > 0 the
    robot is placed uniformly in a disk of that radius around the target and the expert
    chunk homes straight to the centre and stays, which teaches stopping and recovering
    from passing beside the object."""
    objs, _ = make_scenes(n_ep, rng)
    pos, sid, pid, chunks = [], [], [], []
    for i in range(n_ep):
        p = rng.integers(len(prompts))
        tgt = rng.choice(np.where(matches(objs[i:i + 1], prompts[p])[0])[0])
        goal = objs[i, tgt, :2]
        p0 = rng.uniform(-1, 1, 2) * START_JIT
        path = straight_path(p0, goal, noise=noise, rng=rng)
        disp = np.vstack([np.diff(path, axis=0), np.zeros((H, 2))])
        n_move = int(np.ceil(np.linalg.norm(goal - p0) / STEP))
        for t in range(min(n_move, T_EP)):
            pos.append(path[t]); sid.append(i); pid.append(p)
            chunks.append(disp[t:t + H].ravel() / ACT_SCALE)
        for _ in range(n_arrive):
            q = goal.copy()
            if arrive_jit > 0:                                     # no draws otherwise: keeps old demos
                r, a = arrive_jit * np.sqrt(rng.uniform()), rng.uniform(0, 2 * np.pi)
                q += r * np.array([np.cos(a), np.sin(a)])
            home = straight_path(q, goal, n=H + 1)
            pos.append(q); sid.append(i); pid.append(p)
            chunks.append(np.diff(home, axis=0).ravel() / ACT_SCALE)
    return np.array(pos), np.array(sid), objs, np.array(pid), np.array(chunks)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(0)
    objs, R = make_scenes(4, rng)
    pos = rng.uniform(-1, 1, (4, 2)) * START_JIT
    im = render(pos, objs).permute(0, 2, 3, 1).numpy()
    fig, ax = plt.subplots(1, 4, figsize=(12, 3.2))
    for a, x, o, r in zip(ax, im, objs, R):
        a.imshow(x); a.set_title(f"R = {r:.2f}", fontsize=9); a.axis("off")
    os.makedirs("figs", exist_ok=True)
    fig.savefig("figs/render_examples.png", dpi=120, bbox_inches="tight")
    P = train_prompts()
    p_, s_, o_, pid_, ch_ = make_demos(200, rng, P)
    print("demos", p_.shape, ch_.shape, "| prompts:", [instruction(p) for p in P])
