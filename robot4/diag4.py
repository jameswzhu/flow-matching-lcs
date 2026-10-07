"""
Open-loop diagnostics of a fine-tuned pi0 policy on demo states (why do rollouts stall?).

    python diag4.py <policy>     -> prints per-specificity numbers, figs/diag_<policy>.png

For held-out demo states (fresh scenes, expert paths), per prompt group:
  * target   : the predicted chunk's target (chunk_target) matches the prompt
  * expert   : the predicted chunk heads for the same object as the demo's expert
  * speed    : mean step length of the predicted chunk / expert speed (0.1)
  * err      : RMS distance between predicted and expert chunk positions (only where
               they head for the same object)
plus closed-loop trajectories for a few scenes and prompts.
"""
import sys
import numpy as np
import torch
from env4 import (H, STEP, ACT_SCALE, ALL_PROMPTS, KIND_NAMES, make_demos, chunk_target, chunk_positions,
                  matches, train_prompts, render, VIEW, kind_of)
from pi0flow import load, D_ACT

NAME = sys.argv[1]
pol = load(f"policies/{NAME}.pt")
rng = np.random.default_rng(7)
P = train_prompts()
pos, sid, objs, pid, ch = make_demos(300, rng, P)
pick = rng.choice(len(pos), 1200, replace=False)
pos, O, pid, ch = pos[pick], objs[sid[pick]], pid[pick], ch[pick]
pred = np.zeros_like(ch)
for k, p in enumerate(P):
    m = pid == k
    pred[m] = pol.sample(rng.standard_normal((m.sum(), H * D_ACT)), pos[m], O[m], p)
lab_p = chunk_target(pos, pred, O); lab_e = chunk_target(pos, ch, O)
ok = np.array([matches(O[i:i + 1], P[pid[i]])[0, lab_p[i]] for i in range(len(pos))])
step_p = np.linalg.norm(pred.reshape(-1, H, 2), axis=-1).mean(1) * ACT_SCALE
step_e = np.linalg.norm(ch.reshape(-1, H, 2), axis=-1).mean(1) * ACT_SCALE
moving = step_e > 0.05
same = lab_p == lab_e
rms = np.sqrt(((chunk_positions(pos, pred) - chunk_positions(pos, ch)) ** 2).sum(-1).mean(1))
for n in (1, 2, 4):
    g = np.array([len(ALL_PROMPTS[P[i]]) == n for i in pid])
    print(f"{n} match: target ok {ok[g].mean():.3f} | same as expert {same[g].mean():.3f} | "
          f"speed ratio (moving) {np.median(step_p[g & moving] / step_e[g & moving]):.3f} | "
          f"RMS err (same obj, moving) {np.median(rms[g & same & moving]):.3f}")

# closed-loop trajectories (every replan, executing EXEC steps; plotted for 3 scenes x 3 prompts x 6 seeds)
import matplotlib.pyplot as plt  # noqa: E402
from env4 import EXEC, T_EP, make_scenes, first_reached  # noqa: E402
objs3, _ = make_scenes(3, np.random.default_rng(11))
fig, ax = plt.subplots(3, 3, figsize=(10, 10), layout="constrained")
for r, o in enumerate(objs3):
    for c_, p in enumerate(["blue square", "square", "shape"]):
        n = 6; Ob = np.repeat(o[None], n, 0); q = np.zeros((n, 2)); tr = [q]
        for k in range(int(np.ceil(T_EP / EXEC)) + 1):
            chk = pol.sample(rng.standard_normal((n, H * D_ACT)), q, Ob, p)
            pts = chunk_positions(q, chk)[:, :EXEC]; tr.extend(np.transpose(pts, (1, 0, 2))); q = pts[:, -1]
        tr = np.stack(tr, 1)
        a = ax[r, c_]
        a.imshow(render(np.zeros((1, 2)), o[None])[0].permute(1, 2, 0).numpy(), extent=[-VIEW, VIEW, -VIEW, VIEW])
        for t_ in tr:
            a.plot(t_[:, 0], t_[:, 1], lw=1, color="#0b0b0b", alpha=0.7)
            a.scatter(t_[::EXEC, 0], t_[::EXEC, 1], s=4, color="#0b0b0b")
        hit = first_reached(tr, Ob)[0]
        a.set_title(f'"{p}": reached ' + ", ".join(KIND_NAMES[kind_of(o[h:h + 1])[0]] if h >= 0 else "none"
                                                   for h in hit), fontsize=7)
        a.set_xticks([]); a.set_yticks([])
fig.savefig(f"figs/diag_{NAME}.png", dpi=120)
print(f"figs/diag_{NAME}.png")
