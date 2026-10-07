"""
Evaluation of the fine-tuned pi0 policy on the shapes task (mirrors robot3/eval3.py).

    python eval4.py <policy> rollout   # closed loop, every prompt: success, mode shares,
                                       # mid-course switches                     (~35 min)
    python eval4.py <policy> ftle      # FTLE of the first chunk at the start, every prompt (~25 min)
    python eval4.py <policy> slice     # 2-D noise slices: labels + FTLE for 3 specificities (~15 min)
    python eval4.py <policy> plot      # figures from the cached outputs

<policy> is a name in policies/ (e.g. pi0, pi0_holdout). Outputs go to out/<policy>/ and
figs/. Everything runs with the float32 VLM prefix and float32 expert (see README for why
not bfloat16). Sizes are smaller than robot3's (50 scenes, not 100) because each pi0
prefix costs ~27 ms; every seed at the start state shares one prefix, so FTLE and slices
compute it once per scene and prompt.
"""
import json, os, sys, time
import numpy as np
import torch
from env4 import (ALL_PROMPTS, TRAIN_PROMPTS, TEST_PROMPTS, HOLDOUT, KIND_NAMES, EXEC, T_EP, N_OBJ, H,
                  OBJ_R, OBJ_HALF, ROBOT_R, make_scenes, chunk_target, chunk_positions, kind_of, matches, render)
from pi0flow import load, ftle, D_ACT

NAME = sys.argv[1]
STEPS = sys.argv[2:]
OUT = os.path.join("out", NAME)
os.makedirs(OUT, exist_ok=True); os.makedirs("figs", exist_ok=True)
NEED_MODEL = any(s in STEPS for s in ("rollout", "ftle", "slice"))
pol = load(f"policies/{NAME}.pt") if NEED_MODEL else None
HOLDOUT_RUN = NAME.endswith("_holdout")
PROMPTS = list(ALL_PROMPTS)
A = H * D_ACT                    # noise dimension (12 steps x 32 padded dims)
N_REPLAN = int(np.ceil(T_EP / EXEC)) + 1
t00 = time.time()


def log(*a):
    print(f"[{time.time() - t00:6.0f}s]", *a, flush=True)


def prompt_group(p):
    if p == HOLDOUT and HOLDOUT_RUN:
        return "held out"
    return "train" if p in TRAIN_PROMPTS else "paraphrase"


def rollout(objs, prompt, rng):
    """Closed loop from the origin for the whole episode (no early stop, so any arrival
    criterion can be applied afterwards). Returns traj [N, T+1, 2] and the chunk target
    label at every replan [N, N_REPLAN]."""
    N = len(objs)
    pos = np.zeros((N, 2)); traj = [pos]; labs = np.zeros((N, N_REPLAN), int)
    for k in range(N_REPLAN):
        ch = pol.sample(rng.standard_normal((N, A)), pos, objs, prompt)
        labs[:, k] = chunk_target(pos, ch, objs)
        pts = chunk_positions(pos, ch)[:, :EXEC]
        traj.extend(np.transpose(pts, (1, 0, 2)))
        pos = pts[:, -1]
    return np.stack(traj, 1), labs


def arrival(traj, objs, radius):
    """First slot whose disk of `radius` the trajectory enters (-1 if none), and the step."""
    dist = np.linalg.norm(traj[:, :, None, :] - objs[:, None, :, :2], axis=-1)
    hit = dist < radius; any_t = hit.any(2)
    t = np.where(any_t.any(1), np.argmax(any_t, 1), -1)
    return np.where(t >= 0, np.argmax(hit[np.arange(len(traj)), np.maximum(t, 0)], 1), -1), t


# robot3's criterion (centre within OBJ_R = 0.15), and contact in the image (robot disk
# touches the object: OBJ_HALF + ROBOT_R = 0.31). Objects are >= 0.61 from any straight
# path to another object, so contact cannot be triggered by a bystander.
CRITERIA = {"reach": OBJ_R, "contact": OBJ_HALF + ROBOT_R}


def metrics(traj, labs, O, p, radius, n_sc, n_seed):
    slot, t = arrival(traj, O, radius)
    ok_slot = matches(O, p)
    hit = slot >= 0
    success = hit & ok_slot[np.arange(len(O)), np.maximum(slot, 0)]
    kind = np.where(hit, kind_of(O[np.arange(len(O)), np.maximum(slot, 0)]), -1)
    m = sorted(ALL_PROMPTS[p])
    share = np.stack([(kind.reshape(n_sc, n_seed) == k).mean(1) for k in m], 1)
    ent = -(np.where(share > 0, share * np.log(np.maximum(share, 1e-12)), 0)).sum(1)
    # mid-course switching: chunk target changes between replans before arrival
    t_rep = np.where(t >= 0, t // EXEC + 1, N_REPLAN)                 # replans issued before arrival
    sw = np.array([np.any(np.diff(l[:n]) != 0) for l, n in zip(labs, t_rep)])
    return dict(success=float(success.mean()), wrong=float((hit & ~success).mean()),
                unreached=float((~hit).mean()),
                kind_share={KIND_NAMES[k]: float((kind == k).mean()) for k in range(N_OBJ)},
                scene_entropy_norm=(float(ent.mean() / np.log(len(m))) if len(m) > 1 else None),
                scene_min_share=(float(share.min(1).mean()) if len(m) > 1 else None),
                switch_rate=float(sw.mean()),
                final_ne_first=float((hit & (slot != labs[:, 0])).mean()))


# ----------------------------------------------------------------------------- rollout
if "rollout" in STEPS:
    n_sc, n_seed = 50, 8
    objs, R = make_scenes(n_sc, np.random.default_rng(100))
    O = np.repeat(objs, n_seed, 0)
    res, TR, LB = {}, [], []
    for p in PROMPTS:
        traj, labs = rollout(O, p, np.random.default_rng(101))
        TR.append(traj); LB.append(labs)
        res[p] = dict(group=prompt_group(p), n_match=len(ALL_PROMPTS[p]),
                      **{c: metrics(traj, labs, O, p, r, n_sc, n_seed) for c, r in CRITERIA.items()})
        log(f"{p!r:18} [{res[p]['group']:10}] " + " | ".join(
            f"{c}: success {res[p][c]['success']:.3f} wrong {res[p][c]['wrong']:.3f} none {res[p][c]['unreached']:.3f} "
            f"switch {res[p][c]['switch_rate']:.3f}" for c in CRITERIA))
        json.dump(res, open(os.path.join(OUT, "rollout.json"), "w"), indent=1)
    np.savez_compressed(os.path.join(OUT, "rollout.npz"), prompts=np.array(PROMPTS), traj=np.array(TR),
                        labels=np.array(LB), objs=O, R=np.repeat(R, n_seed))

# ----------------------------------------------------------------------------- ftle
if "ftle" in STEPS:
    n_sc, n_seed = 50, 32
    objs, R = make_scenes(n_sc, np.random.default_rng(200))
    O = np.repeat(objs, n_seed, 0); Rr = np.repeat(R, n_seed)
    Z = np.random.default_rng(201).standard_normal((len(O), A))
    zero = np.zeros((1, 2))
    F_all, L_all = [], []
    for p in PROMPTS:
        Fp, Lp = [], []
        for i in range(n_sc):
            with torch.no_grad():
                c = pol.cond(zero, objs[i:i + 1], p)
            sl = slice(i * n_seed, (i + 1) * n_seed)
            a, J = pol.jac(Z[sl], c)
            Fp.append(ftle(J)); Lp.append(chunk_target(np.zeros((n_seed, 2)), a, O[sl]))
        F_all.append(np.concatenate(Fp)); L_all.append(np.concatenate(Lp))
        log(f"ftle {p!r:18} median {np.median(F_all[-1]):+.2f}  p99 {np.quantile(F_all[-1], .99):+.2f}")
    np.savez_compressed(os.path.join(OUT, "ftle.npz"), prompts=np.array(PROMPTS), ftle=np.array(F_all),
                        label=np.array(L_all), R=Rr, objs=O, Z=Z)

# ----------------------------------------------------------------------------- slice
SLICE_PROMPTS = ["blue square", "square", "shape"]
if "slice" in STEPS:
    rng = np.random.default_rng(300)
    objs, _ = make_scenes(1, rng, R=1.2)
    zero = np.zeros((1, 2))
    # plane: top-2 principal directions of the per-mode mean seeds under "shape"
    Zs = rng.standard_normal((20000, A))
    with torch.no_grad():
        c = pol.cond(zero, objs, "shape")
    lab = chunk_target(np.zeros((len(Zs), 2)), pol.flow_np(Zs, c), np.repeat(objs, len(Zs), 0))
    means = np.stack([Zs[lab == k].mean(0) for k in range(N_OBJ) if (lab == k).sum() > 50])
    _, _, Vt = np.linalg.svd(means - means.mean(0))
    e1, e2 = Vt[0], Vt[1]
    g = np.linspace(-3, 3, 81)
    G1, G2 = np.meshgrid(g, g)
    Zg = G1.ravel()[:, None] * e1 + G2.ravel()[:, None] * e2
    O = np.repeat(objs, len(Zg), 0)
    out = dict(objs=objs[0], grid=g, e1=e1, e2=e2, prompts=np.array(SLICE_PROMPTS),
               shape_share=np.bincount(lab, minlength=N_OBJ) / len(lab))
    labs, fts = [], []
    for p in SLICE_PROMPTS:
        with torch.no_grad():
            c = pol.cond(zero, objs, p)
        a, J = pol.jac(Zg, c)
        labs.append(kind_of(objs[0][chunk_target(np.zeros((len(Zg), 2)), a, O)]).reshape(G1.shape))
        fts.append(ftle(J).reshape(G1.shape))
        log(f"slice {p!r}: kinds {np.unique(labs[-1])}, FTLE max {fts[-1].max():+.2f}")
    np.savez_compressed(os.path.join(OUT, "slice.npz"), labels=np.array(labs), ftle=np.array(fts), **out)

# ----------------------------------------------------------------------------- plot
if "plot" in STEPS:
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, LinearSegmentedColormap
    from env4 import VIEW
    SURF, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
    CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]            # by kind index
    BLUES = ["#eaf2fc", "#c4dbf5", "#8fbcec", "#5a9be0", "#2a78d6", "#1d5aa6", "#123d73"]
    plt.rcParams.update({
        "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
        "text.color": INK, "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
        "axes.titlesize": 11, "axes.titleweight": "semibold", "legend.frameon": False})

    # --- slices
    if os.path.exists(os.path.join(OUT, "slice.npz")):
        S = np.load(os.path.join(OUT, "slice.npz"))
        g = S["grid"]; ext = [g[0], g[-1], g[0], g[-1]]
        fig, ax = plt.subplots(2, 5, figsize=(13.5, 6.6), layout="constrained",
                               gridspec_kw=dict(width_ratios=[1, 1, 1, 1, 0.05]))
        ax[0, 4].axis("off")
        a0 = ax[0, 0]
        a0.imshow(render(np.zeros((1, 2)), S["objs"][None])[0].permute(1, 2, 0).numpy(),
                  extent=[-VIEW, VIEW, -VIEW, VIEW])
        for o in S["objs"]:
            k = kind_of(o[None])[0]
            a0.annotate(KIND_NAMES[k], (o[0], o[1]), xytext=(0, 11), textcoords="offset points",
                        ha="center", fontsize=8.5, color=CAT[k])
        a0.set_title("camera image (start)"); a0.set_xticks([]); a0.set_yticks([])
        ax[1, 0].axis("off")
        ax[1, 0].text(0, 0.95, f"policy: {NAME}\nplane: top-2 PCs of the\nper-mode mean seeds\nunder \"shape\"\n\n"
                      "top: chunk target\n(colour = kind, as labelled)\nbottom: FTLE = log σmax(∂Φ/∂z)\n"
                      "(dots = target boundaries)",
                      va="top", fontsize=9, color=INK2, transform=ax[1, 0].transAxes)
        vmax = float(S["ftle"].max()); vmin = float(np.quantile(S["ftle"], 0.02))
        for j, p in enumerate(S["prompts"]):
            L, Fm = S["labels"][j], S["ftle"][j]
            ax[0, j + 1].imshow(L, origin="lower", extent=ext, cmap=ListedColormap(CAT), vmin=-.5, vmax=3.5,
                                interpolation="nearest")
            ax[0, j + 1].set_title(f"\"{p}\"  ({len(ALL_PROMPTS[str(p)])} match)")
            im = ax[1, j + 1].imshow(Fm, origin="lower", extent=ext, cmap=LinearSegmentedColormap.from_list("b", BLUES),
                                     vmin=vmin, vmax=vmax)
            edge = (np.diff(L, axis=0, prepend=L[:1]) != 0) | (np.diff(L, axis=1, prepend=L[:, :1]) != 0)
            yy, xx = np.where(edge)
            ax[1, j + 1].scatter(g[xx], g[yy], s=1.5, color=CAT[1], lw=0)
            for a in (ax[0, j + 1], ax[1, j + 1]):
                a.set_xlabel("z · e1")
                if j == 0:
                    a.set_ylabel("z · e2")
                else:
                    a.set_yticklabels([])
        fig.colorbar(im, cax=ax[1, 4], label="FTLE")
        fig.savefig(f"figs/slice_{NAME}.png", dpi=150)
        log(f"figs/slice_{NAME}.png")

    # --- FTLE by prompt
    if os.path.exists(os.path.join(OUT, "ftle.npz")):
        D = np.load(os.path.join(OUT, "ftle.npz"))
        ps = D["prompts"].tolist(); Fv = D["ftle"]
        order = sorted(range(len(ps)), key=lambda i: (len(ALL_PROMPTS[ps[i]]), ps[i] in TEST_PROMPTS, ps[i]))
        fig, a = plt.subplots(figsize=(10, 4.2))
        for x, i in enumerate(order):
            v = Fv[i]; col = {1: CAT[0], 2: CAT[1], 4: CAT[2]}[len(ALL_PROMPTS[ps[i]])]
            q = np.quantile(v, [0.5, 0.9, 0.99])
            a.plot([x, x], [np.quantile(v, 0.05), q[2]], color=col, lw=2, alpha=0.5)
            a.scatter([x], [q[0]], color=col, s=40, zorder=3)
            a.scatter([x], [q[2]], color=col, s=40, marker="_", zorder=3)
        a.set_xticks(range(len(order)))
        a.set_xticklabels([ps[i] + (" *" if ps[i] in TEST_PROMPTS or (ps[i] == HOLDOUT and HOLDOUT_RUN) else "")
                           for i in order], rotation=50, ha="right", fontsize=8.5)
        a.set_ylabel("first-chunk FTLE at start")
        a.set_title("FTLE by prompt (dot = median, bar = 5th–99th pct; colour = # matches 1/2/4; * = unseen)")
        fig.savefig(f"figs/ftle_by_prompt_{NAME}.png", dpi=150, bbox_inches="tight")
        log(f"figs/ftle_by_prompt_{NAME}.png")
