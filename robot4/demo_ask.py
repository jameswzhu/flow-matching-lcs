"""
Worked example: the robot receives a prompt, checks its own flow for a ridge, finds the
behaviours it is torn between, has a VLM name them, checks the names with the flow, and
asks. Runs clarify_lf.py's pipeline live on one scene and prints every step.

    python demo_ask.py ["navy circle"] [scene]      -> figs/demo_ask_<prompt>.png (~3 min)

The scene is one of the test scenes of eval4's FTLE run (default 1), with the same
detection and model seeds as the evaluation, so the numbers match clarify_lf.py.
No object labels are used anywhere; the last line reports ground truth for reference.
"""
import json, os, sys, textwrap, time
import numpy as np
import torch
import clarify_lf as L
from env4 import ALL_PROMPTS, KIND_NAMES, H, VIEW, render, chunk_positions, instruction
from pi0flow import load, ftle, D_ACT

PROMPT = sys.argv[1] if len(sys.argv) > 1 else "navy circle"
SCENE = int(sys.argv[2]) if len(sys.argv) > 2 else 1
NAME = "pi0_arrive"
A = H * D_ACT
OUT = os.path.join("out", NAME)
t00 = time.time()


def say(*a):
    print(f"[{time.time() - t00:5.0f}s]", *a, flush=True)


# ----------------------------------------------------------------------------- 0. the request
pol = load(f"policies/{NAME}.pt")
ps_f, objs, _ = L.ftle_scenes()
o = objs[SCENE]
text = instruction(PROMPT) if PROMPT in ALL_PROMPTS else PROMPT
say(f'user: "{text}"   (scene {SCENE}; the camera image + text go through PaliGemma once)')
with torch.no_grad():
    c = pol.cond(L.ZERO, o[None], PROMPT)

# ----------------------------------------------------------------------------- 1. is there a ridge?
Zd = np.load(os.path.join(OUT, "ftle.npz"))["Z"].reshape(len(objs), -1, A)[SCENE]
_, J = pol.jac(Zd, c)
F = ftle(J)
tau = json.load(open(os.path.join(OUT, "clarify.json")))["detection"]["tau"]
ridge = F > 0
say(f"1. FTLE of {len(F)} noise seeds (24x384 flow-map Jacobians): max {F.max():+.2f}, "
    f"median {np.median(F):+.2f}; {ridge.sum()} seed(s) with FTLE > 0")
gate = ridge.mean() >= tau
say("   -> " + ("a ridge: the flow tears noise space apart somewhere, so the prompt has more than one "
                "outcome" if gate else "no ridge: one outcome; act on the prompt"))
if not gate:
    sys.exit(0)

# ----------------------------------------------------------------------------- 2. which outcomes?
Zm, _ = L.seeds(len(objs))
a, E, reps, trace = L.find_basins(pol, c, Zm[SCENE], Zd[ridge])
say(f"2. basins: {sum(r[4] == 'sampled' for r in reps)} from clustering where {len(Zm[SCENE])} sampled chunks end, "
    f"{sum(r[4] == 'ridge' for r in reps)} more found by crossing the ridge")
for n, tr in enumerate(trace):
    for cr in tr["crossings"]:
        say(f"   ridge seed {n} (FTLE {F[ridge][n]:+.2f}): chunk jumps at s = {cr['jump_at']:+.2f} along v1, settles at "
            f"s = {cr['settled_at']:+.2f}, ends at ({cr['end'][0]:+.2f}, {cr['end'][1]:+.2f})"
            + ("  -> NEW basin" if cr["kept"] else f"  (new, but only {cr['share']:.0%} of sampled seeds reach it: not offered)"
               if cr["new_basin"] else "  (basin already found)"))
for b, (cen, rep, share, _, src) in enumerate(reps):
    say(f"   basin {b}: {share:.0%} of sampled seeds, chunk ends at ({cen[0]:+.2f}, {cen[1]:+.2f})  [{src}]")
if len(reps) < 2:
    say("   only one basin found: act on the prompt"); sys.exit(0)

# FTLE slice through the ridge: plane spanned by the top two singular vectors at the strongest ridge seed
k = int(np.argmax(F)); _, _, Vt = np.linalg.svd(J[k], full_matrices=False)
g = np.linspace(-2.5, 2.5, 41); G1, G2 = np.meshgrid(g, g)
Zg = Zd[k] + G1.ravel()[:, None] * Vt[0] + G2.ravel()[:, None] * Vt[1]
ag, Jg = pol.jac(Zg, c)
cen_all = np.array([r[0] for r in reps])
Eg = L.endpoints(ag)
slice_basin = np.where(np.linalg.norm(Eg[:, None] - cen_all[None], axis=-1).min(1) < L.CLUSTER_D,
                       np.linalg.norm(Eg[:, None] - cen_all[None], axis=-1).argmin(1), -1).reshape(G1.shape)
slice_ftle = ftle(Jg).reshape(G1.shape)
say(f"   (FTLE slice through the strongest ridge seed: {len(Zg)} more Jacobians)")

# ----------------------------------------------------------------------------- 3. name them
vlm, proc = L.load_vlm()
descs = []
for b, (cen, rep, *_ ) in enumerate(reps):
    path = os.path.join("figs", f"demo_path_{b}.png")
    L.draw_path(o, rep, path)
    _, d = L.describe(vlm, proc, path)
    descs.append(d)
    say(f"3. basin {b}: VLM sees the path drawn on the camera image and says {d}")
del vlm; torch.cuda.empty_cache()

# ----------------------------------------------------------------------------- 4. check the names with the flow
options, checks = [], []
for b, (cen, *_ ) in enumerate(reps):
    chosen, rows = None, []
    for d in descs[b]:
        with torch.no_grad():
            cd = pol.cond(L.ZERO, o[None], L.instruction_from(d))
        ok, m, tight, dist = L.passes(L.endpoints(pol.flow_np(Zm[SCENE], cd)), cen)
        rows.append((d, ok, tight, dist))
        say(f'4. basin {b}: "{L.instruction_from(d)}" -> {tight:.0%} of seeds agree, lands {dist:.2f} from the basin: '
            + ("PASS" if ok else "fail"))
        if ok:
            chosen = (d, m); break
    checks.append(rows)
    if chosen and not any(np.linalg.norm(chosen[1] - m2) < L.MERGE_D for _, m2 in options if m2 is not None):
        options.append(chosen)
    elif chosen is None:
        options.append((None, None))

# ----------------------------------------------------------------------------- 5. ask
names = [d.strip().strip(".") if d else "[show the path]" for d, _ in options]
names = [n[0].lower() + n[1:] if n.lower().startswith(("the ", "a ")) else n for n in names]
question = "Do you mean " + (" or ".join(names) if len(names) == 2 else ", ".join(names[:-1]) + ", or " + names[-1]) + "?"
say(f'5. robot: "{question}"')
truth = [KIND_NAMES[int(L.gt_kind(r[1][None], o)[0])] for r in reps]
say(f"   (ground truth, for reference only: the basins head for {truth})")

# ----------------------------------------------------------------------------- figure
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
SURF, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#c4359b"]
BLUES = ["#eaf2fc", "#c4dbf5", "#8fbcec", "#5a9be0", "#2a78d6", "#1d5aa6", "#123d73"]
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "axes.spines.top": False, "axes.spines.right": False, "font.size": 9.5,
    "axes.titlesize": 10.5, "axes.titleweight": "semibold", "legend.frameon": False})
fig = plt.figure(figsize=(17, 8.6), layout="constrained")
gs = fig.add_gridspec(2, 4, width_ratios=[1, 1.15, 1.25, 1.15])
img = render(L.ZERO, o[None])[0].permute(1, 2, 0).numpy()
ext = [-VIEW, VIEW, -VIEW, VIEW]

ax = fig.add_subplot(gs[0, 0]); ax.imshow(img, extent=ext); ax.set_xticks([]); ax.set_yticks([])
ax.set_title(f'0. user: "{text}"')

ax = fig.add_subplot(gs[0, 1])
order = np.argsort(F)[::-1]
ax.bar(np.arange(len(F)), F[order], color=[CAT[1] if f > 0 else BLUES[3] for f in F[order]])
ax.axhline(0, color=INK2, lw=0.8)
ax.set_xlabel("noise seed (sorted)"); ax.set_ylabel("FTLE = log σmax(∂Φ/∂z)")
ax.set_title(f"1. is there a ridge?\n{ridge.sum()} of {len(F)} seeds have FTLE > 0 → yes")

ax = fig.add_subplot(gs[0, 2])
im = ax.imshow(slice_ftle, origin="lower", extent=[g[0], g[-1], g[0], g[-1]],
               cmap=LinearSegmentedColormap.from_list("b", BLUES))
B = slice_basin
edge = (np.diff(B, axis=0, prepend=B[:1]) != 0) | (np.diff(B, axis=1, prepend=B[:, :1]) != 0)
yy, xx = np.where(edge); ax.scatter(g[xx], g[yy], s=5, color=CAT[1], lw=0)
ax.axhline(0, color=INK, lw=1, ls="--")
ax.scatter([0], [0], color=INK, s=25, zorder=3)
for cr in trace[list(np.where(ridge)[0]).index(k)]["crossings"]:
    ax.scatter([cr["settled_at"]], [0], marker="x", color=INK if cr["kept"] or not cr["new_basin"] else MUTED, s=40, zorder=3)
ax.set_xlabel("z · v1 (most-stretched direction)"); ax.set_ylabel("z · v2")
ax.set_title("2. FTLE in noise space around the ridge seed (•)\norange = basin edge, ✕ = point past the jump")
fig.colorbar(im, ax=ax, shrink=0.75, label="FTLE", location="right", pad=0.02)

ax = fig.add_subplot(gs[0, 3])
tr = trace[list(np.where(ridge)[0]).index(k)]
ax.plot(0.5 * (tr["s"][1:] + tr["s"][:-1]), tr["jump"], color=CAT[0], lw=1.8)
for cr in tr["crossings"]:
    ax.axvline(cr["jump_at"], color=CAT[1], lw=1, ls=":"); ax.axvline(cr["settled_at"], color=INK, lw=1, ls="--")
ax.set_yscale("log"); ax.set_xlabel("s along v1 from the ridge seed")
ax.set_ylabel("chunk change per 0.05 step")
ax.set_title("2. line search: the chunk jumps at the ridge\n(dotted), settles in the next basin (dashed)")

ax = fig.add_subplot(gs[1, 0]); ax.imshow(img, extent=ext); ax.set_xticks([]); ax.set_yticks([])
for b, (cen, rep, share, _, src) in enumerate(reps):
    pts = np.vstack([[0, 0], chunk_positions(np.zeros((1, 2)), rep[None])[0]])
    ax.plot(pts[:, 0], pts[:, 1], color=CAT[[0, 2, 3, 4][b % 4]], lw=3)
    ax.annotate(f"basin {b}: {share:.0%}" + (" (ridge)" if src == "ridge" else ""), pts[-1], xytext=(4, 4),
                textcoords="offset points", fontsize=8.5, color=INK, weight="semibold")
ax.set_title("2. outcomes it is torn between\n(share of 64 sampled seeds)")

ax = fig.add_subplot(gs[1, 1:]); ax.axis("off")
y = 0.97
ax.text(0, y, "3. a VLM names each outcome from its path drawn on the image;   "
              "4. the policy checks each name (prompted with it, do ≥ 90% of seeds land in that basin?)",
        fontsize=10, weight="semibold", va="top", transform=ax.transAxes)
y -= 0.12
for b, rows in enumerate(checks):
    ax.text(0, y, f"basin {b}", fontsize=10, weight="semibold", color=CAT[[0, 2, 3, 4][b % 4]], va="top",
            transform=ax.transAxes)
    for d, ok, tight, dist in rows:
        ax.text(0.09, y, f'"{d}"', fontsize=9.5, va="top", transform=ax.transAxes)
        ax.text(0.62, y, f"{tight:.0%} agree, lands {dist:.2f} away  →  " + ("PASS" if ok else "fail"),
                fontsize=9.5, va="top", color=(CAT[2] if ok else CAT[1]), transform=ax.transAxes)
        y -= 0.075
    y -= 0.03
ax.text(0, max(y - 0.02, 0.06), "5. robot: " + "\n".join(textwrap.wrap(f'"{question}"', 90)), fontsize=13,
        weight="semibold", va="top", transform=ax.transAxes)
out = f"figs/demo_ask_{PROMPT.replace(' ', '_')}.png"
fig.savefig(out, dpi=140)
for b in range(len(reps)):
    os.remove(os.path.join("figs", f"demo_path_{b}.png"))
say(out)
