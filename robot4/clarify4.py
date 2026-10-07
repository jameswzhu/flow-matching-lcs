"""
Clarifying questions from the flow: when a prompt is ambiguous *to the policy*, ask the
user to choose between alternative prompts that the policy executes unambiguously.

    python clarify4.py <policy> dist    # where the first chunk goes, per scene, under every
                                        # prompt: 2 x 64 fresh seeds ("model" / "execution")
    python clarify4.py <policy> ask     # detect ambiguity (label-free FTLE), pick questions,
                                        # simulate a user; out/<policy>/clarify.json
    python clarify4.py <policy> plot    # figs/clarify_<policy>.png

Scenes and FTLE are those of `eval4.py <policy> ftle` (50 scenes x 32 seeds, start state).

1. Detect. Score = fraction of the 32 seeds whose first-chunk FTLE is > 0 (no labels, no
   rollouts). Ground truth "the policy is multimodal here" = at least two objects get >= 10%
   of the execution seeds.
2. Ask. The robot may only say the 9 training prompts (the phrases it is grounded in). For
   a candidate answer a, p_a(k) is the share of model seeds whose chunk heads for kind k;
   p(k) is the same for the user's prompt. A question Q (2-4 options) is worth
   V(Q) = sum_k p(k) max_{a in Q} p_a(k): the chance the robot ends at the object the user
   meant, if the user meant one of the objects the robot is torn between and answers
   with the option that best gets it there. The robot asks the smallest Q with V within
   0.02 of the best 4-option question.
3. Simulated user. Means a kind k uniformly from the prompt's true match set, replies with
   the most specific option whose true meaning contains k (if none, repeats the prompt),
   and the robot executes the reply. Success = execution-seed share of chunks heading for
   k. The ask/no-ask threshold on the FTLE score is chosen on even scenes, tested on odd.
"""
import itertools, json, os, sys, time
import numpy as np
import torch
from env4 import ALL_PROMPTS, TRAIN_PROMPTS, KIND_NAMES, N_OBJ, H, chunk_target, kind_of
from pi0flow import load, D_ACT

NAME = sys.argv[1]
STEPS = sys.argv[2:]
OUT = os.path.join("out", NAME)
pol = load(f"policies/{NAME}.pt") if "dist" in STEPS else None
PROMPTS = list(ALL_PROMPTS)
CANDS = list(TRAIN_PROMPTS)
N_SEED = 64
ZERO = np.zeros((1, 2))
t00 = time.time()


def log(*a):
    print(f"[{time.time() - t00:6.0f}s]", *a, flush=True)


def ftle_scenes():
    D = np.load(os.path.join(OUT, "ftle.npz"))
    n_seed = int(np.sum(np.all(D["objs"] == D["objs"][0], axis=(1, 2))))
    objs = D["objs"][::n_seed]
    F = D["ftle"].reshape(len(D["prompts"]), len(objs), n_seed)
    return D["prompts"].tolist(), objs, F


# ----------------------------------------------------------------------------- dist
if "dist" in STEPS:
    ps, objs, _ = ftle_scenes()
    rng = np.random.default_rng(500)
    Zm = rng.standard_normal((len(objs), N_SEED, H * D_ACT))       # model seeds (choose the question)
    Ze = rng.standard_normal((len(objs), N_SEED, H * D_ACT))       # execution seeds (score it)
    P = np.zeros((2, len(objs), len(PROMPTS), N_OBJ))
    for i, o in enumerate(objs):
        for j, p in enumerate(PROMPTS):
            with torch.no_grad():
                c = pol.cond(ZERO, o[None], p)
            for r, Z in enumerate((Zm, Ze)):
                a = pol.flow_np(Z[i], c)
                k = kind_of(o[chunk_target(np.zeros((N_SEED, 2)), a, np.repeat(o[None], N_SEED, 0))])
                P[r, i, j] = np.bincount(k, minlength=N_OBJ) / N_SEED
        if i % 10 == 9:
            log(f"dist: {i + 1}/{len(objs)} scenes")
    np.savez_compressed(os.path.join(OUT, "clarify_dist.npz"), prompts=np.array(PROMPTS), model=P[0], exec=P[1],
                        objs=objs)


# ----------------------------------------------------------------------------- ask
def best_question(p, Pc, sizes=(2, 3, 4), tol=0.02):
    """p [4] robot's spread under the prompt; Pc [C, 4] spreads under the candidates.
    Returns (option indices, value)."""
    best = {}
    for m in sizes:
        best[m] = max(((Q, float(np.sum(p * Pc[list(Q)].max(0)))) for Q in itertools.combinations(range(len(Pc)), m)),
                      key=lambda x: x[1])
    vmax = max(v for _, v in best.values())
    for m in sizes:
        if best[m][1] >= vmax - tol:
            return best[m]


def user_reply(k, options):
    """Most specific option whose true meaning contains kind k (None if none does)."""
    ok = [a for a in options if k in ALL_PROMPTS[a]]
    return min(ok, key=lambda a: len(ALL_PROMPTS[a])) if ok else None


def phrase(options):
    o = [f"the {a}" if a not in ("red", "blue") else f"something {a}" for a in options]
    return "Do you mean " + (" or ".join(o) if len(o) == 2 else ", ".join(o[:-1]) + ", or " + o[-1]) + "?"


def auroc(score, y):
    from scipy.stats import rankdata
    y = np.asarray(y, bool)
    if y.all() or (~y).all():
        return np.nan
    r = rankdata(score)
    return (r[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum())


if "ask" in STEPS:
    ps_f, objs, F = ftle_scenes()
    D = np.load(os.path.join(OUT, "clarify_dist.npz"))
    Pm, Pe = D["model"], D["exec"]                                  # [scene, prompt, kind]
    n_sc = len(objs)
    ci = [PROMPTS.index(a) for a in CANDS]
    score = np.stack([(F[ps_f.index(p)] > 0).mean(1) for p in PROMPTS], 1)        # [scene, prompt]
    multimodal = (np.sort(Pe, -1)[..., -2] >= 0.1)                                   # [scene, prompt]
    # detection AUROC, bootstrap clustered by scene
    rng = np.random.default_rng(0)
    boots = [auroc(score[ii].ravel(), multimodal[ii].ravel()) for ii in
             (rng.integers(n_sc, size=n_sc) for _ in range(1000))]
    det = dict(auroc=auroc(score.ravel(), multimodal.ravel()), ci=list(np.nanquantile(boots, [.025, .975])),
               within_scene=float(np.nanmean([auroc(score[i], multimodal[i]) for i in range(n_sc)])),
               multimodal_rate=float(multimodal.mean()))
    # threshold on even scenes (balanced accuracy), applied to odd scenes
    ev, od = np.arange(0, n_sc, 2), np.arange(1, n_sc, 2)
    taus = np.unique(score)
    bacc = lambda t, ii: 0.5 * ((score[ii] >= t)[multimodal[ii]].mean() + (score[ii] < t)[~multimodal[ii]].mean())
    tau = float(taus[np.argmax([bacc(t, ev) for t in taus])])
    det.update(tau=tau, test_balanced_accuracy=float(bacc(tau, od)))
    log(f"detection: AUROC {det['auroc']:.3f} [{det['ci'][0]:.3f}, {det['ci'][1]:.3f}], "
        f"within-scene {det['within_scene']:.3f}; tau = {tau:.3f}, test balanced acc {det['test_balanced_accuracy']:.3f}")

    per_prompt, examples = {}, []
    for j, p in enumerate(PROMPTS):
        rows = []
        for i in od:                                                # test scenes only
            Q, V = best_question(Pm[i, j], Pm[i, ci])
            opts = [CANDS[q] for q in Q]
            truth = sorted(ALL_PROMPTS[p])
            s_none, s_ask = [], []
            for k in truth:                                          # user means k (uniform over true matches)
                s_none.append(Pe[i, j, k])
                r = user_reply(k, opts)
                s_ask.append(Pe[i, PROMPTS.index(r), k] if r else Pe[i, j, k])
            asked = score[i, j] >= tau
            rows.append(dict(none=np.mean(s_none), always=np.mean(s_ask), gated=np.mean(s_ask if asked else s_none),
                             asked=asked, value=V, n_opt=len(opts), opts=opts))
            if i == od[0]:
                examples.append(dict(prompt=p, robot_spread={KIND_NAMES[k]: round(float(Pm[i, j, k]), 2)
                                                             for k in range(N_OBJ) if Pm[i, j, k] > 0},
                                     ftle_score=float(score[i, j]), asks=bool(asked), question=phrase(opts),
                                     value=round(V, 3)))
        g = lambda key: float(np.mean([r[key] for r in rows]))
        per_prompt[p] = dict(n_match=len(ALL_PROMPTS[p]), trained=p in TRAIN_PROMPTS, success_no_ask=g("none"),
                             success_always_ask=g("always"), success_ftle_gated=g("gated"), ask_rate=g("asked"),
                             n_options_mean=g("n_opt"),
                             most_common_question=max(set(tuple(r["opts"]) for r in rows),
                                                      key=lambda q: sum(tuple(r["opts"]) == q for r in rows)))
        r_ = per_prompt[p]
        log(f"{p!r:18} success: no ask {r_['success_no_ask']:.2f} | ask if FTLE {r_['success_ftle_gated']:.2f} "
            f"(asks {r_['ask_rate']:.0%}) | always ask {r_['success_always_ask']:.2f} | "
            f"\"{phrase(list(r_['most_common_question']))}\"")
    json.dump(dict(detection=det, per_prompt=per_prompt, examples=examples),
              open(os.path.join(OUT, "clarify.json"), "w"), indent=1, default=float)
    for e in examples:
        print(f"  user: \"{e['prompt']}\"  ->  robot torn between {e['robot_spread']}, FTLE score "
              f"{e['ftle_score']:.2f}: " + (f"asks \"{e['question']}\"" if e["asks"] else "acts"))


# ----------------------------------------------------------------------------- plot
if "plot" in STEPS:
    import matplotlib.pyplot as plt
    C = json.load(open(os.path.join(OUT, "clarify.json")))
    SURF, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
    CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
    plt.rcParams.update({
        "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
        "text.color": INK, "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
        "axes.titlesize": 11, "axes.titleweight": "semibold", "legend.frameon": False})
    pp = C["per_prompt"]
    order = sorted(pp, key=lambda p: (pp[p]["n_match"], not pp[p]["trained"], p))
    x = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(11, 4.4), layout="constrained")
    ax.bar(x - 0.2, [pp[p]["success_no_ask"] for p in order], 0.4, color=CAT[1], label="act on the prompt")
    ax.bar(x + 0.2, [pp[p]["success_ftle_gated"] for p in order], 0.4, color=CAT[0],
           label="ask when FTLE flags ambiguity")
    for xi, p in zip(x, order):
        ax.text(xi + 0.2, pp[p]["success_ftle_gated"] + 0.02, f"{pp[p]['ask_rate']:.0%}", ha="center",
                fontsize=7.5, color=INK2)
    ax.set_xticks(x); ax.set_xticklabels([p + ("" if pp[p]["trained"] else " *") for p in order], rotation=50,
                                         ha="right", fontsize=8.5)
    ax.set_ylim(0, 1.1); ax.set_ylabel("P(robot heads for the object the user meant)")
    d = C["detection"]
    ax.set_title(f"clarifying questions (numbers = ask rate; * = unseen phrasing). Ambiguity detection from FTLE: "
                 f"AUROC {d['auroc']:.2f}", fontsize=10)
    ax.legend(loc="lower left", fontsize=8.5)
    fig.savefig(f"figs/clarify_{NAME}.png", dpi=150)
    log(f"figs/clarify_{NAME}.png")
