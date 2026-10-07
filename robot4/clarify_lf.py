"""
Label-free clarifying questions: the robot finds the behaviours it is torn between from
the flow alone, describes each with a VLM, and keeps only descriptions that the policy
itself executes unambiguously. Ground-truth object labels are used only to score.

    python clarify_lf.py <policy> basins    # pi0: FTLE gate, cluster chunk end points of 64
                                            # seeds, draw each cluster's medoid path
    python clarify_lf.py <policy> caption   # VLM (Qwen3-VL-4B): 3 ways to refer to each path's target
    python clarify_lf.py <policy> verify    # pi0: run every description as a prompt
    python clarify_lf.py <policy> ask       # questions, simulated user, scores vs clarify4
    python clarify_lf.py <policy> plot      # figs/clarify_lf_<policy>.png

Same scenes, seeds and test split as clarify4.py (odd scenes of eval4's FTLE run; model
seeds Zm choose the question, execution seeds Ze score it).

1. Detect: FTLE gate of clarify4 (any of 32 seeds with FTLE > 0; tau from clarify.json).
2. Basins: end point of each of 64 first chunks (start + sum of displacements); complete-
   linkage clusters at 0.4 (basins are >= 0.67 apart and tight; single linkage chains them
   through in-between seeds at the ridge); clusters with < 10% of seeds are dropped. Then
   the ridge itself: from each detection seed with FTLE > 0, line search along v1 for the
   largest chunk jump on each side (> 0.5) and walk past it until the chunk settles; a
   settled point whose end point is > 0.4 from every basin found so far becomes a new
   basin, however few seeds land there.
   Ask only if the gate fires and >= 2 basins are found.
3. Describe: the medoid's path is drawn on the camera image (green arrow; the objects
   are red / blue) and the VLM is asked for three short ways to refer to the object the
   robot is moving toward, that a person who can't see the arrow would understand. It is
   not told the task's vocabulary.
4. Verify: description d becomes the prompt "reach the d". It passes for cluster j if pi0
   executes it tightly (>= 90% of the 64 model seeds end within 0.4 of their mean) and that
   mean is within 0.4 of cluster j's centre. The first passing description of each cluster
   is its option; options reaching the same place (< 0.2) are merged, so a basin that the
   clustering cut into pieces (a smear of in-between headings) becomes one option. A
   cluster with no passing description gets a "visual" option (show the path; the robot
   then draws noise from that basin).
5. Simulated user: means one object uniformly from the prompt's true match set; reads the
   options with a keyword reader (colour and shape words, with synonyms); picks the option
   naming their object, or the visual option showing it, else repeats the prompt.
"""
import json, os, re, sys, time
import numpy as np
import torch
from env4 import (ALL_PROMPTS, TRAIN_PROMPTS, KIND_NAMES, KINDS, N_OBJ, H, VIEW, render, chunk_positions,
                  chunk_target, kind_of)
from pi0flow import D_ACT

NAME = sys.argv[1]
STEPS = sys.argv[2:]
OUT = os.path.join("out", NAME)
IMG_DIR = os.path.join(OUT, "lf_paths")
os.makedirs(IMG_DIR, exist_ok=True)
PROMPTS = list(ALL_PROMPTS)
N_SEED = 64
CLUSTER_D, MIN_SHARE, PASS_SHARE, CACHE_D, MERGE_D = 0.4, 0.10, 0.90, 0.15, 0.2
RIDGE_S, RIDGE_STEP, RIDGE_JUMP, RIDGE_SETTLE = 2.0, 0.05, 0.5, 0.05
VLM = os.environ.get("VLM", "Qwen/Qwen3-VL-4B-Instruct")       # VLM=Qwen/Qwen3-VL-8B-Instruct for the larger one
TAG = "" if VLM.endswith("4B-Instruct") else "_" + VLM.split("-")[-2].lower()                 # "_8b"
CAPS, VER = f"lf_captions{TAG}.json", f"lf_verify{TAG}.json"
ZERO = np.zeros((1, 2))
t00 = time.time()


def log(*a):
    print(f"[{time.time() - t00:6.0f}s]", *a, flush=True)


def ftle_scenes():
    D = np.load(os.path.join(OUT, "ftle.npz"))
    n_seed = int(np.sum(np.all(D["objs"] == D["objs"][0], axis=(1, 2))))
    objs = D["objs"][::n_seed]
    return D["prompts"].tolist(), objs, D["ftle"].reshape(len(D["prompts"]), len(objs), n_seed)


def seeds(n_sc):
    rng = np.random.default_rng(500)                               # as clarify4.py
    return rng.standard_normal((n_sc, N_SEED, H * D_ACT)), rng.standard_normal((n_sc, N_SEED, H * D_ACT))


def endpoints(a):
    return chunk_positions(np.zeros((len(a), 2)), a)[:, -1]


def gt_kind(a, o):
    """Ground truth (scoring only): kind of the object each chunk heads for."""
    return kind_of(o[chunk_target(np.zeros((len(a), 2)), a, np.repeat(o[None], len(a), 0))])


def cluster(E):
    from scipy.cluster.hierarchy import fcluster, linkage
    lab = fcluster(linkage(E, "complete"), CLUSTER_D, "distance") - 1   # complete: no chaining via ridge seeds
    keep = [c for c in np.unique(lab) if np.mean(lab == c) >= MIN_SHARE]
    keep.sort(key=lambda c: -np.mean(lab == c))
    return lab, keep


def instruction_from(desc):
    d = desc.strip().strip(".").lower()
    d = re.sub(r"^(the|a|an)\s+", "", d)
    return f"reach the {d}"


# ----------------------------------------------------------------------------- basins
def draw_path(o, chunk, path):
    """Camera image (2x upscaled) with the chunk's path as a green arrow, extended along
    its heading to the image edge (dashed). Saved as png."""
    from PIL import Image, ImageDraw
    img = render(ZERO, o[None])[0].permute(1, 2, 0).numpy()
    im = Image.fromarray((np.kron(img, np.ones((2, 2, 1))) * 255).astype(np.uint8))
    S = im.size[0]
    px = lambda p: ((p[0] + VIEW) / (2 * VIEW) * S, (VIEW - p[1]) / (2 * VIEW) * S)
    pts = np.vstack([[0, 0], chunk_positions(np.zeros((1, 2)), chunk[None])[0]])
    d = ImageDraw.Draw(im)
    G = (20, 160, 60)
    head = pts[-1] - pts[0]; head /= max(np.linalg.norm(head), 1e-9)
    for k in range(0, 40, 2):                                      # dashed extension
        a, b = pts[-1] + head * (0.25 + 0.1 * k), pts[-1] + head * (0.25 + 0.1 * (k + 1))
        if np.abs(b).max() > VIEW:
            break
        d.line([px(a), px(b)], fill=G, width=2)
    d.line([px(p) for p in pts], fill=G, width=5, joint="curve")
    tip = pts[-1] + head * 0.22; side = np.array([-head[1], head[0]])
    d.polygon([px(tip), px(pts[-1] + side * 0.1), px(pts[-1] - side * 0.1)], fill=G)
    im.save(path)


if "basins" in STEPS:
    from pi0flow import load
    pol = load(f"policies/{NAME}.pt")
    ps_f, objs, F = ftle_scenes()
    tau = json.load(open(os.path.join(OUT, "clarify.json")))["detection"]["tau"]
    Zm, Ze = seeds(len(objs))
    Zd = np.load(os.path.join(OUT, "ftle.npz"))["Z"].reshape(len(objs), -1, H * D_ACT)   # the FTLE detection seeds
    test = list(range(1, len(objs), 2))
    cases, jobs = [], []                                           # jobs: paths to describe (deduped per scene)
    for i in test:
        o = objs[i]; scene_jobs = []
        for p in PROMPTS:
            with torch.no_grad():
                c = pol.cond(ZERO, o[None], p)
            a = pol.flow_np(Zm[i], c); E = endpoints(a)
            lab, keep = cluster(E)
            gate = float((F[ps_f.index(p), i] > 0).mean()) >= tau
            reps = []                                               # (centroid, chunk, share, member chunks, source)
            for cid in keep:
                mem = np.where(lab == cid)[0]
                cen = E[mem].mean(0)
                med = mem[np.argmin(np.linalg.norm(E[mem] - cen, axis=1))]
                reps.append((cen, a[med], len(mem) / N_SEED, a[mem], "sampled"))
            if gate:
                # basins of the ridge itself: from each detection seed with FTLE > 0, line search along
                # v1 for the largest chunk jump on each side and step RIDGE_BEYOND past it
                zr = Zd[i][F[ps_f.index(p), i] > 0]
                _, J = pol.jac(zr, c)
                v1 = np.linalg.svd(J, full_matrices=False)[2][:, 0]
                sg = np.arange(-RIDGE_S, RIDGE_S + 1e-9, RIDGE_STEP); i0 = len(sg) // 2
                aa = pol.flow_np((zr[:, None] + sg[None, :, None] * v1[:, None]).reshape(-1, zr.shape[1]), c)
                aa = aa.reshape(len(zr), len(sg), -1)
                jump = np.linalg.norm(np.diff(aa, axis=1), axis=-1)
                for n in range(len(zr)):
                    for side in (+1, -1):
                        rng_ = np.arange(i0, len(sg) - 1) if side > 0 else np.arange(0, i0)
                        k = rng_[np.argmax(jump[n, rng_])]
                        if jump[n, k] < RIDGE_JUMP:
                            continue
                        # walk past the jump until the chunk settles (< RIDGE_SETTLE per step for 3 steps):
                        # just past the boundary the chunk is still turning toward its basin
                        j = k + 1 if side > 0 else k
                        while 0 < j < len(sg) - 1:
                            nxt = jump[n, j:j + 3] if side > 0 else jump[n, max(j - 3, 0):j]
                            if len(nxt) == 3 and np.all(nxt < RIDGE_SETTLE):
                                break
                            j += side
                        ch = aa[n, j]; e = endpoints(ch[None])[0]
                        if min([np.linalg.norm(e - r[0]) for r in reps] + [np.inf]) > CLUSTER_D:
                            near = np.linalg.norm(E - e, axis=1) < CLUSTER_D
                            reps.append((e, ch, near.mean(), ch[None], "ridge"))
            cl = []
            for cen, rep, share, members, src in reps:
                # describe: reuse a path already drawn in this scene if its end point is within CACHE_D
                jid = next((j for j in scene_jobs if np.linalg.norm(jobs[j]["centroid"] - cen) < CACHE_D), None)
                if jid is None:
                    jid = len(jobs); scene_jobs.append(jid)
                    path = os.path.join(IMG_DIR, f"s{i:02d}_j{jid:04d}.png")
                    draw_path(o, rep, path)
                    jobs.append(dict(scene=i, centroid=cen, image=path, gt_kind=int(gt_kind(rep[None], o)[0])))
                cl.append(dict(centroid=cen.tolist(), share=float(share), job=jid, source=src,
                               gt_kinds=np.bincount(gt_kind(members, o), minlength=N_OBJ).tolist()))
            cases.append(dict(scene=i, prompt=p, gate=gate, clusters=cl,
                              gt_model=np.bincount(gt_kind(a, o), minlength=N_OBJ).tolist()))
        log(f"basins: scene {i}, {len(scene_jobs)} paths to describe")
    json.dump(dict(cases=cases, jobs=[{**j, "centroid": j["centroid"].tolist()} for j in jobs]),
              open(os.path.join(OUT, "lf_basins.json"), "w"), indent=1)


# ----------------------------------------------------------------------------- caption
ASK = ("This is a top-down camera view of a table. The small black dot is a robot. The green arrow shows "
       "the path the robot is about to take; the dashed line continues its direction. First decide which "
       "single object the robot is moving toward. Then give three different short ways (2-5 words each) to "
       "refer to that same object, so that a person looking at the table (who cannot see the arrow) would "
       "know which object you mean. Do not mention the arrow, the path or the robot. Answer only with a "
       "numbered list.")

if "caption" in STEPS:
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor
    B = json.load(open(os.path.join(OUT, "lf_basins.json")))
    proc = AutoProcessor.from_pretrained(VLM)
    vlm = AutoModelForImageTextToText.from_pretrained(VLM, dtype=torch.bfloat16).to("cuda").eval()
    caps = []
    for n, j in enumerate(B["jobs"]):
        msg = [{"role": "user", "content": [{"type": "image", "image": Image.open(j["image"]).convert("RGB")},
                                            {"type": "text", "text": ASK}]}]
        inp = proc.apply_chat_template(msg, tokenize=True, add_generation_prompt=True, return_dict=True,
                                       return_tensors="pt").to(vlm.device)
        with torch.no_grad():
            out = vlm.generate(**inp, max_new_tokens=80, do_sample=False)
        text = proc.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        descs = [re.sub(r"^\s*\d+[.)]\s*", "", l).strip() for l in text.splitlines() if re.match(r"^\s*\d+[.)]", l)]
        caps.append(dict(raw=text, descs=descs[:3]))
        if n < 8 or n % 25 == 0:
            log(f"caption {n + 1}/{len(B['jobs'])} (truth: {KIND_NAMES[j['gt_kind']]}): {descs[:3]}")
    json.dump(caps, open(os.path.join(OUT, CAPS), "w"), indent=1)


# ----------------------------------------------------------------------------- verify
if "verify" in STEPS:
    from pi0flow import load
    pol = load(f"policies/{NAME}.pt")
    B = json.load(open(os.path.join(OUT, "lf_basins.json")))
    caps = json.load(open(os.path.join(OUT, CAPS)))
    _, objs, _ = ftle_scenes()
    Zm, Ze = seeds(len(objs))
    texts = {}                                                     # (scene, instruction) -> endpoints / gt kinds
    for j, cp in zip(B["jobs"], caps):
        i = j["scene"]; o = objs[i]
        for d in cp["descs"]:
            key = f"{i}|{instruction_from(d)}"
            if key in texts:
                continue
            with torch.no_grad():
                c = pol.cond(ZERO, o[None], instruction_from(d))
            am, ae = pol.flow_np(Zm[i], c), pol.flow_np(Ze[i], c)
            texts[key] = dict(E_model=endpoints(am).tolist(), gt_exec=gt_kind(ae, o).tolist())
    log(f"verify: {len(texts)} distinct (scene, description) prompts")
    json.dump(texts, open(os.path.join(OUT, VER), "w"))


# ----------------------------------------------------------------------------- ask
COLOUR = {"red": ["red", "crimson", "scarlet", "maroon"], "blue": ["blue", "navy", "azure", "cobalt"]}
SHAPE = {"square": ["square", "box", "cube", "rectangle", "rectangular", "block"],
         "circle": ["circle", "circular", "round", "disk", "disc", "dot", "ball", "sphere"]}


def reads_as(desc):
    """Keyword reader (the simulated user, and the truth check): set of kinds the text names."""
    w = set(re.findall(r"[a-z]+", desc.lower()))
    col = [c for c, syn in COLOUR.items() if w & set(syn)]
    shp = [s for s, syn in SHAPE.items() if w & set(syn)]
    return {k for k, (c, s) in enumerate(KINDS) if (not col or c in col) and (not shp or s in shp)} if (col or shp) else set()


if "ask" in STEPS:
    B = json.load(open(os.path.join(OUT, "lf_basins.json")))
    caps = json.load(open(os.path.join(OUT, CAPS)))
    V = json.load(open(os.path.join(OUT, VER)))
    G = np.load(os.path.join(OUT, "clarify_dist.npz"))             # ground-truth version's execution shares
    Pe = G["exec"]
    jobs = B["jobs"]
    # caption quality (truth check with the keyword reader): first description names exactly the true kind
    cap_true = [reads_as(c["descs"][0]) == {j["gt_kind"]} if c["descs"] else False for j, c in zip(jobs, caps)]
    any_true = [any(reads_as(d) == {j["gt_kind"]} for d in c["descs"]) for j, c in zip(jobs, caps)]
    per_prompt, examples, n_clusters_ok, opt_stats = {}, [], [], dict(verbal=0, visual=0)
    for case in B["cases"]:
        i, p, cl = case["scene"], case["prompt"], case["clusters"]
        cen = np.array([c["centroid"] for c in cl])
        # options: first verified description per cluster. A description passes if pi0 executes it
        # tightly (>= 90% of end points within CLUSTER_D of their mean) and lands within CLUSTER_D
        # of the cluster's centre; options reaching the same place (< MERGE_D) are merged, so a
        # basin split into several clusters becomes one option
        opts, opt_end, opt_cl = [], [], []
        for j, c in enumerate(cl):
            choice = None
            for d in caps[c["job"]]["descs"]:
                E = np.array(V[f"{i}|{instruction_from(d)}"]["E_model"]); m = E.mean(0)
                tight = np.mean(np.linalg.norm(E - m, axis=1) < CLUSTER_D) >= PASS_SHARE
                if tight and np.linalg.norm(m - cen[j]) < CLUSTER_D:
                    choice = (d, m); break
            if choice is None:
                opts.append(None); opt_end.append(None); opt_cl.append(j)
            elif not any(e is not None and np.linalg.norm(e - choice[1]) < MERGE_D for e in opt_end):
                opts.append(choice[0]); opt_end.append(choice[1]); opt_cl.append(j)
        n_basins_lf = len(opts)
        asked = case["gate"] and n_basins_lf >= 2
        # ground-truth bookkeeping (scoring only): clustering vs truth
        gt_major = {k for k, s in enumerate(case["gt_model"]) if s / N_SEED >= MIN_SHARE}
        n_clusters_ok.append(n_basins_lf == len(gt_major))
        truth = sorted(ALL_PROMPTS[p]); s_none, s_ask = [], []
        for k in truth:
            s_none.append(Pe[i, PROMPTS.index(p), k])
            if not asked:
                s_ask.append(s_none[-1]); continue
            pick = next((d for d in opts if d and reads_as(d) == {k}), None)        # verbal option naming k
            if pick is not None:
                s_ask.append(np.mean(np.array(V[f"{i}|{instruction_from(pick)}"]["gt_exec"]) == k)); continue
            vis = [j for j, d in zip(opt_cl, opts) if d is None and np.argmax(cl[j]["gt_kinds"]) == k]
            if vis:                                                 # user recognises the drawn path
                s_ask.append(cl[vis[0]]["gt_kinds"][k] / sum(cl[vis[0]]["gt_kinds"])); continue
            s_ask.append(s_none[-1])                                # nothing fits: repeats the prompt
        if asked:
            opt_stats["verbal"] += sum(d is not None for d in opts); opt_stats["visual"] += sum(d is None for d in opts)
        r = per_prompt.setdefault(p, dict(none=[], ask=[], asked=[], q=[]))
        r["none"].append(np.mean(s_none)); r["ask"].append(np.mean(s_ask)); r["asked"].append(asked)
        if asked:
            r["q"].append([d if d else "[path]" for d in opts])
        if i == 1 and asked:
            examples.append(dict(prompt=p, clusters=[f"{c['share']:.0%}" for c in cl],
                                 descriptions=[caps[c["job"]]["descs"] for c in cl], options=opts))
    summ = dict(
        captions=dict(n=len(jobs), first_names_true_object=float(np.mean(cap_true)),
                      any_names_true_object=float(np.mean(any_true))),
        clustering=dict(cluster_count_matches_truth=float(np.mean(n_clusters_ok))),
        options=dict(**opt_stats, verbal_share=opt_stats["verbal"] / max(sum(opt_stats.values()), 1)),
        per_prompt={p: dict(success_no_ask=float(np.mean(r["none"])), success_label_free=float(np.mean(r["ask"])),
                            ask_rate=float(np.mean(r["asked"])),
                            example_question=(r["q"][0] if r["q"] else None)) for p, r in per_prompt.items()},
        examples=examples)
    gt = json.load(open(os.path.join(OUT, "clarify.json")))["per_prompt"]
    for p in PROMPTS:
        r = summ["per_prompt"][p]; r["success_ground_truth_labels"] = gt[p]["success_ftle_gated"]
        log(f"{p!r:18} no ask {r['success_no_ask']:.2f} | label-free {r['success_label_free']:.2f} "
            f"(asks {r['ask_rate']:.0%}) | GT labels {r['success_ground_truth_labels']:.2f} | e.g. {r['example_question']}")
    log(json.dumps({k: summ[k] for k in ("captions", "clustering", "options")}))
    for e in examples:
        print("  ", json.dumps(e))
    json.dump(summ, open(os.path.join(OUT, f"clarify_lf{TAG}.json"), "w"), indent=1)


# ----------------------------------------------------------------------------- plot
if "plot" in STEPS:
    import matplotlib.pyplot as plt
    C = json.load(open(os.path.join(OUT, f"clarify_lf{TAG}.json")))
    SURF, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
    CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
    plt.rcParams.update({
        "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
        "text.color": INK, "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
        "axes.titlesize": 11, "axes.titleweight": "semibold", "legend.frameon": False})
    pp = C["per_prompt"]
    order = sorted(pp, key=lambda p: (len(ALL_PROMPTS[p]), p not in TRAIN_PROMPTS, p))
    x = np.arange(len(order)); w = 0.28
    fig, ax = plt.subplots(figsize=(12, 4.4), layout="constrained")
    for k, (key, col, lab) in enumerate([("success_no_ask", CAT[1], "act on the prompt"),
                                         ("success_label_free", CAT[0], "ask: label-free (flow + VLM)"),
                                         ("success_ground_truth_labels", CAT[2], "ask: ground-truth labels")]):
        ax.bar(x + (k - 1) * w, [pp[p][key] for p in order], w, color=col, label=lab)
    for xi, p in zip(x, order):
        ax.text(xi, max(pp[p]["success_label_free"], pp[p]["success_ground_truth_labels"]) + 0.02,
                f"{pp[p]['ask_rate']:.0%}", ha="center", fontsize=7.5, color=INK2)
    ax.set_xticks(x); ax.set_xticklabels([p + ("" if p in TRAIN_PROMPTS else " *") for p in order], rotation=50,
                                         ha="right", fontsize=8.5)
    ax.set_ylim(0, 1.12); ax.set_ylabel("P(robot heads for the object the user meant)")
    ax.set_title("label-free clarifying questions (numbers = label-free ask rate; * = unseen phrasing)")
    ax.legend(loc="lower left", fontsize=8.5)
    fig.savefig(f"figs/clarify_lf_{NAME}{TAG}.png", dpi=150)
    log(f"figs/clarify_lf_{NAME}{TAG}.png")
