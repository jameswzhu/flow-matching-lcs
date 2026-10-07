# pi0 on the language-conditioned shapes task

robot3's task (four objects, red/blue × square/circle, on a ring; prompts of 1, 2 or 4
matches; paraphrases; a held-out combination), solved by **pi0** instead of a small
object-attention policy. The point is to run the FTLE / LCS analysis on a real VLA,
where words are bound to pixels by a pretrained VLM rather than by a hand-built
encoder.

## Task and observation

Scenes, prompts, match sets, the expert, the chunk label (`chunk_target`) and the
closed-loop labels are imported unchanged from `../robot3/env3.py`. What changes:

| | robot3 | robot4 (pi0) |
|---|---|---|
| objects | 16 features (rel. position, colour ±1, shape ±1) | **224×224 top-down camera image** (`env4.render`); objects drawn with half-size 0.2, robot as a dark disk of radius 0.11 |
| robot | implicit (relative positions) | image + state = position / 1.5, zero-padded to 32 |
| prompt | frozen CLIP / MiniLM / oracle vector | instruction ("reach the blue square", `env4.instruction`) tokenised and encoded by pi0's own PaliGemma |
| chunk | 12 × 2 | 12 × 2, padded to pi0's 32 action dims → noise space is **384-D** |
| sampler | 20 RK4 steps, noise at τ = 0 | pi0's own: **10 forward-Euler steps, noise at t = 1** |

## pi0 as a flow map (`pi0flow.py`)

pi0 = PaliGemma 3B (SigLIP + Gemma 2B) + a 300M Gemma action expert. Sampling has two
stages: (1) image + prompt tokens go through PaliGemma once, leaving a per-layer K/V
cache — **the only place image and language enter**; (2) the expert integrates
da/dt = v(a, t | cache, state) with 10 Euler steps. So Φ(z; c) with c = the prefix
cache, and FTLE = log σ_max(∂Φ_task/∂z), a 24 × 384 Jacobian.

- The expert step is written out with explicit attention against the cached K/V
  (lerobot's version goes through transformers' `DynamicCache`, which `torch.func`
  cannot trace). The state token cannot attend to action tokens (suffix mask
  `[1, 1, 0, …]`), so its per-layer K/V are computed once per flow and appended to the
  cache; each velocity call processes only the 12 action tokens.
- Jacobians: `vmap(jacrev)` (24 reverse passes), ~45 ms each at batch 32 on the 5090.
  Every seed at the same state shares one prefix (`Pi0Flow.jac(z, c)` with c of batch 1).
- **Checks** (`python pi0flow.py check [policies/pi0.pt]`):

  | | pi0_base | fine-tuned |
  |---|---|---|
  | explicit flow vs lerobot `sample_actions`, same noise (fp32) | 8e-7 | 1.5e-6 |
  | ∂Φ/∂z vs central FD, 12 random directions × 3 scenes (fp64 expert) | 2e-9 | 8e-10 |

  Reverse mode also matches forward mode (`torch.func.jvp`) to 2e-15.

### Precision: the VLM prefix must be float32

In bfloat16, the prefix K/V of a scene **depend on which other scenes share the batch**
(max relative change 6%; chunk moves by 0.01, Jacobian by 2%, measured on pi0_base). In
float32 the same change is 2e-5. All analysis therefore uses a float32 prefix (and a
float32 expert; float64 for derivative checks). This is also a finding in itself:
deployed pi0, which runs in bfloat16, has percent-level action noise that depends on
batch composition.

## Fine-tuning

Expert-only (VLM frozen; 300M trainable parameters: the Gemma expert and the
state/action/time projections), pi0's loss (x_t = t z + (1 − t) a, target z − a,
t ~ 0.001 + 0.999 Beta(1.5, 1), loss on the 2 task dims as in lerobot). 20k demo episodes
(372k states, the 9 training prompts), 5000 steps × 32 prefixes × 8 (noise, t) draws per
prefix (the frozen VLM forward dominates the cost, so each prefix is reused), AdamW
1e-4, warmup 300, cosine to 1e-5. The step runs under bfloat16 autocast (fp32 master
weights): 0.5 s/step, 42 min on the RTX 5090. Loss 0.69 → 0.118 (still slowly falling).

Freezing the VLM means the word → pixel binding is pi0's pretrained one: the paraphrase
and held-out tests probe PaliGemma's (pi0-finetuned) language understanding, not ours.

## LCS: distance from each seed to the mode boundary (`lcs4.py`, policy `pi0_arrive`)

Method as robot2: from each seed, line search along the top right singular vector v1 of
the 24 × 384 flow-map Jacobian (s ∈ [−4, 4], step 0.1), bisect the nearest label change on
each side (label = object the first chunk heads for). Margins d₊, d₋. Then random nudges
of size ε in 384-D, 8 per seed, and the flat-LCS prediction with no fitting. 30 scenes ×
16 seeds × 21 prompts (10,080 seeds; 80,640 nudges per ε). About 70 min on the 5090.

**The flow only listens to the task noise.** v1 has 99.7% of its energy (median) in the 24
noise dims that become the task's actions; the 360 padding dims of pi0's 32-wide action
space barely affect the mode. The effective noise space is 24-D, like robot2's.

| prompts | seeds with a boundary within 4 | median margin |
|---|---|---|
| 1 match, trained | 0–2% | – |
| 2 matches, trained | 66–75% | 0.64–0.85 |
| "shape"; "box", "round object", "(any) object" | 87–92% | 0.41–0.45 |
| other 1-match paraphrases | 49–80% | 0.67–0.84 |

| nudge ε | 0.5 | 1 | 2 | 4 | 8 |
|---|---|---|---|---|---|
| observed flip rate | 0.80% | 1.5% | 3.1% | 6.2% | 12.1% |
| LCS predicted (analytic, no fit) | 0.82% | 1.5% | 2.9% | 5.4% | 9.6% |
| per-nudge rule: precision / recall | 0.98 / 0.98 | 0.97 / 0.96 | 0.94 / 0.90 | 0.91 / 0.80 | 0.88 / 0.70 |
| per-nudge accuracy | 1.000 | 0.999 | 0.995 | 0.983 | 0.952 |
| AUROC, LCS margin [95% CI] | 0.992 [.991, .993] | 0.985 [.983, .987] | 0.968 [.965, .970] | 0.931 [.926, .936] | 0.870 [.864, .876] |
| AUROC, FTLE | 0.989 | 0.977 | 0.950 | 0.892 | 0.819 |
| destination = basin beyond the LCS | 92% | 94% | 96% | 96% | 95% |

(CIs clustered by scene; within-scene AUROCs agree to ±0.005.)

- **As in robot2, the LCS margin predicts mode switches quantitatively**: the flip rate is
  exact up to ε = 2 and the per-nudge rule is ≥ 99.5% accurate there. At ε ≥ 4 the
  sideways displacement (~ε) reaches the curvature of the boundary and the flat model
  under-predicts (recall 0.70 at ε = 8). The margin beats FTLE at every ε, with
  non-overlapping CIs from ε = 1.
- **The label-free ridge is the separatrix**: of 2,964 boundaries within 0.5, the
  largest chunk jump lies within 0.1 of the bisected label change in 95% (median gap
  0.026). The boundaries are hard: the chunk jumps by a median 3.2–4.3 across a 0.1
  window (robot2's "hard" threshold was 1).
- **Slivers.** For 2-match prompts, 20–30% of boundaries have a thin layer labelled with
  a third object: rotating from one square to the other, the chunk sweeps past the
  heading of an object lying between them. Destinations are therefore taken as the first
  label beyond the boundary that persists for 0.3; with that, 92–96% of flips land in
  the predicted basin (robot2: 100%). "shape"-like prompts have no slivers (3%): every
  object between two basins is itself a basin.

## Clarifying questions (`clarify4.py`, policy `pi0_arrive`)

Goal: when a prompt is ambiguous *to the policy*, the robot asks the user to choose
between alternative prompts that it executes unambiguously ("square" → "Do you mean the
red square or the blue square?").

1. **Detect ambiguity without labels or rollouts**: score = fraction of 32 seeds whose
   first-chunk FTLE at the start is > 0. Against "the policy sends ≥ 10% of 64 fresh
   seeds to each of ≥ 2 objects": **AUROC 0.94 [0.93, 0.95]** (within scene 0.94). The
   threshold (τ = 1/32: any seed with FTLE > 0) was chosen on even scenes; balanced
   accuracy on odd scenes 0.89.
2. **Choose the question with the policy itself.** The robot may only offer the 9 trained
   phrases. For each candidate a, the flow gives p_a(object) (share of 64 seeds); for the
   user's prompt, p(object). Question Q (2–4 options) maximises
   V(Q) = Σₖ p(k) max_{a∈Q} p_a(k), the chance of ending at the object meant if the user
   meant one of those the robot is torn between; smallest Q within 0.02 of the best.
3. **Simulated user**: means one object uniformly from the prompt's true match set,
   replies with the most specific option that names it, and the robot executes the reply
   (scored on 64 separate seeds; success = first chunk heads for the meant object).
   25 held-out scenes.

| prompts | act on the prompt | ask when FTLE flags ambiguity | ask rate |
|---|---|---|---|
| 1 match, trained | 1.00 | 1.00 | **0%** |
| 2 matches, trained | 0.49–0.50 | 0.94–0.98 | 88–96% |
| "shape" | 0.25 | 0.99 | 100% |
| "box", "round object", "(any) object" | 0.24–0.31 | 0.98–1.00 | 100% |
| ungrounded 1-match ("navy circle", "red box", "blue box") | 0.29–0.49 | 0.88–1.00 | 84–100% |
| partly grounded ("red disk", "blue disk", "crimson square") | 0.71–0.80 | 0.96–0.98 | 68–88% |

The questions are the ones a person would ask: "square" → "red square or blue square?",
"red" → "red square or red circle?", "navy circle" → "red circle or blue circle?",
"box" / "shape" → all four. Example (scene 1): for "navy circle" the robot sends 88% of
seeds to the blue circle and 12% to the red one; the FTLE score (2 of 32 seeds) triggers
the question, so even the 12% minority is caught.

- **The robot doesn't pester**: it never asked for a trained 1-match prompt in the test
  scenes; those prompts have essentially no ridge (1 of 6,400 seeds with FTLE > 0,
  "blue circle" in an even scene).
- **It fixes paraphrase fragility.** A paraphrase the VLM doesn't ground makes the flow
  multimodal, which FTLE detects; the question is answered in phrases the policy knows.
- Missed questions (ask rate < 100%) are scenes where one object takes nearly all seeds
  and none of the 32 detection seeds lands near the ridge; asking always reaches 1.00
  but asks for unambiguous prompts too.

**Caveats.** The user is simulated (picks the right option, understands the phrases).
Success is scored on the first chunk at the start; closed loop for trained 1-match
prompts is 1.000, so this should carry over, but it is not measured. The candidate
phrases are a fixed list; an LLM / VLM could propose candidates, which the flow would
then verify (this is the step that needs no labels: p_a only needs the policy's own
mode labels, which could be replaced by clustering the chunks). The cost of asking is
not modelled.

## Label-free clarifying questions (`clarify_lf.py`, policy `pi0_arrive`)

Same task as above, but the robot never sees an object label: it finds the behaviours it
is torn between from the flow, a VLM (Qwen3-VL-4B-Instruct) names them, and the policy
checks the names. Ground truth is used only to score. Same 25 test scenes and seeds.

1. **Detect**: the FTLE gate above (any of 32 seeds with FTLE > 0).
2. **Find the basins**:
   - Cluster the end points of 64 sampled first chunks (complete linkage at 0.4; clusters
     with < 10% of seeds dropped). Single linkage fails: seeds at the ridge give in-between
     headings that chain two basins together.
   - **Cross the ridge**: from each detection seed with FTLE > 0, line search along v1 for
     the largest chunk jump on each side, then walk past it until the chunk settles (< 0.05
     change per 0.05 step). A settled point whose end point is > 0.4 from every basin found
     so far is a new basin. This finds minority basins that few or no random seeds reach:
     86 new basins (4 duplicates) and 33 more questions asked. Stepping a fixed 0.3 past the
     jump was not enough (197 of 340 duplicates): just past the boundary the chunk is still
     turning.
3. **Name them**: each basin's representative path is drawn on the camera image (green
   arrow, `out/pi0_arrive/lf_paths/`). The VLM is asked for three short ways to refer to
   the object the robot is moving toward, without mentioning the arrow. It is not told
   the task's vocabulary.
4. **Verify with the policy**: description d becomes the prompt "reach the d". It passes
   if pi0 executes it tightly (≥ 90% of 64 seeds end within 0.4 of their mean) and lands
   within 0.4 of the basin. Options that reach the same place are merged. A basin with no
   passing description is offered as the drawn path ("[path]").
5. **Simulated user**: as before, but reads the options with a keyword reader (colour and
   shape words with synonyms).

| | 4B | 8B |
|---|---|---|
| first description names the right object | 89% | 71% |
| any of the three does | 91% | 81% |
| options that are verified phrases (rest shown as paths) | 90% | 92% |
| basins found = true number | 85% | 85% |
| mean success over the 17 prompts the policy finds ambiguous | **0.953** | 0.902 |
| (act on the prompt / ground-truth-label questions) | 0.451 / 0.971 | |

Per prompt (4B; `figs/clarify_lf_pi0_arrive.png`): "square" 0.94 (ground-truth version
0.94), "red" 0.96 (0.96), "shape" 0.98 (0.99), "navy circle" 0.84 (0.88), "box" 0.93
(1.00), "object" 0.93 (0.98). It never asks for a trained 1-match prompt. The questions
it asks are e.g. "the blue square or the red square?" for "square", and all four for
"shape".

- **The flow check catches the VLM's errors.** Of 152 basins with no passing description,
  89 had descriptions naming the wrong object (all rejected) and 63 had the right words
  but landed on a split basin before merging was added. The merging fixed most of those.
- **Bigger isn't better here.** Qwen3-VL-8B prefers spatial descriptions ("the red square
  on the left") that pi0 was not trained on, and more often names the wrong object. Part
  of its gap is the keyword-reading simulated user, which reads relational phrases ("the
  red circle near the blue square") as naming several objects.
- Remaining gap to the ground-truth version (0.953 vs 0.971): a few missed minority basins
  (10 cases), wrong descriptions for which the fallback path option is less pure, and
  smeared basins in scenes where two objects are close in angle.

**Caveats.** Simulated user; first-chunk scoring; one task where every behaviour is "go to
an object", so descriptions are noun phrases. The VLM call dominates the cost (~0.5 s
per path on the 5090); the flow parts take ~1–2 s per request.

## Results: `pi0_arrive` (20k steps, 6 jittered arrived states per episode)

Same evaluation as below. Training: `python pi0flow.py train arrive 20000` (2.8 h), loss
0.076 (vs 0.118 for `pi0`). Demos add 6 states per episode placed uniformly within 0.31
(contact distance) of the target, whose expert chunk homes to the centre and stays.
Checks: flow vs lerobot 5e-6, ∂Φ/∂z vs FD 4e-10.

| | success | wrong | none | switch | FTLE median | seeds with FTLE > 0 |
|---|---|---|---|---|---|---|
| 1 match (trained) | 1.000 | 0 | 0 | 0 | −1.83 … −1.81 | 0–0.1% |
| 2 matches (trained) | 0.995–1.000 | ≤ 0.5% | 0 | 1.5–2.8% | −1.57 … −1.39 | 10–11% |
| "shape" (4) | 1.000 | – | 0 | 4.0% | −0.99 | 23% |

Reach and contact criteria now agree exactly (the policy stops on the object). Per-scene
entropy / max for 2-match prompts is 0.72–0.80. Open loop at demo states: 100% matching
target, speed 1.00× expert, RMS error 0.035 (was 0.08–0.11).

- **Training prompts are solved, matching robot3.** The `pi0` gaps were undertraining
  and missing stop/recovery data, not pi0.
- **Slices (`figs/slice_pi0_arrive.png`) are now clean.** "blue square" is one basin
  (a 0.7% sliver to the red square in a corner, with its own thin ridge); "square" has
  exactly two basins (the two squares) split by one straight ridge; "shape" has four
  basins meeting at a single 4-way junction. The ridge lies on the label boundary
  everywhere. The `pi0` finding that "square" looks like "shape" is gone: it was
  undertraining.
- **Paraphrases: pi0 drops the words it doesn't ground, and the prompt becomes the
  remaining attribute.** Where the rollouts go (contact):

  | paraphrase | intended | behaves like | destinations |
  |---|---|---|---|
  | box, round object | 2 | "shape" | all four, 16–36% each |
  | red box | 1 | "red" | red square 38%, red circle 62% |
  | blue box | 1 | "blue" | blue square 49%, blue circle 50% |
  | navy circle | 1 | "circle" | red circle 58%, blue circle 42% |
  | red disk / blue disk | 1 | mostly the colour+circle | 81% / 72% correct |
  | crimson square | 1 | mostly "square" | red 79%, blue 20% |
  | a red thing | 2 | "red" | correct (1.00) |
  | something blue | 2 | mostly "blue" | 85% correct |

  So "box", "navy", "round object" are not grounded; "disk", "crimson" partly.
- **The FTLE tail at the start state predicts the closed-loop multimodality, with no
  rollouts.** Fraction of seeds with first-chunk FTLE > 0 vs the effective number of
  destinations exp(H) of the closed-loop kind shares, over all 21 prompts: Spearman
  ρ = 0.92 (p = 4e-9). Ungrounded 1-match paraphrases sit at the 2-match level
  (10–13%), "box" / "round object" at the "shape" level (20%), trained 1-match prompts at
  0. The FTLE fraction is the noise mass near a basin boundary, which grows with the
  number of boundaries; it reads off how ambiguous the policy *took* the prompt to be.

## Results: `pi0` (one 5k-step run, 2 arrived states on the centre)

Closed loop: 50 scenes × 8 seeds per prompt, whole episode, trajectories saved
(`out/pi0/rollout.npz`). Two arrival criteria: **reach** = robot3's (centre within 0.15)
and **contact** = robot disk touches the object (0.31). The policy rarely stops inside
0.15 (it passes beside or through the object; open-loop it aims correctly, see below), so
contact is the meaningful one here. FTLE: first chunk at the start, 50 scenes × 32 seeds.

| | contact success | wrong object | none | switch | FTLE median | FTLE p99 |
|---|---|---|---|---|---|---|
| 1 match (trained) | 0.79–0.95 | 1–10% | 4–11% | 5–9% | −0.84 … −0.48 | +0.82 … +1.48 |
| 2 matches (trained) | 0.58–0.70 | 12–24% | 17–21% | 17–20% | +0.60 … +0.74 | +2.71 … +3.00 |
| "shape" (4) | 0.77 | – | 24% | 23% | +0.89 | +3.40 |
| 1-match paraphrases | 0.26–0.67 | 18–51% | 16–25% | 13–23% | −0.24 … +0.79 | +2.16 … +2.96 |

robot3 (object features, trained from scratch): ~1.00 success everywhere on training
prompts, 0–0.2% switching for 1 match.

- **Open-loop the policy is accurate; errors build up in closed loop.** At expert demo
  states (`diag4.py`), the first chunk heads for a matching object 99% (1 match), 95%
  (2 matches), 100% ("shape") of the time, at 98% of the expert's speed, RMS position
  error < 0.11. Off the demo distribution it drifts, overshoots and switches.
- **The FTLE ordering of robot3 holds**: medians negative for 1 match, positive for
  ambiguous prompts, and the tail grows with the number of matches. Unlike robot3,
  single-match prompts keep a tail (12% of "blue square" seeds have FTLE > 0, p99 ≈ +1.4),
  matching their 1–10% wrong-object rate.
- **Slices (`figs/slice_pi0.png`): the ridges are exactly the basin boundaries**, as in
  every earlier system, with a 4-way junction near the origin for "shape".
- **pi0 hardly separates "square" from "shape".** In the slice, "square" has all four
  basins in nearly the same layout as "shape"; the two circles take 32% of the slice
  area (red circle 21%, blue circle 11%). This is the 2-match wrong-object rate (12–24%)
  seen from noise space: the attribute is applied as a soft re-weighting of the four
  basins rather than by removing two of them.
- **Paraphrases are much more fragile than with CLIP / MiniLM in robot3**: "blue box"
  0.28 (51% wrong), "red disk" 0.26, "box" 0.30, "navy circle" 0.36; best "a red thing"
  0.79, "crimson square" 0.67. Most single-match paraphrases get ambiguous-prompt FTLE
  (median ≈ 0 to +0.8). "red disk" fails badly with a low median (−0.04): possibly a
  confident collapse onto a wrong object (as "box" in robot3), not yet checked.

**Caveats.** One run of 5k steps with the loss still falling, so part of the gap to
robot3 is probably undertraining, not pi0. No confidence intervals yet. The held-out
("blue circle") run hasn't been trained.

## Next steps

1. Holdout run (`train arrive holdout 20000`): does "blue circle" compose zero-shot?
2. Prompt-space margins: perturb the prompt token embeddings (or interpolate between a
   paraphrase and its canonical prompt) and differentiate through the VLM prefix
   (`torch.func.jvp` through PaliGemma, float32). Keep the batch shape fixed in any finite
   difference (see Precision).
3. Clarification in closed loop and mid-course (ask when the ridge reaches the seed's
   neighbourhood during a rollout), with candidate phrases proposed by a language model.

## Setup

lerobot's pi0 needs Python ≥ 3.12 and transformers 5.x. From the project root:

    uv venv --python 3.12 .venv
    uv pip install --python .venv/bin/python torch torchvision --index-url https://download.pytorch.org/whl/cu128
    uv pip install --python .venv/bin/python "lerobot[pi] @ git+https://github.com/huggingface/lerobot.git" \
        sentencepiece scipy matplotlib sentence-transformers
    # PaliGemma tokenizer: the HF copy is gated; openpi uses this public one
    mkdir -p robot4/assets && curl -o robot4/assets/paligemma_tokenizer.model \
        https://storage.googleapis.com/big_vision/paligemma_tokenizer.model

`lerobot/pi0_base` (14 GB, fp32) downloads on first use; so does `Qwen/Qwen3-VL-4B-Instruct`
(9 GB) for `clarify_lf.py caption`. The VLM and pi0 are never loaded together. Tested with torch 2.11+cu128,
transformers 5.5.4, lerobot 0.6.2. Memory: the fp32 VLM + expert take ~13 GB; training
peaks at ~25 GB.

From `robot4/`:

    ../.venv/bin/python pi0flow.py check                 # on pi0_base
    ../.venv/bin/python pi0flow.py train [holdout]       # -> policies/pi0[_holdout].pt (~45 min)
    ../.venv/bin/python pi0flow.py check policies/pi0.pt
    ../.venv/bin/python eval4.py pi0 rollout ftle slice plot
    ../.venv/bin/python diag4.py pi0                     # open-loop accuracy + example trajectories

## Files

| File | Purpose |
|---|---|
| `env4.py` | robot3's task + renderer, instructions, demo states (rendered on the GPU at batch time) |
| `pi0flow.py` | pi0 as a differentiable flow map; fine-tuning; derivative checks |
| `eval4.py` | rollout / ftle / slice / plot, as `robot3/eval3.py` |
| `diag4.py` | open-loop diagnostics and example closed-loop trajectories |
| `lcs4.py` | LCS line search, bisection, nudge tests (`scan nudge plot`). `out/pi0_arrive/lcs_scan.npz` (19 MB) is not committed; `scan` rebuilds it (~50 min) |
| `clarify4.py` | ambiguity detection from FTLE, question selection, simulated user (`dist ask plot`) |
| `clarify_lf.py` | label-free version: ridge crossing + clustering, VLM descriptions, flow verification (`basins caption verify ask plot`; `VLM=` env picks the VLM) |
