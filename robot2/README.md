# Multi-obstacle validation: how well does FTLE predict mode changes?

## Task

A point robot travels from x = −2.5 to a goal at (2.5, g) past two columns of
vertically stacked disks:

- **Column A** (x = −1): 2 stacked disks, giving **3 gaps** (top / mid / bottom).
- **Column B** (x = +1): 3 stacked disks; the expert uses the **2 inner gaps** (upper / lower).

That gives **6 routes, chosen by two decisions made at different times**. The goal
height g is the conditioning. The expert prefers A gaps near 1.6·g, and B gaps near
0.6·g + 0.35·(chosen A height), so the second choice depends on the first.

The policy is a flow-matching model over **12-step action chunks (24-D)**,
conditioned on (x, y, g). It is an MLP with 4×256 SiLU layers, trained for 50k steps on
6,000 demos. It executes 4 actions and then replans, and samples with 20 RK4 steps
throughout. Flow-map Jacobians with respect to noise and goal match finite
differences to 4e-11.

**Labels.** "Which gap is this chunk heading for?" is decided by the nearest
expert-like prototype chunk. This matches the true gap on 99.7–100% of expert chunks.
Closed-loop routes are read from where each trajectory crosses each column.

## Files

| File | Purpose |
|---|---|
| `env2.py` | Environment, expert, demos, route and chunk labels |
| `policy2.py` | Training, sampling, Jacobians; also records Eulerian diagnostics |
| `eval2.py` | `python eval2.py [S0 S1 S2 S3]` (~55 min total on 2 CPU cores) |
| `plot_s2.py`, `plot_s3.py` | Redraw the S2/S3 figures from cached results |
| `results2.json`, `s3_rows.npz`, `policy2.npz`, `figs/` | Outputs |

Reuses `../robot/mlp.py`.

## Results

**S0, imitation (`s0_imitation.png`).** No collisions and no unfinished episodes.
The policy captures the structure of both decisions, but it **underweights the
middle gap of column A** (about 0.5 vs the expert's 0.65 near g = 0). It also makes the
B choice almost deterministic after the top or bottom A gaps. Total-variation distance
to the expert's route distribution is 0.05–0.17. All FTLE analysis is of *this
policy's* modes, not the expert's.

**S1, spatial: 3-way choices and triple junctions (`s1_spatial.png`).** On 2-D
slices of the 24-D noise space, spanned by the discriminant directions, the FTLE ridges
coincide with the gap boundaries. Precision and recall are both **1.00** in all three
cases, including the two with a three-way choice. The ridges meet at a triple
junction, and there are no false ridges: the 99th-percentile FTLE off the boundaries is
−1.5, versus a median of +0.6 on them.

**S2, temporal: when the flow can see a decision (`s2_temporal.png`).**
Along 40 rollouts at g = 0.1:
- The A choice is effectively made by the **first chunk**. Its closed-loop entropy
  drops from 1.07 bits to 0.07 within one replan.
- The B choice stays open in closed loop (about 0.4 bits) from the start. But between
  x ≈ −2.2 and −1.6 the generative flow shows **no** bifurcation: chunk entropy is 0, max
  FTLE is −1.6, and no boundary exists to bisect. B's turn is beyond the chunk horizon,
  so the flow cannot see it.
- Once the robot enters column A, B's turn comes within range. The chunk entropy jumps
  to about 0.6 bits and tracks the closed-loop B entropy. FTLE and ridge height rise,
  peaking near x ≈ −1. The ridge height grows as the decision approaches and then the
  ridge disappears once B is decided (x > 0.2).
- Across states, max FTLE correlates with the pending decision's closed-loop entropy
  (Spearman 0.56); ridge height correlates at 0.55.

**FTLE is a horizon-limited detector.** It flags a bifurcation exactly when that
decision enters the action chunk, not earlier.

**S3, per-seed prediction of flips (`s3_prediction.png`).** 360 states visited in
rollouts with g ∈ [−0.9, 0.9], × 24 noise seeds = 8,640 cases. Targets:
- whether the pending gap changes under a 0.1 nudge to the seed or a ±0.05 nudge to the
  goal (open loop);
- whether the *final route* changes when only the first chunk's seed is nudged, or when
  the goal is nudged for the whole episode (closed loop).

AUROC with 95% cluster-bootstrap CIs:

| target (flips) | FTLE | goal sens. | Eulerian max σ | log-norm ∫μ | state entropy |
|---|---|---|---|---|---|
| open loop, noise (42) | **0.96** / 0.90 | 0.95 / 0.92 | 0.51 / 0.63 | 0.94 / 0.90 | 0.89 / – |
| open loop, goal (141) | **0.98** / 0.95 | 0.98 / 0.96 | 0.48 / 0.59 | 0.97 / 0.95 | 0.90 / – |
| closed loop, noise (29) | **0.95** / 0.93 | 0.90 / 0.90 | 0.48 / 0.69 | 0.95 / 0.89 | 0.89 / – |
| closed loop, goal (216) | 0.73 / 0.71 | 0.70 / 0.70 | 0.49 / 0.53 | 0.76 / 0.72 | 0.77 / – |

The first number in each cell is across all cases; the second is within states, i.e.
which seeds at the same state flip.

- **FTLE predicts which seeds flip, not just which states are uncertain.** Within a
  state, AUROC is 0.90–0.95. A state-level entropy score scores 0.5 by construction.
  76–83% of all noise-induced flips come from the top 1% of FTLE.
- **Open-loop fragility carries into behavior.** When a seed's chunk flips under the
  nudge, the final closed-loop route flips 55% of the time, versus 0.07% otherwise.
- **The Eulerian score fails.** The largest instantaneous stretch rate of the velocity
  field is at chance (0.48–0.51). Its integral, the log-norm, bounds the FTLE
  (FTLE ≤ ∫μ in 98.6% of cases; the rest are discretization error) and tracks it
  closely (r = 0.88), so it predicts almost as well. Here the information is in the
  accumulated stretching along the trajectory, not in any single instant.
- **Goal nudges held for a whole episode are predicted poorly (0.71–0.76).** The goal
  also changes every later chunk and the later B decision. FTLE of the first chunk
  cannot see those, which is the same horizon limit as S2.

## Caveats

- Flips are rare: 29–216 positives per target. CIs are wide for the closed-loop noise
  target. The "B-only" closed-loop flips for states before column A were too rare to
  test (2 cases).
- The perturbation sizes (0.1 in noise, 0.05 in goal) were fixed in advance and not
  swept.
- The policy underweights one mode, and there was one training run.
- The Eulerian zero-finding comparison from the GMM study was not repeated.

---

# Multi-replan (composed) FTLE

**Question.** The one-chunk FTLE can't see a decision until it enters the action
chunk. Can composing flow-map Jacobians across successive replans see further ahead?

**Method (`multichunk.py`, `mc_eval.py`, `plot_mc.py`).** The closed loop is the map
s_{k+1} = s_k + E·Φ(z_k; s_k, g), where Φ is the generative flow map and E sums the
4 executed steps. Forward-mode tangents of the flow ODE with respect to (x, y, g) give
∂Φ_k/∂s and ∂Φ_k/∂g at every replan. Chaining them gives the *total* sensitivity of
each future chunk to the first seed z₀ and to the goal g. The score is the largest
such sensitivity over future chunks while a decision is still pending, measured on
the **whole 12-step chunk**. All derivatives match finite differences (≤1e-7 relative).

The later seeds z₁, z₂, … must be fixed in order to compose. Four choices:
- the **actual** stream the evaluation used;
- **zero** noise (a noise-free lookahead);
- **random** independent streams, scored per seed;
- the random streams **aggregated per state**: the maximum over 24 seeds × 2 futures =
  48 sampled futures.

**Evaluation.** The same 8,640 cases and flip labels as S3; the replayed rollouts
match exactly. The main target is whether the route flips when the goal is nudged by
±0.05 for the whole episode. One-chunk FTLE predicted this poorly because the goal
also moves later decisions. Regions: x < −2.2 (A being chosen), −2.2…−1.5 (B beyond
the chunk), −1.5…0.5 (B inside the chunk).

| goal-flip AUROC (flips) | one chunk | zero noise | random, per seed | random → state risk | actual noise |
|---|---|---|---|---|---|
| all states (216) | 0.70 | 0.66 | 0.68 | 0.84 | **0.88** |
| only B flips (125) | 0.53 | 0.55 | 0.58 | 0.85 | **0.88** |
| B beyond the chunk (56) | 0.43 | 0.50 | 0.52 | 0.85 | **0.89** |
| B inside the chunk (50) | 0.70 | 0.62 | 0.64 | 0.88 | 0.88 |

Seed-nudge flips (29 in total): one-chunk 0.945, composed with actual noise 0.998.
Of these, 5 are B flips with B inside the chunk; there it goes from 0.70 to 0.999.

**Findings.**
1. **Composition sees beyond the chunk, given the future noise.** With the actual future
   noise, seed-level prediction of B flips while B is still beyond the chunk rises from
   0.43 (chance or worse) to 0.89. Within-state AUROC is 0.86, so it identifies *which
   seeds* will flip.
2. **The future noise decides.** With zero or independent random future noise, the
   per-seed score drops back to chance beyond the horizon (0.50–0.52). A later flip
   depends on where the *future* seeds land relative to the ridges, which is unknowable
   at the current replan. A policy that pre-draws its future seeds (fixed noise
   schedule) makes the exact monitor available at deployment.
3. **Without the future noise, compose over sampled futures and score the state.**
   Taking the maximum composed sensitivity over 48 sampled futures gives a state-level
   risk score: AUROC 0.85 beyond the horizon, versus 0.43 for one-chunk FTLE and 0.55
   for one-chunk mode entropy. Its Spearman correlation with the state's flip rate is
   0.67 (one-chunk entropy: 0.57 overall, 0.26 beyond the horizon). It cannot rank seeds
   within a state (0.5 by construction).
4. **What to compose matters.** Two natural alternatives failed: the norm of the
   robot's state sensitivity over the whole trajectory (0.42), and the sensitivity of
   y where the path crosses each column (0.74). Only 4 of 12 actions are executed, and
   a mode flip lives mostly in the **unexecuted tail** of a chunk. The state trajectory
   only registers it after the fact. The informative quantity is each future chunk's
   full sensitivity.
5. Where the A choice is still being made (x < −2.2), the one-chunk score is already
   good (0.85, and 0.93 for A flips). Composing adds noise there unless the actual
   future noise is used.

**Cost.** About 8.5 min per composed future on 2 CPU cores for 8,640 cases × 15 replans
(3 tangents per replan, plus a full noise Jacobian at the first replan).

**Caveats.** Only 2 random futures per seed (48 per state). Positives for seed-nudge
B flips are very few (5–7). One policy, one goal-nudge size.

---

# Lagrangian coherent structures: predicting when a mode switches

**Idea.** FTLE ranks how fragile a seed is, but it has no units. The LCS is the
actual separating surface in noise space. Locating it turns fragility into a
**margin**: how far a seed (or the goal) must move before the mode switches.

**Method (`lcs.py`, `lcs_eval.py`, `plot_lcs.py`).** Repelling LCS are normal to the
most-stretched direction of the flow map, ξ_max (the top right singular vector of
∂Φ/∂z). From each seed, search along z + s·ξ_max for s ∈ [−3, 3] in steps of 0.1:
- the **ridge**: where the directional stretch ‖∂Φ/∂s‖ peaks. This uses no labels;
  it is the LCS proper.
- the **separatrix**: where the gap label changes, refined by bisection. This is the
  ground truth.

The margins d₊ and d₋ are the distances to the nearest separatrix on each side. The
LCS's velocity with respect to the goal, ds*/dg = −(w·∂Φ/∂g)/‖∂Φ/∂s‖, turns a noise
margin into a goal margin. Treating the LCS as locally flat, a nudge of size ε in a
random direction crosses it with probability P(ε·uₙ > d), where uₙ is one component
of a uniform unit vector in 24-D (a Beta distribution). **No parameters are fitted.**
All derivatives were checked against finite differences.

Evaluated on the same 8,640 cases (360 states × 24 seeds) with fresh perturbation
sweeps.

**1. The FTLE ridge is the separatrix (`lcs_prediction.png`, left).** When the
separatrix is within 0.5 of the seed, the label-free ridge lies within 0.1 of it in
73% of cases (median gap 0.03). 11% of cases have an LCS within 3 units along ξ_max;
most seeds sit deep inside a basin.

**2. Seed nudges of any size are predicted without refitting.**

| nudge ε | 0.03 | 0.05 | 0.1 | 0.2 | 0.4 |
|---|---|---|---|---|---|
| observed flip rate | 0.19% | 0.22% | 0.56% | 1.00% | 1.93% |
| LCS predicted (analytic) | 0.18% | 0.27% | 0.51% | 0.95% | 1.78% |
| FTLE → logistic fit at ε = 0.1 | 0.56% | 0.56% | 0.56% | 0.56% | 0.56% |
| AUROC, LCS / FTLE | 0.997 / 0.89 | 0.999 / 0.98 | 0.999 / 0.95 | 0.998 / 0.97 | 0.997 / 0.97 |
| Brier skill, LCS | 0.49 | 0.56 | 0.71 | 0.66 | 0.70 |

FTLE ranks seeds well, but a mapping from FTLE to probability fitted at one nudge size
cannot transfer to other sizes. The LCS margin has units, so it does.

**3. Goal nudges: a deterministic switching threshold.** Rule: predict a flip when the
goal margin is at most δ.

| goal δ | flips | precision | recall | AUROC (margin) | AUROC (one-chunk ‖∂Φ/∂g‖) |
|---|---|---|---|---|---|
| ±0.02 | 63 | 0.98 | 0.97 | 0.997 | 0.957 |
| ±0.05 | 141 | 0.99 | 0.98 | 0.999 | 0.976 |
| ±0.1 | 289 | 1.00 | 0.99 | 0.999 | 0.968 |
| ±0.2 | 555 | 0.98 | 0.87 | 0.950 | 0.880 |

Recall drops at ±0.2 because some separatrices lie beyond the search range, or the
first-order velocity no longer holds.

**4. When does a switch happen along a rollout? (`lcs_timing.png`)**
- *Fresh noise at every replan* (the default). Whether the next plan switches gap
  depends on the **next** fresh seed. The drawn seed's LCS margin (AUROC 0.63) and its
  FTLE (0.61) carry little information. The probability mass beyond the boundary, from
  256 Monte Carlo seeds, predicts well (0.86). A single line search from one seed
  usually doesn't reach the boundary (only 11% of seeds have one within 3), so it is
  not a cheap substitute here.
- *One seed reused at every replan* (a fixed noise schedule). Switches occur when the
  LCS, moving with the robot's state, sweeps past the seed. One replan before a switch,
  an LCS lies within 3 of the seed 63% of the time, versus 12% otherwise (AUROC 0.76;
  FTLE 0.75). A first-order prediction of how the LCS moves with the next executed
  displacement adds little (0.76; 35% recall, 22% precision for "margin ≤ 0"). The LCS
  moves about 1–1.5 noise units per replan, which is too far for a linear forecast.

**Bottom line.** Within one replan, the LCS gives quantitative, calibrated
predictions: which seeds switch, for nudges of any size, and a goal threshold that is
almost exact. For timing over successive replans, it works only when the noise is
fixed in advance, and even then it is a coarse warning rather than a precise clock.

**Cost.** The 61-point line search took about 22 min for 8,640 cases; bisection and
LCS velocities about 1 min.

---

# How well the LCS predicts a mode switch of its own chunk

**Scope.** Only the chunk the LCS was computed for: does *that chunk's* pending-gap
label change under a perturbation? There is no closed loop and no later replans. Same
8,640 cases (360 states × 24 seeds) and LCS as the previous section; new tests in
`chunk_eval.py`, analysis in `plot_chunk.py`, numbers in `results_chunk.json`.

**1. Per perturbation, the flat-LCS rule is almost always right.** For the 740 seeds
with an LCS within 1 unit, I drew 8 random nudges per seed per size and predicted
"flip iff the nudge's component along the LCS normal exceeds the margin":

| nudge ε | perturbations | flips | precision | recall | accuracy |
|---|---|---|---|---|---|
| 0.1 | 5,920 | 126 | 0.96 | 0.88 | 0.997 |
| 0.2 | 5,920 | 240 | 0.97 | 0.93 | 0.996 |
| 0.4 | 5,920 | 479 | 0.96 | 0.94 | 0.992 |

Recall is a little below precision, matching the small under-prediction of flip
rates seen earlier.

**2. The LCS is locally flat, and it names the destination mode (`chunk_lcs.png`, left).**
Pushing seeds toward the LCS at an angle θ to its normal, the crossing distance is
d / cos θ. The median ratio of actual to predicted is 0.98–1.00 at every angle up to
85°. 92% of crossings fall within 10% of the prediction at 30°, 71% at 60°, and 44% at
85°. The boundary's depth does not drift until about 1–2 noise units sideways. The mode
the chunk switches *to* is the one beyond the LCS along the normal in **100%** of
crossings, including at three-way junctions.

**3. Goal margins are exact for goal changes up to about ±0.2 (`chunk_lcs.png`, middle).**
Exact margins were found by bisecting the goal change that switches each chunk
(818 seeds, both directions):

| exact margin | 0–0.05 | 0.05–0.1 | 0.1–0.2 | 0.2–0.4 | 0.4–0.6 |
|---|---|---|---|---|---|
| within 10% (up / down) | 92% / 95% | 97% / 96% | 94% / 88% | 63% / 73% | 58% / 82% |

The LCS says which direction of goal change switches the chunk first in 99.7% of cases.
Of 70 switches within ±0.2 that the threshold rule missed, 14 are near-misses
(predicted ≤ 0.25). The other 52 have no LCS within 3 along the seed's
most-stretched direction: the goal change brings in a boundary from elsewhere.

**4. It holds in every context (`chunk_lcs.png`, right).** AUROC of the margin is
0.996–1.000 whether the pending choice is 3-way (column A) or 2-way (column B), and
whether the chunk reaches the column or not. One-chunk FTLE drops to 0.82 (ε = 0.1)
for column-B choices beyond the chunk's end. Most noise flips (39 of 48 at ε = 0.1)
occur where the decision lies beyond the chunk's end. There the label is the chunk's
heading, which is the more fragile case.

**5. Without labels (`chunk_lcs_labelfree.png`).** A deployed policy has no gap
labeller, so I located the LCS from the flow alone. Along the LCS normal, find where
the chunk itself jumps (a ridge can be narrower than any grid, but its jump cannot
hide), then refine by bisection on the chunk. One threshold on the jump size
(τ = 0.3) was chosen on even-numbered states and tested on odd ones.
- Where a real boundary exists, it is located to a median of 0.0004–0.0008 noise
  units. There are 728 real boundaries detected against 19 spurious ones.
- **3-way choice (A): label-free is as good as labelled**, AUROC 0.99 vs 0.999,
  Brier skill 0.63–0.73.
- **2-way choice (B): label-free reaches only 0.74–0.88.** All 564 column-A
  separatrices are *hard*: the chunk jumps by more than 1. 378 of 542 column-B
  separatrices are *soft* (jump < 0.3): the chunk varies smoothly and only the
  labeller's threshold flips. This holds even when B lies inside the chunk (30 of 50
  soft), though those jumps (0.08–0.2) are larger than when B is beyond the chunk end.
  A soft label boundary is not a bifurcation of the flow, so no flow-based detector
  can see it. Whether it matters depends on whether that label distinction matters for
  behavior.

**Bottom line.** For its own chunk, the LCS is an essentially complete account of
mode switching wherever the switch is a real bifurcation of the flow:
- it predicts which perturbations switch the chunk (≥ 99% accuracy per perturbation);
- it predicts which mode results (100%);
- it gives exact goal margins up to ±0.2, and the direction of goal change that
  switches first (99.7%).

The label-free version matches this for the hard (3-way) choice. The remaining gap is
"soft" label boundaries, which the flow does not treat as bifurcations.

**Caveats.** Flip counts are small in some subgroups (2 flips for column B inside the
chunk at ε = 0.1; 1–2 held-out B flips at ε ≤ 0.05, which is why those are not plotted).
One policy, one labeller: hard vs soft depends on how modes are defined. Cost on 2 shared
CPU cores: angle tests about 6 min, goal bisection under 1 min, label-free scan about
5 min.
