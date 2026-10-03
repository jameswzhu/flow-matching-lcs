# Language-conditioned shapes task

## Task

A point robot starts near the origin (jitter ±0.3). Four goal objects, one of each kind
(red/blue × square/circle), sit at random angles on a ring of radius R around the
origin, at least 50° apart. Their slot order in the observation is shuffled every
episode, so the policy has to read the attributes rather than the slot.

The prompt names a set of matching kinds. The expert picks a matching object uniformly
and drives straight to it at 0.1 per step, then stays:

| specificity | training prompts | zero-shot prompts |
|---|---|---|
| 1 match | red square, red circle, blue square, blue circle | red box, blue box, red disk, blue disk, crimson square, navy circle |
| 2 matches | square, circle, red, blue | box, round object, a red thing, something blue |
| 4 matches | shape | any object, object |

The held-out run drops "blue circle" from training.

**Chunk reach vs. object distance.** A 12-step chunk reaches 1.2. R is drawn from
U[0.8, 2.4] per episode (0.67–2× the chunk reach) and stored, so FTLE results can be
split by whether the object lies inside the first chunk.

**Policy.** A flow-matching model over 12-step chunks (24-D), conditioned on 16 object
features and a frozen prompt embedding. The object features are each object's position
relative to the robot, plus colour ±1 and shape ±1. The embedding is standardised with
the mean and RMS of the training prompts' vectors, and the same affine map is applied to
unseen prompts. Sampling uses 20 RK4 steps. The policy executes 4 actions and then
replans. Written in PyTorch. Flow-map Jacobians come from `torch.func.jacfwd` / `jvp`
and match central finite differences: 2e-9 with respect to noise, 9e-8 along a prompt-embedding
direction.

**Architecture: an MLP is not enough.** The first version was a 4×256 MLP on the
concatenated slots, like robot2. After 40k steps with the oracle encoder it had not
learned the binding from attributes to slot. On 1-match prompts it chose a matching
object 30% of the time, against 25% for chance. In closed loop it re-picked an object at
random at each replan: 50% of rollouts switched target, and 84% never arrived anywhere.

The current model uses object attention. Each object becomes a token
f([rel pos, colour, shape, e]). A query stream g([a, τ, e]) attends over the tokens
three times; width is 128 and there are 4 heads. The model is permutation-invariant by
construction. The noisy chunk enters the query, so the noise selects among several
matching objects. The attention is written out explicitly because PyTorch's fused CPU
kernel has no forward-mode AD. The tokens are computed once per flow. Training: 20k
steps, batch 1024, about 13 min on CPU.

**Encoders.** The `embed.py` encoders are:

| encoder | dim | pairwise cosine of the 21 prompts (min / median / max) |
|---|---|---|
| CLIP ViT-B/32 text tower | 512 | 0.67 / 0.80 / 0.97 |
| all-MiniLM-L6-v2 (sentence-transformers) | 384 | 0.09 / 0.36 / 0.87 |
| oracle | 4 | multi-hot [red, blue, square, circle] of the intended match set |

The oracle is a control with perfect attribute binding; paraphrases map exactly onto
their canonical prompt.

## Setup and running on the GPU

Build the virtual environment per machine: the CUDA build of torch depends on the GPU
generation. From the project root:

| machine | GPU | torch wheel index |
|---|---|---|
| laptop / desktop 1 | GTX 1050 Ti (Pascal, sm_61) | `https://download.pytorch.org/whl/cu126` (12.8+ wheels dropped Pascal) |
| desktop 2 | RTX 5090 (Blackwell, sm_120) | `https://download.pytorch.org/whl/cu128` or newer (older wheels lack sm_120) |

    python -m venv .venv
    .venv/Scripts/python -m pip install torch --index-url <index above>   # Linux: .venv/bin/python
    .venv/Scripts/python -m pip install numpy scipy matplotlib transformers sentence-transformers
    .venv/Scripts/python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0), torch.cuda.get_arch_list())"

Check the new machine against the stored checkpoints, from `robot3/`:

    ../.venv/Scripts/python policy3.py check policies/clip.pt     # CPU float64: expect ~1e-9
    ../.venv/Scripts/python eval3.py clip slice plot             # GPU; compare figs/slice_clip.png

Training and evaluation pick CUDA automatically:

    ../.venv/Scripts/python policy3.py clip [holdout]
    ../.venv/Scripts/python eval3.py clip rollout ftle slice plot

Precision: float32 on GPU, float64 on CPU (`TORCH_DEVICE`, `TORCH_DTYPE` and
`TORCH_THREADS` override). Consumer GPUs run float64 at 1/32 (Pascal) to 1/64
(Blackwell) of the float32 rate. Leave TF32 off (the torch default) for Jacobian work.
For 500 flow Jacobians on the 1050 Ti, the time and maximum relative error against CPU
float64 are:

| setting | time | max relative error |
|---|---|---|
| CPU float64 | 86 s | – |
| GPU float32 | 4.8 s | 9e-4 |
| GPU float64 | 15 s | 3e-12 |

Use float64 where precision matters, e.g. LCS bisection. A training step takes 21 ms on
the 1050 Ti vs 37–150 ms on the CPU (depending on load). `policy3.py check` always runs
on the CPU in float64. The prompt embeddings are stored in `emb/`, so the encoders are
only needed to embed new prompts.

## Files

| File | Purpose |
|---|---|
| `env3.py` | Scenes, prompts and match sets, expert, demos, chunk and closed-loop labels |
| `embed.py` | `python embed.py clip st oracle` → `emb/<enc>.npz` |
| `policy3.py` | `python policy3.py <enc> [holdout]` → `policies/<enc>[_holdout].pt`; `python policy3.py check <ckpt>` checks the Jacobians against finite differences |
| `eval3.py` | `python eval3.py <policy> rollout ftle slice plot` → `out/<policy>/`, `figs/` |

## Results

Four policies, one training run each: oracle, MiniLM (`st`), CLIP, and CLIP with "blue
circle" held out. Closed loop uses 100 scenes × 16 seeds per prompt. FTLE is the
first-chunk FTLE at the start state, over 100 scenes × 32 seeds. **Success** = the first
object reached matches the prompt. **Switch** = the chunk's target object changed between
replans before arrival. Raw numbers are in `out/<policy>/rollout.json` and `ftle.npz`.

### Training prompts: all four encoders agree

| | success | per-scene entropy / max | switch | FTLE p99 | seeds with FTLE > 0 |
|---|---|---|---|---|---|
| 1 match | 0.999–1.000 | – | 0.0–0.2% | −1.47 … −1.39 | 0.03–0.06% |
| 2 matches | 1.000 | 0.93–0.96 | 4.9–6.7% | +0.42 … +1.09 | 1.8–2.4% |
| "shape" (4) | 1.000 | 0.92–0.93 | 13.7–14.9% | +1.37 … +1.66 | 3.9–4.6% |

- **Ridges appear only for ambiguous prompts.** The FTLE tail and the rate of
  mid-course switching both grow with the number of matches. Single-match prompts give
  a flat field with no ridge.
- **Slices (`figs/slice_<policy>.png`).** These are 2-D noise slices spanned by the
  top two principal components of the per-mode mean seeds under "shape". In them:
  - "blue square" is a single basin;
  - "square" has one ridge, which lies exactly on the label boundary;
  - "shape" has four basins whose ridges cross near the origin. At 0.05 grid resolution
    this looks like a 4-way junction or two very close triple junctions; it hasn't been
    resolved further.

  For CLIP, "blue square" also shows a faint high-FTLE band in one corner with no label
  change inside the slice. It's possibly a boundary toward the blue circle just outside
  the slice. Not yet checked.

### Zero-shot paraphrases: language fragility shows up in the flow

| prompt (matches) | MiniLM: success / switch / FTLE p99 | CLIP | where the misses go (CLIP) |
|---|---|---|---|
| navy circle (1) | 0.50 / 5.4% / +1.22 | 0.62 / 5.4% / +0.91 | red circle 38% (MiniLM: 50%) |
| red box (1) | 0.80 / 5.3% / +0.16 | 0.89 / 5.0% / +0.56 | red circle 11% |
| blue box (1) | 0.85 / 4.2% / +0.31 | 0.93 / 3.1% / +0.37 | blue circle 7% |
| crimson square (1) | 0.98 / 1.6% / −0.37 | 0.95 / 3.4% / +0.44 | |
| red disk (1) | 0.94 / 2.8% / +0.17 | 0.98 / 1.8% / −0.49 | |
| box (2) | 0.95 / 11.4% / +0.79 | 0.85 / 11.7% / +1.55 | red circle 14%, blue circle 2%; of the squares, 71% red vs 14% blue |
| round object (2) | 0.76 / 11.8% / +2.19 | 0.99 / 5.4% / +1.26 | 77% red circle vs 22% blue |
| any object, object (4) | 1.00 | 1.00 | skewed to the red circle (57–69%); per-scene entropy / max 0.55–0.69 |

- **A paraphrase that the encoder places between training prompts creates ambiguity
  that isn't in the language.** Single-match paraphrases get an FTLE tail (1.0–1.5% of
  seeds have FTLE > 0, against 0.04% for trained single-match prompts) and mid-course
  switching (2–5% vs 0.1%).
- **The failures depend on the encoder.** MiniLM reads "navy circle" as "circle" (an
  exact 50/50 split) and fails on "round object". CLIP handles "round object" but
  collapses "box" onto the red square. Ridges track the embedding geometry, so a
  second encoder is needed to separate language effects from encoder effects, as
  CLAUDE.md anticipated.
- **The oracle encoder maps every paraphrase onto its canonical prompt** and is
  perfect by construction. It is the reference for what the flow does when attribute
  binding is perfect.

### Held-out combination (CLIP, "blue circle" never trained)

Zero-shot, "blue circle" reaches a blue circle 99.1% of the time (0.9% go to the red
circle), with 1.3% switching and an FTLE p99 of −1.24. The trained version gives
−1.44 and 0.1%. Composition from "blue" + "circle" works, and leaves only a slight raise
in sensitivity. The held-out policy is otherwise like the CLIP policy, though "navy
circle" gets worse (0.40 success).

## Caveats

- One training run per encoder. Paraphrase numbers can vary between runs in a way the
  training-prompt numbers likely won't; not measured.
- R is sampled per episode but **FTLE has not yet been split by R / chunk reach.** With
  a straight-line expert the decision is visible in the first chunk at every R, so the
  "decision beyond the chunk" regime of robot2 doesn't arise here unless the expert is
  changed (e.g. delayed commitment).
- No confidence intervals yet. Per CLAUDE.md, the next analyses should report AUROC with
  95% bootstrap intervals clustered by scene.

## Next steps

1. Ridge position vs Φ⁻¹(p): for 2-match prompts, line search and bisect from each seed
   (port `robot2/lcs.py` to `policy3.flow`). Compare the boundary's offset along its
   normal with Φ⁻¹ of the per-scene mode share.
2. LCS margins in prompt-embedding space: line search along e(paraphrase) −
   e(canonical) using `Policy.jvp_cond`. The margin to the first label change should
   predict which paraphrases fail.
3. Per-seed flip prediction under small noise / embedding nudges, as in robot2 S3.
4. Split FTLE by R relative to chunk reach. Optionally add an expert that commits late,
   so that decisions can lie beyond the chunk.
