# Project: Lagrangian coherent structures (LCS) of flow-matching policies

Research code for a paper on using dynamical-systems tools (FTLE, LCS, contraction) to
analyze the stability and mode switching of flow-matching robot policies under changes
in their conditioning (goal, prompt).

## Core idea

A flow-matching policy maps a noise seed z to an action chunk Φ(z; state, conditioning)
by integrating a learned velocity field over flow time τ ∈ [0, 1]. The flow partitions
noise space into basins, one per behaviour mode. FTLE = log σ_max(∂Φ/∂z); its ridges
are repelling LCS = the boundaries between modes. Locating the LCS from a seed (line
search along the top right singular vector of ∂Φ/∂z, then bisection on the mode label)
gives a switching margin with units, which predicts mode switches for perturbations of
any size, with no fitting.

## Layout

| Path | What it is |
|---|---|
| `gmmflow.py`, `experiments.py` | Closed-form 2-D Gaussian-mixture flow (exact velocity field) and the first FTLE experiments. `README.md` has results. |
| `robot/` | One-obstacle point robot, 16-D chunks, learned policy (`policy.npz`). `README.md` has results. |
| `robot2/` | Two obstacle columns (3-way then 2-way choice), 24-D chunks, learned policy (`policy2.npz`). All LCS tooling lives here. `README.md` has every study's method and results. |
| `robot2/lcs.py` | LCS primitives: directional JVP of the flow, line search, bisection, ridge finder, analytic flip probability. |
| `robot2/multichunk.py` | Composing flow-map Jacobians across replans. |
| `robot2/*_eval.py`, `robot2/plot_*.py` | Experiments (cached outputs in `robot2/lcs/`, `robot2/mc/`, `robot2/chunk/`) and their figures. |
| `robot/mlp.py` | Hand-written NumPy MLP with backprop, Adam and forward-mode JVP. |
| `robot3/` | Language-conditioned shapes task (PyTorch, object-attention flow policy, CLIP / MiniLM / oracle prompt embeddings). `README.md` has setup, GPU notes and results. |

`robot2/mc/prep.npz` (first-chunk Jacobians, 23 MB) is not in the bundle; rebuild it with
`cd robot2 && python mc_eval.py prep` (about 4 min) before running the LCS or chunk studies.

## Conventions

- Everything was written in plain NumPy because the original workspace could not install
  packages. Moving to PyTorch (`torch.func.jvp`, `vmap`) is welcome and will be much faster.
- Every derivative was checked against finite differences before use; keep doing that.
- Experiments cache results to `.npz` / `.json` so analysis and figures rerun without
  recomputing. Each study has a `python <x>_eval.py <step>` interface.
- Figures use a light surface, recessive axes, a single-hue blue sequential ramp and a fixed
  categorical order (blue, orange, aqua, yellow, magenta).
- Report AUROC overall and within state, with 95% bootstrap intervals clustered by state.

## Key findings so far (details in robot2/README.md)

- The FTLE ridge is the basin boundary in every system tested (toy, 16-D, 24-D).
- FTLE measures sensitivity, not the number of modes; Eulerian fixed points mislead.
- One-chunk FTLE only sees decisions inside the current chunk; composing Jacobians across
  replans sees further, but only given the future noise (else per state, by sampling).
- For its own chunk, the LCS predicts 99% of individual perturbations, the destination mode
  in 100% of crossings, and goal margins within 10% up to ±0.2.
- Label-free LCS (where the chunk jumps along the normal) matches the labelled version for
  hard boundaries; "soft" label boundaries are not bifurcations of the flow.

## Language-conditioned shapes environment (implemented in `robot3/`; first results in its README)

Build a 2-D point-robot task with four goal objects (red/blue × square/circle) at random
positions on a ring around the start. The policy is a flow-matching model over 12-step
action chunks, conditioned on the robot state, each object's position/colour/shape, and a
frozen text embedding of the goal prompt.

- Prompts at three levels of specificity: "blue square" (1 match), "square" / "red"
  (2 matches), "shape" (4 matches). Demos go to a matching object.
- First run: include ambiguous prompts in training (ground-truth multimodality). Then test
  zero-shot ambiguity and paraphrases ("box"); hold out one combination (e.g. "blue circle").
- Embed prompts once with CLIP's text encoder (`open_clip` or `transformers`), save the
  vectors, train on them. Repeat with a second encoder (e.g. a sentence-transformer) to
  check that ridges track the language, not CLIP's weak attribute binding.
- Keep object distance relative to chunk reach a deliberate variable (FTLE only sees
  decisions inside the chunk).
- Predictions to test: ridges only for ambiguous prompts; ridge position at Φ⁻¹(p) of the
  mode probabilities; a 4-way junction for "shape"; LCS margins in prompt-embedding space as
  a measure of paraphrase fragility; mid-course switching under ambiguous prompts.
