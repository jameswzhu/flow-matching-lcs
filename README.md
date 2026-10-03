# FTLE / Lagrangian coherent structures of a conditional flow-matching map — 2D GMM testbed

A closed-form testbed for studying how the generative flow of a flow-matching
policy partitions noise space into behaviour basins, and how those basins move
when the conditioning changes.

## Files

| File | What it is |
|---|---|
| `gmmflow.py` | Exact flow-matching velocity for a GMM target (linear / rectified-flow path), its analytic spatial Jacobian, RK4 or Euler integration of the flow and the variational equation, FTLE. |
| `experiments.py` | Five experiments; writes `figs/` and `results.json`. Run all with `python experiments.py`, or a subset with `python experiments.py 4 5`. Experiments 1–2 take ~5 min on CPU. |
| `sanity_checks.py` | Sampled mode masses equal the target weights; variational Jacobian equals finite differences (rel. error 1.6e-6). |
| `results.json` | Numbers behind every figure. |

Only NumPy, SciPy, Matplotlib.

## Model

Path x_τ = (1−τ)x₀ + τx₁, x₀ ~ N(0, I), x₁ ~ Σ_k w_k(c) N(μ_k(c), σ_k² I).
Each component is a Gaussian path with velocity v_k = μ_k + (s_k′/s_k)(x − τμ_k),
s_k² = (1−τ)² + τ²σ_k², and the exact marginal velocity is the posterior average
v = Σ_k r_k(x,τ) v_k. This form is regular at τ = 1. The conditioning c enters only
through the mixture parameters. FTLE = log σ_max(∂Φ/∂a₀) over flow time [0, 1].

## Results

**1. The FTLE ridge is the basin boundary (fig1).** Two modes, left weight c.
Because the modes lie on one axis with equal widths, the dynamics along that axis
are a 1D ODE, so the flow map is the monotone rearrangement and the boundary is
exactly b(c) = Φ⁻¹(F_target(0)). The computed ridge matches to ≤ 3e-4 for
c = 0.5, 0.3, 0.1.

**2. Ridges in joint (noise, conditioning) space (fig2a, fig2b).**
Augmenting the state with c (dc/dτ = 0) gives a field whose ridge is the set of
(seed, prompt) pairs where an infinitesimal prompt change flips the mode. The
joint FTLE is largest when one mode is rare (c near 0 or 1), because
|db/dc| = 1/φ(b) blows up there: *rare behaviours are the most prompt-fragile.*
With three modes (non-separable), the top 1% FTLE pixels all lie within 2 px of
basin edges computed independently from endpoint labels, and basin noise-mass
matches the weights to ~1e-3, at every c tested.

**3. FTLE ridges do not mark the bimodality transition (fig3).** Merging two
equal modes, the ridge height equals log φ(0) − log p_target(0) exactly (error 8e-4)
and the ridge curvature equals −1 + (1 − m²/σ²)e^{m²/σ²} (rel. error 2e-3).
The curvature is negative for every m > 0, so a ridge exists even when the
target is unimodal (m < σ). Nothing qualitative happens at m = σ. **Implication:
FTLE measures sensitivity, not mode count.** Bifurcations in the number of
behaviours need a separate criterion (label/endpoint topology or target critical
points); a ridge-height threshold is one candidate.

**4. Eulerian fixed points mislead (fig4).** At c = 0.2 the instantaneous
field v(·, τ) undergoes a saddle-node at τ ≈ 0.28, creating an unstable zero at
a = −1.39 while the true separatrix is at −0.37. Pulling that Eulerian point back
to τ = 0 places the boundary at −1.91, implying a 2.8% left-mode probability
versus the true 20%. The unstable zero and the separatrix only coincide late
(τ ≳ 0.6). The stable zeros also escape to infinity as τ → 1/(1+σ²), where the
per-component rate s′/s changes sign. This is the classic fluid-dynamics warning
that instantaneous structure is not transport structure, now in a generative flow.

**5. Few-step Euler moves the separatrix (fig5).** The boundary shift scales
~1/N (0.94, 0.47, 0.26, 0.12, 0.06, 0.02 for N = 2, 3, 5, 10, 20, 50). The
induced mode-probability error peaks around 15% at N = 2 and 2% at N = 10.
Its sign makes the minority mode rarer, so discretization acts like an implicit
sharpening toward the dominant behaviour.

## Caveats

- The two-mode results rely on separability. The three-mode case confirms ridges
  equal basin edges without it, but has no closed-form boundary.
- The joint (noise, c) FTLE depends on how c is scaled relative to noise; report
  the c-column of the Jacobian separately when comparing across parameterizations.
- These are optimal velocity fields. Learned fields will smooth the ridges, which
  is the natural next experiment.

## Next steps worth trying

1. Train a small MLP flow-matching model on the same GMMs and compare its FTLE
   ridges with the exact ones (ridge position, height, and width).
2. Add classifier-free guidance (v_u + w(v_c − v_u)) and track how the
   separatrix moves with w.
3. Find a criterion that *does* detect mode birth: for example, a ridge-height
   threshold, or counting critical points of the pushforward density.
