# Validation on a learned, goal-conditioned robot policy

This tests whether the 2D-GMM findings (parent folder) survive on a learned,
higher-dimensional, closed-loop system.

## Task

A 2D point robot starts at the left and must reach a goal at (2, g). On the way it
passes a disk obstacle either **over** or **under** it. The goal height g ∈ [−1, 1]
is the conditioning, playing the role of the prompt. The expert takes the over-route
with probability σ(3g) and follows a smooth path through a waypoint.

The policy is a flow-matching model over **action chunks of 8 displacements (16-D)**.
It is conditioned on (position, goal) and trained on 3,000 expert episodes:
an MLP with 3×256 SiLU layers, 30k Adam steps, written in NumPy because package installs
were blocked here. It runs closed loop by executing 4 actions and then replanning.
Reference sampling is RK4 with 30 steps.

Flow-map Jacobians come from forward-mode tangents through the network, integrated with
the variational equation. They match finite differences to 1e-10. FTLE in float32 agrees
with float64 to 6e-5.

## Files

| File | Purpose |
|---|---|
| `robot_task.py` | Environment, expert, demos, route label |
| `mlp.py` | NumPy MLP: backprop + Adam, forward-mode Jacobian (`jvp`) |
| `policy.py` | Training (`python policy.py 30000`), sampling, flow-map Jacobian w.r.t. noise and goal |
| `evalutil.py` | Open-loop route probability, closed-loop rollouts |
| `validate.py` | All checks: `python validate.py [0 1 2 2b 3 4]` (~35 min on 2 CPU cores) |
| `policy.npz`, `results.json`, `figs/` | Trained weights, numbers, figures |

## Results

**V0: the policy imitates the expert (`v0_imitation.png`).** Open-loop route
probability is within 0.044 of the expert across g, with zero collisions. In closed loop
the route probability is pulled toward 0.5 (largest gap 0.10). This happens because
replanning re-samples before the robot commits, and a few rollouts switch route mid-way.

**V1: the FTLE ridge is the route boundary (`v1_slices.png`). Confirmed.**
- Along the noise direction that best separates the routes, the ridge sits on the
  label flip to within **0.005** at all 5 goals tested.
- On 2D slices, **100%** of the top-1% FTLE pixels lie on the label boundary.
- The boundary is nearly a hyperplane: a linear classifier separates the routes at
  98–99% accuracy.
- Its location is predicted by the route probability, Φ⁻¹(1 − P(over)), to within
  0.14. The residual is the boundary's curvature.

**V2: the boundary sweeps through noise as the goal moves (`v2_joint_noise_goal.png`).**
The ridge in the (noise, goal) plane follows the label boundary. The Φ⁻¹(1−p)
prediction tracks it to within 0.08 on average (max 0.18). Again, the residual comes from
using one fixed direction for all goals.

**V2b: rare routes are fragile (`v2_rare_route_fragility.png`). Confirmed.**
Each seed's goal is nudged by 0.05 toward the other route.
- When P(over) = 0.06, **13.6%** of over-seeds switch, versus 1.0% of under-seeds in
  the mirror case.
- Not a single seed switched in the "wrong" direction, so the boundary sweeps
  monotonically. The flip rate therefore equals Δp / P(route) exactly; the dotted
  prediction lines are hidden under the data.

**V3: FTLE as a route counter (`v3_ftle_vs_route_count.png`). Consistent, but weak.**
Across 39 states, the max FTLE falls smoothly as the minority route's mass goes to zero,
with no threshold. Some states where the minority route was never observed still show
positive stretching. The estimator is confounded, though. Max FTLE over 2,000 random
seeds only finds the ridge if some seed lands near it, and that gets rarer as the
boundary moves into the tail.

**V4: few-step Euler biases the route preference (`v4_euler.png`). Partly confirmed.**
- Away from balance, Euler sampling makes the minority route rarer, as in the GMM. The
  largest open-loop error is 5.9% at N = 2, and it shrinks to under 1% at N = 20.
- Unlike the symmetric GMM, the neutral point sits near P(over) ≈ 0.4, not 0.5.
- The error isn't monotone in N for g > 0 (N = 3 is worse than N = 2).
- Closed loop shows the same sign pattern (±4.5% at N = 2). With 400 episodes,
  differences under about 0.03 are within noise.
- No collisions at any N.

## What did not carry over or is not yet tested

- The Eulerian-vs-Lagrangian comparison (GMM Fig 4) was not repeated. Finding all
  zeros of a 16-D learned velocity field needs a dedicated solver.
- V3 needs a ridge-finding estimator that doesn't rely on random seeds hitting the
  ridge, such as a line search along the separating direction.
- Only one training seed and one architecture were used.
