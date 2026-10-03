"""
Flow-matching policy for the obstacle task: training, sampling, and the
flow-map Jacobian (variational equation) w.r.t. the noise seed and the goal g.

    python policy.py            # trains and saves policy.npz
"""
import time
import numpy as np
from mlp import MLP, tau_features
from robot_task import make_demos, norm_obs, OBS_STD, H

A = 2 * H          # action-chunk dim (16)
NF = 8             # tau features
D_IN = A + 3 + NF


class Policy:
    def __init__(self, net):
        self.net = net

    # ---- velocity field -----------------------------------------------------
    def _inp(self, a, tau, obs_n):
        N = a.shape[0]
        tf = tau_features(np.full(N, tau))
        ob = np.broadcast_to(obs_n, (N, 3))
        return np.concatenate([a, ob, tf], 1).astype(self.net.params[0][0].dtype)

    def v(self, a, tau, obs_n):
        return self.net.forward(self._inp(a, tau, obs_n))

    # ---- sampling -----------------------------------------------------------
    def sample(self, a0, obs_n, n_steps=30, method="rk4"):
        a = a0.copy(); h = 1.0 / n_steps
        for i in range(n_steps):
            t = i * h
            if method == "euler":
                a = a + h * self.v(a, t, obs_n)
            else:
                k1 = self.v(a, t, obs_n); k2 = self.v(a + .5 * h * k1, t + .5 * h, obs_n)
                k3 = self.v(a + .5 * h * k2, t + .5 * h, obs_n); k4 = self.v(a + h * k3, t + h, obs_n)
                a = a + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        return a

    # ---- flow map + Jacobian -------------------------------------------------
    def _vjac(self, a, tau, obs_n, with_g):
        T = np.zeros((D_IN, A + int(with_g)))
        T[:A, :A] = np.eye(A)
        if with_g:
            T[A + 2, A] = 1.0 / OBS_STD[2]      # d(normalised g)/dg
        v, JT = self.net.jvp(self._inp(a, tau, obs_n), T.astype(self.net.params[0][0].dtype))
        return v, JT[:, :, :A], (JT[:, :, A] if with_g else None)

    def sample_jac(self, a0, obs_n, n_steps=30, with_g=False):
        """RK4 on (a, F, f_g):  a' = v,  F' = J_a F,  f_g' = J_a f_g + J_g.
        Returns a1, F = da1/da0 [N, A, A], f_g = da1/dg [N, A] (or None)."""
        N = a0.shape[0]
        a = a0.copy(); F = np.broadcast_to(np.eye(A), (N, A, A)).copy()
        fg = np.zeros((N, A)) if with_g else None
        h = 1.0 / n_steps

        def rhs(a, F, fg, t):
            v, Ja, Jg = self._vjac(a, t, obs_n, with_g)
            dF = Ja @ F
            dfg = (np.einsum("nij,nj->ni", Ja, fg) + Jg) if with_g else None
            return v, dF, dfg

        for i in range(n_steps):
            t = i * h
            k1 = rhs(a, F, fg, t)
            k2 = rhs(a + .5 * h * k1[0], F + .5 * h * k1[1], fg + .5 * h * k1[2] if with_g else None, t + .5 * h)
            k3 = rhs(a + .5 * h * k2[0], F + .5 * h * k2[1], fg + .5 * h * k2[2] if with_g else None, t + .5 * h)
            k4 = rhs(a + h * k3[0], F + h * k3[1], fg + h * k3[2] if with_g else None, t + h)
            a = a + h / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
            F = F + h / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
            if with_g:
                fg = fg + h / 6 * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2])
        return a, F, fg

    def save(self, path):
        np.savez(path, **{f"W{i}": W for i, (W, b) in enumerate(self.net.params)},
                 **{f"b{i}": b for i, (W, b) in enumerate(self.net.params)})

    @staticmethod
    def load(path, dtype=np.float64):
        z = np.load(path)
        n = len([k for k in z.files if k.startswith("W")])
        net = MLP([1], np.random.default_rng(0))
        net.params = [[z[f"W{i}"].astype(dtype), z[f"b{i}"].astype(dtype)] for i in range(n)]
        return Policy(net)


def ftle(F):
    return np.log(np.linalg.svd(F, compute_uv=False)[:, 0])


def train(n_ep=3000, steps=20000, batch=1024, width=256, depth=3, lr=1e-3, seed=0):
    rng = np.random.default_rng(seed)
    obs, ch, _ = make_demos(n_ep, rng)
    obs_n = norm_obs(obs).astype(np.float32); ch = ch.astype(np.float32)
    net = MLP([D_IN] + [width] * depth + [A], rng)
    pol = Policy(net)
    t0 = time.time(); ema = None; log = []
    for s in range(steps):
        idx = rng.integers(0, len(ch), batch)
        a1 = ch[idx]; a0 = rng.standard_normal(a1.shape).astype(np.float32)
        tau = rng.random(batch).astype(np.float32)
        at = (1 - tau[:, None]) * a0 + tau[:, None] * a1
        x = np.concatenate([at, obs_n[idx], tau_features(tau).astype(np.float32)], 1)
        pred = net.forward(x, cache=True)
        err = pred - (a1 - a0)
        loss = float((err ** 2).mean())
        grads = net.backward(2 * err / err.size)
        lr_s = lr * 0.5 * (1 + np.cos(np.pi * s / steps))     # cosine decay
        net.adam_step(grads, lr_s)
        ema = loss if ema is None else 0.99 * ema + 0.01 * loss
        if s % 1000 == 0 or s == steps - 1:
            log.append((s, ema)); print(f"step {s:6d}  loss(ema) {ema:.4f}  {time.time() - t0:.0f}s", flush=True)
    return pol, log


if __name__ == "__main__":
    import sys
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
    pol, log = train(steps=steps)
    pol.save("policy.npz")
    np.save("train_log.npy", np.array(log))
