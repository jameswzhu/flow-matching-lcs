"""
Flow-matching policy for the multi-obstacle task (24-D action chunks).

    python policy2.py 50000      # trains, saves policy2.npz
"""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "robot"))
import numpy as np
from mlp import MLP, tau_features
from env2 import make_demos, norm_obs, OBS_STD, H

A = 2 * H
NF = 8
D_IN = A + 3 + NF


class Policy:
    def __init__(self, net):
        self.net = net
        self.dt = net.params[0][0].dtype

    def _inp(self, a, tau, obs_n):
        N = a.shape[0]
        return np.concatenate([a, np.broadcast_to(obs_n, (N, 3)),
                               tau_features(np.full(N, tau))], 1).astype(self.dt)

    def v(self, a, tau, obs_n):
        return self.net.forward(self._inp(a, tau, obs_n))

    def sample(self, a0, obs_n, n_steps=20):
        a = a0.astype(self.dt); h = 1.0 / n_steps
        for i in range(n_steps):
            t = i * h
            k1 = self.v(a, t, obs_n); k2 = self.v(a + .5 * h * k1, t + .5 * h, obs_n)
            k3 = self.v(a + .5 * h * k2, t + .5 * h, obs_n); k4 = self.v(a + h * k3, t + h, obs_n)
            a = a + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        return a

    def _vjac(self, a, tau, obs_n, with_g):
        T = np.zeros((D_IN, A + int(with_g)), self.dt)
        T[:A, :A] = np.eye(A)
        if with_g:
            T[A + 2, A] = 1.0 / OBS_STD[2]
        v, JT = self.net.jvp(self._inp(a, tau, obs_n), T)
        return v, JT[:, :, :A], (JT[:, :, A] if with_g else None)

    def sample_jac(self, a0, obs_n, n_steps=20, with_g=False, record=False):
        """RK4 on a' = v, F' = J F, f_g' = J f_g + J_g.
        With record=True also returns Eulerian diagnostics along the path:
          eul_max  = max_tau sigma_max(J(tau))           (instantaneous stretch rate)
          lognorm  = int_0^1 mu_2(J(tau)) dtau           (log-norm bound: FTLE <= lognorm)"""
        N = a0.shape[0]
        a = a0.astype(self.dt); F = np.broadcast_to(np.eye(A, dtype=self.dt), (N, A, A)).copy()
        fg = np.zeros((N, A), self.dt) if with_g else None
        h = 1.0 / n_steps
        eul = np.full(N, -np.inf); lnorm = np.zeros(N)

        def rhs(a, F, fg, t):
            v, Ja, Jg = self._vjac(a, t, obs_n, with_g)
            return v, Ja @ F, ((np.einsum("nij,nj->ni", Ja, fg) + Jg) if with_g else None), Ja

        for i in range(n_steps):
            t = i * h
            k1 = rhs(a, F, fg, t)
            if record:
                J = k1[3].astype(np.float64)
                eul = np.maximum(eul, np.linalg.norm(J, ord=2, axis=(1, 2)))
                lnorm += h * np.linalg.eigvalsh(0.5 * (J + np.transpose(J, (0, 2, 1))))[:, -1]
            k2 = rhs(a + .5 * h * k1[0], F + .5 * h * k1[1], fg + .5 * h * k1[2] if with_g else None, t + .5 * h)
            k3 = rhs(a + .5 * h * k2[0], F + .5 * h * k2[1], fg + .5 * h * k2[2] if with_g else None, t + .5 * h)
            k4 = rhs(a + h * k3[0], F + h * k3[1], fg + h * k3[2] if with_g else None, t + h)
            a = a + h / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
            F = F + h / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
            if with_g:
                fg = fg + h / 6 * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2])
        out = (a, F, fg)
        return out + ((eul, lnorm) if record else ())

    def save(self, path):
        np.savez(path, **{f"W{i}": W for i, (W, b) in enumerate(self.net.params)},
                 **{f"b{i}": b for i, (W, b) in enumerate(self.net.params)})

    @staticmethod
    def load(path, dtype=np.float32):
        z = np.load(path)
        n = len([k for k in z.files if k.startswith("W")])
        net = MLP([1], np.random.default_rng(0))
        net.params = [[z[f"W{i}"].astype(dtype), z[f"b{i}"].astype(dtype)] for i in range(n)]
        return Policy(net)


def ftle(F):
    return np.log(np.linalg.svd(F.astype(np.float64), compute_uv=False)[:, 0])


def train(n_ep=6000, steps=50000, batch=1024, width=256, depth=4, lr=1e-3, seed=0):
    rng = np.random.default_rng(seed)
    obs, ch, _ = make_demos(n_ep, rng)
    obs_n = norm_obs(obs).astype(np.float32); ch = ch.astype(np.float32)
    net = MLP([D_IN] + [width] * depth + [A], rng)
    t0 = time.time(); ema = None
    for s in range(steps):
        idx = rng.integers(0, len(ch), batch)
        a1 = ch[idx]; a0 = rng.standard_normal(a1.shape).astype(np.float32)
        tau = rng.random(batch).astype(np.float32)
        at = (1 - tau[:, None]) * a0 + tau[:, None] * a1
        x = np.concatenate([at, obs_n[idx], tau_features(tau).astype(np.float32)], 1)
        err = net.forward(x, cache=True) - (a1 - a0)
        loss = float((err ** 2).mean())
        net.adam_step(net.backward(2 * err / err.size), lr * 0.5 * (1 + np.cos(np.pi * s / steps)))
        ema = loss if ema is None else 0.99 * ema + 0.01 * loss
        if s % 2500 == 0 or s == steps - 1:
            print(f"step {s:6d}  loss(ema) {ema:.4f}  {time.time() - t0:.0f}s", flush=True)
    return Policy(net)


if __name__ == "__main__":
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 50000
    train(steps=steps).save("policy2.npz")
