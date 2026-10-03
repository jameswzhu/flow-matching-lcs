"""
Minimal NumPy MLP for a flow-matching velocity field, with:
  * manual backprop + Adam for training,
  * forward-mode Jacobians (tangent propagation) w.r.t. chosen input dims,
    used to integrate the variational equation of the generative flow.

Input layout:  [a (A dims) | obs (3 dims, normalised) | tau features (F dims)]
"""
import numpy as np


def silu(z):
    s = 1.0 / (1.0 + np.exp(-z))
    return z * s, s * (1 + z * (1 - s))      # value, derivative


def tau_features(tau, n=8):
    tau = np.asarray(tau, float).reshape(-1, 1)
    k = np.arange(n // 2)[None, :]
    return np.concatenate([np.sin(np.pi * (2 ** k) * tau), np.cos(np.pi * (2 ** k) * tau)], 1)


class MLP:
    def __init__(self, sizes, rng, dtype=np.float32):
        self.params = []
        for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
            W = rng.standard_normal((a, b)) * np.sqrt(2.0 / a)
            if i == len(sizes) - 2:
                W *= 0.1
            self.params.append([W.astype(dtype), np.zeros(b, dtype)])
        self.m = [[np.zeros_like(W), np.zeros_like(b)] for W, b in self.params]
        self.v = [[np.zeros_like(W), np.zeros_like(b)] for W, b in self.params]
        self.t = 0

    def astype(self, dtype):
        self.params = [[W.astype(dtype), b.astype(dtype)] for W, b in self.params]
        return self

    def forward(self, x, cache=False):
        hs, ds = [x], []
        h = x
        L = len(self.params)
        for i, (W, b) in enumerate(self.params):
            z = h @ W + b
            if i < L - 1:
                h, d = silu(z)
                ds.append(d)
            else:
                h = z
            hs.append(h)
        if cache:
            self._cache = (hs, ds)
        return h

    def backward(self, gout):
        hs, ds = self._cache
        grads = []
        g = gout
        L = len(self.params)
        for i in reversed(range(L)):
            W, _ = self.params[i]
            grads.append([hs[i].T @ g, g.sum(0)])
            if i > 0:
                g = (g @ W.T) * ds[i - 1]
        return grads[::-1]

    def adam_step(self, grads, lr, b1=0.9, b2=0.999, eps=1e-8, wd=0.0):
        self.t += 1
        for p, m, v, gr in zip(self.params, self.m, self.v, grads):
            for j in range(2):
                m[j] = b1 * m[j] + (1 - b1) * gr[j]
                v[j] = b2 * v[j] + (1 - b2) * gr[j] ** 2
                mh = m[j] / (1 - b1 ** self.t); vh = v[j] / (1 - b2 ** self.t)
                p[j] -= lr * (mh / (np.sqrt(vh) + eps) + (wd * p[j] if j == 0 else 0))

    def jvp(self, x, T):
        """Output and input-Jacobian applied to tangents.
        x: [N, d_in]; T: [d_in, k] (same tangents for every sample).
        Returns y [N, d_out] and dy/dx @ T as [N, d_out, k]."""
        h = x
        Th = None                                 # tangents, layout [N, k, width]
        L = len(self.params)
        for i, (W, b) in enumerate(self.params):
            z = h @ W + b
            Tz = (T.T @ W)[None] if Th is None else Th @ W
            if i < L - 1:
                h, d = silu(z)
                Th = Tz * d[:, None, :]
            else:
                h, Th = z, np.broadcast_to(Tz, (x.shape[0],) + Tz.shape[1:])
        return h, np.transpose(Th, (0, 2, 1))
