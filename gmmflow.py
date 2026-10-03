"""
Closed-form conditional flow matching for a Gaussian-mixture target, plus
flow-map Jacobians and FTLE fields.

Path (rectified flow / linear interpolation):
    x_tau = (1 - tau) x0 + tau x1,   x0 ~ N(0, I),   x1 ~ sum_k w_k N(mu_k, sig_k^2 I)

Conditional on component k, x_tau ~ N(tau mu_k, s_k(tau)^2 I) with
    s_k(tau)^2 = (1 - tau)^2 + tau^2 sig_k^2,
and the component's own Gaussian path has velocity
    v_k(x, tau) = mu_k + (s_k'/s_k) (x - tau mu_k),   s_k' = (tau sig_k^2 - (1 - tau)) / s_k.
The exact marginal (= optimal flow matching) velocity is the posterior average
    v(x, tau) = sum_k r_k(x, tau) v_k(x, tau),   r_k ∝ w_k N(x; tau mu_k, s_k^2 I).
This form is non-singular at tau = 1 (unlike (E[x1|x] - x)/(1 - tau)).

Conditioning c enters only through (w_k(c), mu_k(c), sig_k(c)), supplied by the
caller as a function `params(c) -> (w[K], mu[K,d], sig[K])`.
"""
import numpy as np


# ----------------------------------------------------------------------------
# velocity field and its spatial Jacobian
# ----------------------------------------------------------------------------
def _sched(tau, sig):
    s2 = (1 - tau) ** 2 + tau ** 2 * sig ** 2          # [K]
    s = np.sqrt(s2)
    ds = (tau * sig ** 2 - (1 - tau)) / s               # [K]
    return s2, s, ds


def velocity(x, tau, w, mu, sig, jac=False):
    """x: [N, d]. Returns v [N, d] and optionally J = dv/dx [N, d, d]."""
    s2, s, ds = _sched(tau, sig)
    a = ds / s                                          # [K] per-component rate
    diff = x[:, None, :] - tau * mu[None, :, :]         # [N, K, d]
    d = x.shape[1]
    logp = (np.log(w)[None, :]
            - 0.5 * np.sum(diff ** 2, -1) / s2[None, :]
            - 0.5 * d * np.log(s2)[None, :])            # [N, K]
    logp -= logp.max(1, keepdims=True)
    r = np.exp(logp)
    r /= r.sum(1, keepdims=True)                        # [N, K]
    vk = mu[None, :, :] + a[None, :, None] * diff       # [N, K, d]
    v = np.einsum("nk,nkd->nd", r, vk)
    if not jac:
        return v
    g = -diff / s2[None, :, None]                       # grad log N_k  [N, K, d]
    gbar = np.einsum("nk,nkd->nd", r, g)
    # J = sum_k r_k a_k I + sum_k r_k v_k (g_k - gbar)^T
    J = np.einsum("nk,k->n", r, a)[:, None, None] * np.eye(d)[None]
    J = J + np.einsum("nk,nki,nkj->nij", r, vk, g - gbar[:, None, :])
    return v, J


# ----------------------------------------------------------------------------
# integration of the flow and the variational equation
# ----------------------------------------------------------------------------
def flow(x0, params_fn, c, n_steps=200, method="rk4", jac=False, tau_end=1.0,
         return_path=False):
    """Integrate dx/dtau = v(x, tau; c) from 0 to tau_end.

    c may be a scalar (shared) or an array [N] (one conditioning per particle).
    If jac, also integrates dF/dtau = J F to return the flow-map Jacobian.
    method: 'rk4' (reference) or 'euler' (what deployed policies often use).
    """
    x = np.array(x0, float)
    N, d = x.shape
    cs = np.broadcast_to(np.asarray(c, float), (N,))
    uniq = np.unique(cs)
    groups = [(cv, np.where(cs == cv)[0]) for cv in uniq]
    P = {cv: params_fn(cv) for cv in uniq}

    def f(xx, tau):
        v = np.empty_like(xx)
        J = np.empty((N, d, d)) if jac else None
        for cv, idx in groups:
            w, mu, sig = P[cv]
            if jac:
                v[idx], J[idx] = velocity(xx[idx], tau, w, mu, sig, jac=True)
            else:
                v[idx] = velocity(xx[idx], tau, w, mu, sig)
        return v, J

    F = np.broadcast_to(np.eye(d), (N, d, d)).copy() if jac else None
    h = tau_end / n_steps
    path = [x.copy()] if return_path else None
    for i in range(n_steps):
        t = i * h
        if method == "euler":
            v, J = f(x, t)
            if jac:
                F = F + h * J @ F
            x = x + h * v
        else:
            k1, J1 = f(x, t)
            k2, J2 = f(x + 0.5 * h * k1, t + 0.5 * h)
            k3, J3 = f(x + 0.5 * h * k2, t + 0.5 * h)
            k4, J4 = f(x + h * k3, t + h)
            if jac:
                # RK4 on the variational equation (J evaluated along the stages)
                G1 = J1 @ F
                G2 = J2 @ (F + 0.5 * h * G1)
                G3 = J3 @ (F + 0.5 * h * G2)
                G4 = J4 @ (F + h * G3)
                F = F + h / 6 * (G1 + 2 * G2 + 2 * G3 + G4)
            x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        if return_path:
            path.append(x.copy())
    out = [x]
    if jac:
        out.append(F)
    if return_path:
        out.append(np.stack(path, 1))
    return out[0] if len(out) == 1 else tuple(out)


def ftle_from_jac(F, T=1.0):
    """Largest finite-time Lyapunov exponent from flow-map Jacobians [N, d, d]."""
    sv = np.linalg.svd(F, compute_uv=False)[:, 0]
    return np.log(sv) / T


def ftle_fd(x0, params_fn, c, delta=1e-4, **kw):
    """FTLE via central finite differences (independent check of the
    variational-equation Jacobian)."""
    N, d = x0.shape
    F = np.empty((N, d, d))
    for j in range(d):
        e = np.zeros(d); e[j] = delta
        xp = flow(x0 + e, params_fn, c, **kw)
        xm = flow(x0 - e, params_fn, c, **kw)
        F[:, :, j] = (xp - xm) / (2 * delta)
    return ftle_from_jac(F), F
