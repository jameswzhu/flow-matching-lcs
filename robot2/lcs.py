"""
Lagrangian coherent structure (LCS) of the one-chunk generative flow, located from
each noise seed, and the margins it implies.

Repelling LCS theory: material surfaces that maximize repulsion are normal to the
dominant eigenvector xi_max of the Cauchy-Green tensor F^T F, i.e. the top right
singular vector v1 of the flow-map Jacobian F = dPhi/dz. From a seed z we therefore
search along the line z + s v1 and find

  * the RIDGE (label-free LCS): where the stretching along the line, |dPhi(z+s v1)/ds|,
    is maximal;
  * the SEPARATRIX (ground truth): where the chunk's pending-gap label changes.

The signed distances d+ and d- to the nearest LCS on each side are the margins.
"""
import numpy as np
from scipy.stats import beta, norm
from policy2 import A
from multichunk import jvp_rows
from mlp import tau_features
from env2 import norm_obs, chunk_pending_label


def flow_dir(pol, z, v, pos, g, n_steps=20):
    """Chunk Phi(z) and the directional derivative dPhi(z + s v)/ds at s=0 (forward mode)."""
    N = len(z); dt = pol.dt
    obs_n = norm_obs(np.column_stack([pos, g])).astype(dt)
    a = z.astype(dt); dl = v.astype(dt)[:, None, :]
    zpad = np.zeros((N, 1, 3 + 8), dt)
    h = 1.0 / n_steps

    def rhs(a, dl, t):
        x = np.concatenate([a, obs_n, tau_features(np.full(N, t))], 1).astype(dt)
        return jvp_rows(pol.net, x, np.concatenate([dl, zpad], 2))

    for i in range(n_steps):
        t = i * h
        k1 = rhs(a, dl, t); k2 = rhs(a + .5 * h * k1[0], dl + .5 * h * k1[1], t + .5 * h)
        k3 = rhs(a + .5 * h * k2[0], dl + .5 * h * k2[1], t + .5 * h); k4 = rhs(a + h * k3[0], dl + h * k3[1], t + h)
        a = a + h / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        dl = dl + h / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
    return a.astype(float), dl[:, 0].astype(float)


def labels(pol, z, pos, g, bs=10000):
    out = []
    for i in range(0, len(z), bs):
        ch = pol.sample(z[i:i + bs].astype(np.float32), norm_obs(np.column_stack([pos[i:i + bs], g[i:i + bs]])), 20)
        out.append(chunk_pending_label(pos[i:i + bs], ch.astype(float))[1])
    return np.concatenate(out)


def scan(pol, z, n, pos, g, s_grid, bs=3000, log=None):
    """Labels and directional stretch along z + s n for every s in s_grid.
    Returns lab [N, S], stretch [N, S] (= log |dPhi/ds|)."""
    N, S = len(z), len(s_grid)
    lab = np.empty((N, S), int); st = np.empty((N, S))
    for j, s in enumerate(s_grid):
        zz = z + s * n
        a1 = []; dd = []
        for i in range(0, N, bs):
            a_, d_ = flow_dir(pol, zz[i:i + bs], n[i:i + bs], pos[i:i + bs], g[i:i + bs])
            a1.append(a_); dd.append(d_)
        a1 = np.concatenate(a1); dd = np.concatenate(dd)
        lab[:, j] = np.concatenate([chunk_pending_label(pos[i:i + 10000], a1[i:i + 10000])[1]
                                    for i in range(0, N, 10000)])
        st[:, j] = np.log(np.linalg.norm(dd, axis=1) + 1e-12)
        if log and j % 10 == 0:
            log(f"   scan {j + 1}/{S}")
    return lab, st


def crossings(lab, s_grid, i0):
    """Nearest label change on each side of s_grid[i0] (grid resolution).
    Returns (lo, hi) brackets for the + side and - side (nan if none)."""
    N = len(lab)
    l0 = lab[:, i0]
    pp = np.full((N, 2), np.nan); mm = np.full((N, 2), np.nan)
    diff = lab != l0[:, None]
    for n in range(N):
        r = np.where(diff[n, i0 + 1:])[0]
        if len(r):
            j = i0 + 1 + r[0]; pp[n] = s_grid[j - 1], s_grid[j]
        l_ = np.where(diff[n, :i0][::-1])[0]
        if len(l_):
            j = i0 - 1 - l_[0]; mm[n] = s_grid[j + 1], s_grid[j]
    return pp, mm


def bisect(pol, z, n, pos, g, bracket, l0, iters=10):
    """Refine a label change inside bracket [inside, outside] along z + s n."""
    lo, hi = bracket[:, 0].copy(), bracket[:, 1].copy()
    ok = ~np.isnan(lo)
    if not ok.any():
        return lo
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        lm = labels(pol, z[ok] + mid[ok, None] * n[ok], pos[ok], g[ok])
        same = lm == l0[ok]
        lo[ok] = np.where(same, mid[ok], lo[ok]); hi[ok] = np.where(same, hi[ok], mid[ok])
    out = 0.5 * (lo + hi); out[~ok] = np.nan
    return out


def ridges(st, s_grid, i0):
    """Label-free LCS: position of the stretch maximum on each side (parabola-refined)."""
    N, S = st.shape
    res = np.full((N, 2), np.nan)
    ds = s_grid[1] - s_grid[0]
    for side, rng_ in [(0, range(i0, S)), (1, range(0, i0 + 1))]:
        idx = np.array(list(rng_))
        j = idx[np.argmax(st[:, idx], 1)]
        j = np.clip(j, 1, S - 2)
        y0, y1, y2 = st[np.arange(N), j - 1], st[np.arange(N), j], st[np.arange(N), j + 1]
        den = y0 - 2 * y1 + y2
        off = np.where(np.abs(den) > 1e-12, 0.5 * (y0 - y2) / np.where(den == 0, 1, den), 0)
        res[:, side] = s_grid[j] + np.clip(off, -1, 1) * ds
    return res


# ------------------------------------------------------------------ analytic predictions
D_NOISE = A


def p_cross(margin, eps, D=D_NOISE):
    """P(eps * (u . n) > margin) for u uniform on the unit sphere in R^D."""
    m = np.asarray(margin, float)
    a = np.clip(m / eps, 0, 1)
    p = 0.5 * beta.sf(a ** 2, 0.5, (D - 1) / 2)
    return np.where(np.isnan(m), 0.0, np.where(m <= 0, 0.5, p))


def p_flip_noise(d_plus, d_minus, eps, n_dirs=4, D=D_NOISE):
    """Flip probability under n_dirs random nudges of size eps, given margins to the LCS
    on each side along its normal (locally flat LCS)."""
    p1 = p_cross(d_plus, eps, D) + p_cross(d_minus, eps, D)
    return 1 - (1 - np.clip(p1, 0, 1)) ** n_dirs


def mass_beyond(d_signed_from_origin):
    """Gaussian mass of N(0, I) beyond a hyperplane whose signed offset from the origin
    along its normal is d (mass on the far side, i.e. n . z > d)."""
    return norm.sf(d_signed_from_origin)
