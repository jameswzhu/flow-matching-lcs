"""
Language-conditioned flow-matching policy (PyTorch), 24-D action chunks.

    v_theta(a, tau | obs, e)    obs = 16 object features, e = frozen prompt embedding

The prompt embedding is standardised with the mean / RMS of the *training* prompts'
vectors (CLIP's prompt vectors all have cosine ~0.8 with each other), and the same
affine map is applied to unseen prompts.

    python policy3.py <encoder> [holdout] [steps]   # e.g. python policy3.py clip
        -> policies/<encoder>[_holdout].pt
    python policy3.py check <ckpt>                  # Jacobians vs finite differences

Device: CUDA if available, else CPU (override with TORCH_DEVICE=cpu|cuda). Precision:
float32 on GPU (consumer GPUs run float64 at 1/32 rate), float64 on CPU (override with
TORCH_DTYPE=float32|float64). Training is always float32; `check` always runs on CPU
in float64.
"""
import os, sys, time
import numpy as np
import torch
from torch import nn
from torch.func import jvp, jacfwd, vmap
from env3 import make_demos, train_prompts, H, D_OBS

A = 2 * H
NF = 8
torch.set_num_threads(int(os.environ.get("TORCH_THREADS", max(1, os.cpu_count() // 2))))
DEVICE = torch.device(os.environ.get("TORCH_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
DTYPE = getattr(torch, os.environ.get("TORCH_DTYPE", "float32" if DEVICE.type == "cuda" else "float64"))


def tau_features(tau, n=NF):
    k = torch.arange(n // 2, dtype=tau.dtype, device=tau.device)
    x = np.pi * (2.0 ** k) * tau[..., None]
    return torch.cat([torch.sin(x), torch.cos(x)], -1)


def load_embeddings(enc):
    z = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "emb", f"{enc}.npz"))
    return dict(zip(z["prompts"].tolist(), z["vectors"]))


def mlp(d_in, width, d_out, n_hidden=2):
    layers, d = [], d_in
    for _ in range(n_hidden):
        layers += [nn.Linear(d, width), nn.SiLU()]; d = width
    return nn.Sequential(*layers, nn.Linear(d, d_out))


class CrossAttn(nn.Module):
    """One query vs a few object tokens, written out explicitly: torch's fused CPU
    attention kernel has no forward-mode AD (needed for jacfwd / jvp)."""

    def __init__(self, width, n_heads):
        super().__init__()
        self.h, self.dh = n_heads, width // n_heads
        self.q, self.kv, self.o = nn.Linear(width, width), nn.Linear(width, 2 * width), nn.Linear(width, width)

    def forward(self, x, tok):
        """x [..., W], tok [..., K, W] -> [..., W]."""
        q = self.q(x).unflatten(-1, (self.h, self.dh))                              # [..., h, dh]
        k, v = self.kv(tok).unflatten(-1, (2, self.h, self.dh)).unbind(-3)          # [..., K, h, dh]
        w = torch.softmax(torch.einsum("...hd,...khd->...hk", q, k) / self.dh ** 0.5, -1)
        return self.o(torch.einsum("...hk,...khd->...hd", w, v).flatten(-2))


class VelocityNet(nn.Module):
    """Object-attention velocity field. A plain MLP on the shuffled concatenation of
    slots fails to learn the attribute -> slot binding (first oracle run: 30% correct
    object for 1-match prompts, chance 25%), so:

      tokens_i = f([rel_i, colour_i, shape_i, e])        one per object (shared f)
      h        = g([a, tau feats, e])
      repeat L times:  h += CrossAttn(query = h, keys/values = tokens);  h += MLP(h)
      v        = out(h)

    Permutation-invariant in the object slots by construction; the noise a enters the
    query, so it can select among several matching objects."""

    def __init__(self, d_emb, width=128, n_layers=3, n_heads=4):
        super().__init__()
        self.d_emb = d_emb
        self.tok = mlp(4 + d_emb, width, width)
        self.inp = mlp(A + NF + d_emb, width, width)
        self.attn = nn.ModuleList([CrossAttn(width, n_heads) for _ in range(n_layers)])
        self.ln_q = nn.ModuleList([nn.LayerNorm(width) for _ in range(n_layers)])
        self.ln_m = nn.ModuleList([nn.LayerNorm(width) for _ in range(n_layers)])
        self.ff = nn.ModuleList([mlp(width, width, width, 1) for _ in range(n_layers)])
        self.out = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, A))
        with torch.no_grad():
            self.out[1].weight.mul_(0.1); self.out[1].bias.zero_()

    def tokens(self, cond):
        """Object tokens [N, 4, W]; independent of (a, tau), so computed once per flow."""
        objs = cond[..., :D_OBS].reshape(*cond.shape[:-1], D_OBS // 4, 4)
        e = cond[..., D_OBS:]
        return self.tok(torch.cat([objs, e[..., None, :].expand(*objs.shape[:-1], self.d_emb)], -1))

    def forward(self, a, tau, cond, tok=None):
        """a [N, A], tau [N] or scalar, cond [N, D_OBS + d_emb] (embedding already standardised)."""
        tau = torch.as_tensor(tau, dtype=a.dtype, device=a.device).expand(a.shape[:-1])
        e = cond[..., D_OBS:]
        tok = self.tokens(cond) if tok is None else tok
        h = self.inp(torch.cat([a, tau_features(tau), e], -1))                         # [N, W]
        for attn, lq, lm, ff in zip(self.attn, self.ln_q, self.ln_m, self.ff):
            h = h + attn(lq(h), tok)
            h = h + ff(lm(h))
        return self.out(h)


class Policy:
    def __init__(self, net, enc, emb_mu, emb_s, holdout=False):
        self.net, self.enc, self.holdout = net.eval(), enc, holdout
        p0 = next(net.parameters()); self.dev, self.dt = p0.device, p0.dtype
        self.emb_mu, self.emb_s = np.asarray(emb_mu), float(emb_s)
        self.table = load_embeddings(enc)

    # ---------------------------------------------------------------- conditioning
    def emb(self, prompt):
        """Standardised embedding of a prompt (string) or a raw vector."""
        e = self.table[prompt] if isinstance(prompt, str) else np.asarray(prompt)
        return (e - self.emb_mu) / self.emb_s

    def cond(self, obs, prompt):
        """obs [N, 16], prompt str / raw vector / [N, E] raw vectors -> [N, 16 + E] tensor."""
        e = self.emb(prompt)
        e = np.broadcast_to(e, (len(obs), e.shape[-1]))
        return self._t(np.concatenate([obs, e], 1))

    def _t(self, x):
        return torch.as_tensor(np.asarray(x), dtype=self.dt, device=self.dev)

    @staticmethod
    def _np(x):
        return x.cpu().numpy().astype(np.float64)

    # ---------------------------------------------------------------- flow map
    def flow(self, z, cond, n_steps=20):
        """RK4 integration of da/dtau = v from tau = 0 to 1. Pure torch (differentiable,
        vmap-able); z [..., A], cond [..., C]."""
        h = 1.0 / n_steps
        a = z
        tok = self.net.tokens(cond)
        v = lambda a, t: self.net(a, t, cond, tok)
        for i in range(n_steps):
            t = i * h
            k1 = v(a, t); k2 = v(a + .5 * h * k1, t + .5 * h)
            k3 = v(a + .5 * h * k2, t + .5 * h); k4 = v(a + h * k3, t + h)
            a = a + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        return a

    @torch.no_grad()
    def sample(self, z, obs, prompt, n_steps=20, bs=20000):
        z = self._t(z)
        c = self.cond(obs, prompt)
        return self._np(torch.cat([self.flow(z[i:i + bs], c[i:i + bs], n_steps)
                                   for i in range(0, len(z), bs)]))

    @torch.no_grad()
    def jac_z(self, z, obs, prompt, n_steps=20, bs=2000):
        """Chunk Phi(z) [N, A] and flow-map Jacobian dPhi/dz [N, A, A] (forward mode)."""
        z = self._t(z)
        c = self.cond(obs, prompt)
        f = lambda z1, c1: self.flow(z1[None], c1[None], n_steps)[0]
        out_a, out_F = [], []
        for i in range(0, len(z), bs):
            F = vmap(jacfwd(f, argnums=0))(z[i:i + bs], c[i:i + bs])
            out_F.append(self._np(F)); out_a.append(self._np(self.flow(z[i:i + bs], c[i:i + bs], n_steps)))
        return np.concatenate(out_a), np.concatenate(out_F)

    @torch.no_grad()
    def jvp_cond(self, z, obs, prompt, dcond, n_steps=20):
        """Directional derivative of Phi along a conditioning direction dcond [N, C]
        (e.g. a direction in standardised prompt-embedding space, zero-padded on obs)."""
        z = self._t(z)
        c = self.cond(obs, prompt)
        a, da = jvp(lambda c_: self.flow(z, c_, n_steps), (c,), (self._t(dcond),))
        return self._np(a), self._np(da)

    # ---------------------------------------------------------------- io
    def save(self, path):
        torch.save(dict(state=self.net.state_dict(), enc=self.enc, emb_mu=self.emb_mu,
                        emb_s=self.emb_s, holdout=self.holdout, d_emb=len(self.emb_mu),
                        width=self.net.out[1].in_features), path)

    @staticmethod
    def load(path, device=None, dtype=None):
        ck = torch.load(path, weights_only=False, map_location="cpu")
        net = VelocityNet(ck["d_emb"], ck["width"])
        net.load_state_dict(ck["state"])
        net.to(device=device or DEVICE, dtype=dtype or DTYPE)
        return Policy(net, ck["enc"], ck["emb_mu"], ck["emb_s"], ck["holdout"])


def ftle(F):
    return np.log(np.linalg.svd(F, compute_uv=False)[:, 0])


def train(enc, holdout=False, n_ep=30000, steps=20000, batch=1024, lr=1e-3, seed=0):
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    prompts = train_prompts(holdout)
    table = load_embeddings(enc)
    E = np.stack([table[p] for p in prompts])
    mu = E.mean(0); s = float(np.sqrt(((E - mu) ** 2).sum(1).mean()))
    dev = DEVICE
    En = torch.tensor((E - mu) / s, dtype=torch.float32, device=dev)
    obs, pid, ch, _ = make_demos(n_ep, rng, prompts)
    print(f"{enc}{' holdout' if holdout else ''}: {len(ch)} samples, {len(prompts)} prompts, "
          f"emb dim {E.shape[1]}, device {dev}", flush=True)
    obs = torch.tensor(obs, dtype=torch.float32, device=dev); ch = torch.tensor(ch, dtype=torch.float32, device=dev)
    pid = torch.tensor(pid, device=dev)
    net = VelocityNet(E.shape[1]).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    t0 = time.time(); ema = None
    for s_ in range(steps):
        idx = torch.randint(len(ch), (batch,), device=dev)
        a1 = ch[idx]; a0 = torch.randn_like(a1); tau = torch.rand(batch, device=dev)
        at = (1 - tau[:, None]) * a0 + tau[:, None] * a1
        loss = ((net(at, tau, torch.cat([obs[idx], En[pid[idx]]], 1)) - (a1 - a0)) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        ema = loss.detach() if ema is None else 0.99 * ema + 0.01 * loss.detach()   # no per-step GPU sync
        if s_ % 2500 == 0 or s_ == steps - 1:
            print(f"step {s_:6d}  loss(ema) {ema.item():.4f}  {time.time() - t0:.0f}s", flush=True)
    return Policy(net, enc, mu, s, holdout)


def check(path, n=6, eps=1e-6):
    """Flow-map Jacobians (noise, prompt embedding) vs central finite differences."""
    from env3 import make_scenes, obs_features
    pol = Policy.load(path, "cpu", torch.float64)
    rng = np.random.default_rng(0)
    objs, _ = make_scenes(n, rng)
    obs = obs_features(np.zeros((n, 2)), objs)
    z = rng.standard_normal((n, A))
    a, F = pol.jac_z(z, obs, "square")
    Ffd = np.stack([(pol.sample(z + eps * np.eye(A)[j], obs, "square") -
                     pol.sample(z - eps * np.eye(A)[j], obs, "square")) / (2 * eps) for j in range(A)], 2)
    print("dPhi/dz   max rel err vs FD:", np.abs(F - Ffd).max() / np.abs(F).max())
    e0 = pol.table["square"]; de = rng.standard_normal(e0.shape); de /= np.linalg.norm(de)
    dcond = np.concatenate([np.zeros((n, D_OBS)), np.broadcast_to(de / pol.emb_s, (n, len(de)))], 1)
    _, da = pol.jvp_cond(z, obs, "square", dcond)
    dfd = (pol.sample(z, obs, e0 + eps * de) - pol.sample(z, obs, e0 - eps * de)) / (2 * eps)
    print("dPhi/de.d max rel err vs FD:", np.abs(da - dfd).max() / np.abs(da).max())


if __name__ == "__main__":
    if sys.argv[1] == "check":
        check(sys.argv[2])
    else:
        enc = sys.argv[1]
        holdout = "holdout" in sys.argv[2:]
        nums = [int(x) for x in sys.argv[2:] if x.isdigit()]
        os.makedirs("policies", exist_ok=True)
        train(enc, holdout, steps=nums[0] if nums else 20000).save(
            f"policies/{enc}{'_holdout' if holdout else ''}.pt")
