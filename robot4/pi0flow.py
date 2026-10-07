"""
pi0 as a differentiable flow map, for FTLE / LCS analysis.

pi0 (PaliGemma 3B + a 300M Gemma "action expert") samples an action chunk in two stages:

  1. prefix: SigLIP image tokens + prompt tokens go through PaliGemma once, leaving a
     per-layer key/value cache. This is the *only* place the image and the language
     enter, so the cache is the conditioning c.
  2. flow: the expert integrates da/dt = v(a, t | c, state) from t = 1 (noise) to t = 0
     (actions) with 10 forward-Euler steps (openpi convention: noise at t = 1).

Phi(z; c) = the chunk after step 10, with z in R^{H x 32} (pi0 pads actions to 32 dims;
only the first 2 are the task's). FTLE = log sigma_max(dPhi_task / dz).

The expert step is written out here with explicit attention against the cached K/V
(lerobot's version goes through transformers' DynamicCache, which torch.func cannot
trace). The state token cannot attend to the action tokens (suffix mask [1, 1, 0, ...]),
so its per-layer K/V are computed once per flow and appended to the prefix cache; each
velocity call then only processes the H action tokens. `check` verifies that this
matches lerobot's `sample_actions`, and that the Jacobians match finite differences.

    python pi0flow.py check                       # explicit expert vs lerobot; dPhi/dz vs FD
    python pi0flow.py train [holdout] [steps]     # fine-tune the expert on the shapes task
                                                  #   -> policies/pi0[_holdout].pt
    python pi0flow.py check policies/pi0.pt       # same checks on the fine-tuned policy

Precision: the frozen VLM runs in bfloat16 (as in training); the expert runs in float32
(TORCH_DTYPE=float64 for derivative checks). TF32 stays off.
"""
import math, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F
from torch.func import jacrev, vmap

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from env4 import H, ALL_PROMPTS, render, state_vec, instruction, train_prompts, make_demos  # noqa: E402

BASE = "lerobot/pi0_base"
TOKENIZER = os.path.join(HERE, "assets", "paligemma_tokenizer.model")   # big_vision's public copy
MAX_TOK = 48
D_ACT = 32                       # pi0's padded action / state width
A_TASK = 2                       # real action dims per step
N_STEPS = 10
MASK_NEG = -2.3819763e38         # openpi's additive mask value
DEVICE = torch.device(os.environ.get("TORCH_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
DTYPE = getattr(torch, os.environ.get("TORCH_DTYPE", "float32"))


def up(dt):
    return torch.promote_types(dt, torch.float32)


def rms_norm(x, w, eps=1e-6):
    """Gemma RMSNorm (scale 1 + w), computed in at least float32 as in lerobot's PiGemmaRMSNorm."""
    xf = x.to(up(x.dtype))
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * (1.0 + w.to(xf.dtype))).type_as(x)


def rope(x, cos, sin):
    """x [..., T, d], cos/sin [..., T, d] (HF rotate_half convention)."""
    x1, x2 = x.chunk(2, -1)
    return x * cos + torch.cat([-x2, x1], -1) * sin


def cmap(c, fn):
    """Apply fn to every tensor of a conditioning dict."""
    return {k: ([fn(u) for u in v] if isinstance(v, list) else tuple(fn(u) for u in v) if isinstance(v, tuple)
                else fn(v)) for k, v in c.items()}


def cbatch(c):
    return c["mask"].shape[0]


def cslice(c, i, j):
    return cmap(c, lambda u: u[i:j])


def cexpand(c, n):
    return cmap(c, lambda u: u.expand(n, *u.shape[1:]))


class Pi0Flow:
    def __init__(self, policy, vlm_dtype=torch.float32, dtype=DTYPE, device=DEVICE):
        import sentencepiece
        self.pol = policy
        self.m = policy.model                                      # PI0Pytorch
        self.dev, self.dt = torch.device(device), dtype
        pwe = self.m.paligemma_with_expert
        pwe.paligemma.to(device=self.dev, dtype=vlm_dtype)
        for mod in (pwe.paligemma.model.vision_tower, pwe.paligemma.model.multi_modal_projector):
            mod.to(torch.float32)                                  # lerobot feeds SigLIP float32
        for mod in self.expert_modules():
            mod.to(device=self.dev, dtype=dtype)
        self.layers = pwe.gemma_expert.model.layers
        self.final_norm = pwe.gemma_expert.model.norm
        cfg = pwe.gemma_expert.config
        self.n_heads, self.hd = cfg.num_attention_heads, cfg.head_dim
        self.inv_freq = 1.0 / (cfg.rope_parameters["rope_theta"] **
                               (torch.arange(0, self.hd, 2, dtype=torch.float64, device=self.dev) / self.hd))
        self.sp = sentencepiece.SentencePieceProcessor(model_file=TOKENIZER)
        self.m.eval()
        self.m.requires_grad_(False)                               # train() re-enables the expert

    def expert_modules(self):
        m = self.m
        return [m.paligemma_with_expert.gemma_expert, m.action_in_proj, m.action_out_proj, m.state_proj,
                m.action_time_mlp_in, m.action_time_mlp_out]

    def expert_parameters(self):
        return [p for mod in self.expert_modules() for p in mod.parameters()]

    # ---------------------------------------------------------------- prefix (conditioning)
    def tokenize(self, texts):
        """openpi's PaligemmaTokenizer: BOS + text + '\\n', right-padded to MAX_TOK."""
        toks = np.zeros((len(texts), MAX_TOK), np.int64); mask = np.zeros((len(texts), MAX_TOK), bool)
        for i, t in enumerate(texts):
            ids = self.sp.encode(t.strip().replace("_", " ").replace("\n", " "), add_bos=True) + self.sp.encode("\n")
            ids = ids[:MAX_TOK]
            toks[i, :len(ids)] = ids; mask[i, :len(ids)] = True
        return (torch.as_tensor(toks, device=self.dev), torch.as_tensor(mask, device=self.dev))

    def _rope_tables(self, pos):
        """pos [N, T] (int) -> cos, sin [N, 1, T, hd] in the expert dtype."""
        f = pos.to(torch.float64)[..., None] * self.inv_freq
        emb = torch.cat([f, f], -1)[:, None]
        return emb.cos().to(self.dt), emb.sin().to(self.dt)

    def prefix(self, images, texts, state):
        """images [N, 3, 224, 224] in [0, 1]; texts: list of N instructions; state [N, 32].
        Returns the conditioning: per-layer K/V of (prefix + state token) and their mask."""
        from lerobot.policies.common.vla_utils import make_att_2d_masks, prepare_attention_masks_4d
        m, pwe = self.m, self.m.paligemma_with_expert
        tok, tmask = self.tokenize(texts)
        img = images.to(self.dev, torch.float32) * 2 - 1
        imask = torch.ones(len(img), dtype=torch.bool, device=self.dev)
        with torch.no_grad():
            embs, pad, att = m.embed_prefix([img], [imask], tok, tmask)
            att2 = make_att_2d_masks(pad, att)
            pos = torch.cumsum(pad, 1) - 1
            pwe.paligemma.model.language_model.config._attn_implementation = "eager"
            _, kv = pwe.forward(attention_mask=prepare_attention_masks_4d(att2), position_ids=pos,
                                past_key_values=None, inputs_embeds=[embs.to(pwe.paligemma.dtype), None],
                                use_cache=True)
        K = [k[:, 0].to(self.dt) for k, v, *_ in kv]               # 1 kv head: [N, P, hd]
        V = [v[:, 0].to(self.dt) for k, v, *_ in kv]
        offset = pad.sum(1)                                        # number of valid prefix tokens
        c = dict(K=K, V=V, mask=pad, offset=offset)
        # state token: attends to the prefix and itself only -> run it through the layers once
        state = torch.as_tensor(state, device=self.dev, dtype=self.dt)
        cs = self._rope_tables(offset[:, None])
        h = m.state_proj(state)[:, None]                           # [N, 1, W]
        Ks, Vs = [], []
        for li, layer in enumerate(self.layers):
            h, k, v = self._layer(layer, h, cs, K[li], V[li], pad, li)
            Ks.append(k); Vs.append(v)
        c["K"] = [torch.cat([a, b], 1) for a, b in zip(K, Ks)]
        c["V"] = [torch.cat([a, b], 1) for a, b in zip(V, Vs)]
        c["mask"] = torch.cat([pad, torch.ones_like(pad[:, :1])], 1)
        c["rope"] = self._rope_tables(offset[:, None] + 1 + torch.arange(H, device=self.dev))
        c["state"] = state
        return c

    # ---------------------------------------------------------------- expert
    def _layer(self, layer, h, cs, Kc, Vc, cmask, li):
        """One expert layer for query tokens h [..., T, W] against context K/V [..., P, hd]
        (mask [..., P]) plus the tokens' own K/V. Returns new h and the tokens' (k, v)."""
        at = layer.self_attn
        x = rms_norm(h, layer.input_layernorm.weight)
        q = at.q_proj(x).unflatten(-1, (self.n_heads, self.hd)).transpose(-3, -2)    # [..., nh, T, hd]
        k = at.k_proj(x); v = at.v_proj(x)                                           # [..., T, hd]
        cos, sin = cs
        q = rope(q, cos, sin); k = rope(k, cos[..., 0, :, :], sin[..., 0, :, :])
        # context (prefix [+ state]) and the tokens themselves, scored separately so the
        # context can broadcast over extra leading dims of h (several draws per prefix)
        s_c = torch.einsum("...htd,...pd->...htp", q, Kc) * at.scaling
        s_c = s_c + torch.where(cmask, 0.0, MASK_NEG).to(s_c.dtype)[..., None, None, :]
        s_s = torch.einsum("...htd,...sd->...hts", q, k) * at.scaling
        w = torch.softmax(torch.cat([s_c, s_s], -1).to(up(q.dtype)), -1).to(q.dtype)
        P = Kc.shape[-2]
        o = (torch.einsum("...htp,...pd->...thd", w[..., :P], Vc) +
             torch.einsum("...hts,...sd->...thd", w[..., P:], v)).flatten(-2)
        h = h + at.o_proj(o)
        h = h + layer.mlp(rms_norm(h, layer.post_attention_layernorm.weight))
        return h, k, v

    def velocity(self, x, t, c):
        """x [N, H, 32], t [N] -> v [N, H, 32]. Pure torch: jvp / vjp / vmap-able."""
        from lerobot.policies.common.vla_utils import create_sinusoidal_pos_embedding
        m = self.m
        temb = create_sinusoidal_pos_embedding(t, m.action_in_proj.out_features, m.config.min_period,
                                               m.config.max_period, device=t.device).to(x.dtype)
        a = m.action_in_proj(x)
        h = m.action_time_mlp_out(F.silu(m.action_time_mlp_in(
            torch.cat([a, temb[..., None, :].expand_as(a)], -1))))
        for li, layer in enumerate(self.layers):
            h, _, _ = self._layer(layer, h, c["rope"], c["K"][li], c["V"][li], c["mask"], li)
        return m.action_out_proj(rms_norm(h, self.final_norm.weight))

    def flow(self, z, c, n_steps=N_STEPS):
        """Forward Euler from t = 1 to 0, exactly as pi0 samples. z [N, H, 32]."""
        dt = -1.0 / n_steps
        x = z
        for i in range(n_steps):
            t = torch.full(z.shape[:1], 1.0 + i * dt, dtype=torch.float32, device=z.device)
            x = x + dt * self.velocity(x, t, c)
        return x

    # ---------------------------------------------------------------- convenience (numpy in/out)
    def cond(self, pos, objs, prompt):
        """pos [N, 2], objs [N, 4, 4], prompt: robot3 prompt key (or a raw instruction)."""
        n = len(pos)
        text = instruction(prompt) if prompt in ALL_PROMPTS else prompt
        st = np.zeros((n, D_ACT)); st[:, :2] = state_vec(pos)
        return self.prefix(render(pos, objs, self.dev), [text] * n, st)

    def _z(self, z):
        return torch.as_tensor(np.asarray(z).reshape(len(z), H, D_ACT), dtype=self.dt, device=self.dev)

    @torch.no_grad()
    def flow_np(self, z, c, bs=512):
        """Task chunks [N, 2H] (robot3 layout: step-major (dx, dy) / ACT_SCALE) for seeds z
        [N, 32H] under conditioning c of batch N, or of batch 1 (shared by every seed)."""
        out = []
        for i in range(0, len(z), bs):
            zz = self._z(z[i:i + bs])
            out.append(self.flow(zz, cexpand(c, len(zz)) if cbatch(c) == 1 else cslice(c, i, i + bs))
                       [..., :A_TASK].flatten(1))
        return torch.cat(out).double().cpu().numpy()

    def jac(self, z, c, bs=32):
        """Chunks [N, 2H] and dPhi_task/dz [N, 2H, 32H] (reverse mode: 2H = 24 rows), for c of
        batch N or 1 (shared)."""
        shared = cbatch(c) == 1
        c = cmap(c, lambda u: u.detach())

        def f(z1, c1):
            return self.flow(z1[None], cmap(c1, lambda u: u[None]))[0, :, :A_TASK].flatten()

        out_J = []
        for i in range(0, len(z), bs):
            zz = self._z(z[i:i + bs])
            ci = cmap(c, lambda u: u[0]) if shared else cslice(c, i, i + bs)
            out_J.append(vmap(jacrev(f), in_dims=(0, None if shared else 0))(zz, ci).flatten(-2).double().cpu())
        return self.flow_np(z, c), torch.cat(out_J).numpy()

    def sample(self, z, pos, objs, prompt, bs=64):
        """Per-sample conditioning: z [N, 32H], pos [N, 2], objs [N, 4, 4] -> chunks [N, 2H]."""
        out = []
        for i in range(0, len(z), bs):
            with torch.no_grad():
                out.append(self.flow_np(z[i:i + bs], self.cond(pos[i:i + bs], objs[i:i + bs], prompt)))
        return np.concatenate(out)

    def jac_z(self, z, pos, objs, prompt, bs=32):
        out_a, out_J = [], []
        for i in range(0, len(z), bs):
            with torch.no_grad():
                c = self.cond(pos[i:i + bs], objs[i:i + bs], prompt)
            a, J = self.jac(z[i:i + bs], c, bs)
            out_a.append(a); out_J.append(J)
        return np.concatenate(out_a), np.concatenate(out_J)

    # ---------------------------------------------------------------- io
    def save(self, path, **meta):
        state = {k: v.detach().cpu() for mod_name, mod in zip(
            ["expert", "action_in_proj", "action_out_proj", "state_proj", "action_time_mlp_in", "action_time_mlp_out"],
            self.expert_modules()) for k, v in [(f"{mod_name}.{n}", p) for n, p in mod.state_dict().items()]
            if "lm_head" not in k}                                 # the expert's unused 257k-vocab head
        torch.save(dict(state=state, **meta), path)

    def load_expert(self, path):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        names = ["expert", "action_in_proj", "action_out_proj", "state_proj", "action_time_mlp_in", "action_time_mlp_out"]
        for name, mod in zip(names, self.expert_modules()):
            sd = {k[len(name) + 1:]: v for k, v in ck["state"].items() if k.startswith(name + ".")}
            missing, unexpected = mod.load_state_dict(sd, strict=False)
            assert not unexpected and all("lm_head" in k for k in missing), (missing, unexpected)
            mod.to(device=self.dev, dtype=self.dt)
        self.meta = {k: v for k, v in ck.items() if k != "state"}
        return self


def load(ckpt=None, **kw):
    """pi0_base with chunk size H, optionally with a fine-tuned expert from `ckpt`."""
    from lerobot.configs import PreTrainedConfig
    from lerobot.policies.pi0.modeling_pi0 import PI0Policy
    cfg = PreTrainedConfig.from_pretrained(BASE)
    cfg.chunk_size = cfg.n_action_steps = H
    cfg.device = "cpu"                                             # moved per-module in Pi0Flow
    pol = PI0Policy.from_pretrained(BASE, config=cfg)
    f = Pi0Flow(pol, **kw)
    return f.load_expert(ckpt) if ckpt else f


def ftle(J):
    return np.log(np.linalg.svd(J, compute_uv=False)[:, 0])


# ----------------------------------------------------------------------------- training
def train(holdout=False, steps=5000, n_ep=20000, batch=32, n_tau=8, lr=1e-4, warmup=300, seed=0, log_every=100,
          ckpt=None, n_arrive=2, arrive_jit=0.0):
    """Fine-tune the action expert (VLM frozen) with pi0's loss: x_t = t z + (1-t) a,
    target z - a, t ~ 0.001 + 0.999 Beta(1.5, 1); loss on the task dims only (as lerobot).
    Each prefix (image + prompt) is reused for n_tau (noise, t) draws, since the frozen
    VLM forward dominates the cost. The step runs under bfloat16 autocast (fp32 master
    weights): 0.4 s for the 32 prefixes vs 0.9 s in fp32. Evaluation uses the fp32 prefix,
    which is the centre of the bf16 one's rounding noise (see README)."""
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    prompts = train_prompts(holdout)
    pos, sid, objs, pid, ch = make_demos(n_ep, rng, prompts, n_arrive=n_arrive, arrive_jit=arrive_jit)
    print(f"pi0 expert fine-tune{' (holdout)' if holdout else ''}: {len(ch)} samples, "
          f"{len(prompts)} prompts, {steps} steps x {batch} prefixes x {n_tau} draws, "
          f"{n_arrive} arrived states/episode (jitter {arrive_jit})", flush=True)
    f = load(dtype=torch.float32)
    params = f.expert_parameters()
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.AdamW(params, lr, betas=(0.9, 0.95), weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1, (s + 1) / warmup) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1, s / steps)))))
    texts = [instruction(p) for p in prompts]
    t0 = time.time(); ema = None
    for s_ in range(steps):
        idx = rng.integers(len(ch), size=batch)
        st = np.zeros((batch, D_ACT)); st[:, :2] = state_vec(pos[idx])
        a1 = torch.zeros(batch, H, D_ACT, device=f.dev)
        a1[..., :A_TASK] = torch.as_tensor(ch[idx], dtype=torch.float32, device=f.dev).view(batch, H, A_TASK)
        # n_tau draws per prefix: add a draw axis and let the cache broadcast over it
        a1 = a1[:, None].expand(batch, n_tau, H, D_ACT)
        z = torch.randn_like(a1)
        t = f.m.sample_time(batch * n_tau, f.dev).view(batch, n_tau)
        xt = t[..., None, None] * z + (1 - t[..., None, None]) * a1
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=f.dev.type == "cuda"):
            c = f.prefix(render(pos[idx], objs[sid[idx]], f.dev), [texts[i] for i in pid[idx]], st)
            cb = dict(K=[k[:, None] for k in c["K"]], V=[v[:, None] for v in c["V"]], mask=c["mask"][:, None],
                      rope=tuple(r[:, None] for r in c["rope"]))
            v = f.velocity(xt, t, cb).float()
        loss = ((v - (z - a1))[..., :A_TASK] ** 2).mean()
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sched.step()
        ema = loss.detach() if ema is None else 0.98 * ema + 0.02 * loss.detach()
        if s_ % log_every == 0 or s_ == steps - 1:
            print(f"step {s_:6d}  loss(ema) {ema.item():.4f}  lr {sched.get_last_lr()[0]:.2e}  "
                  f"{time.time() - t0:.0f}s", flush=True)
        if ckpt and (s_ + 1) % 1000 == 0:
            f.save(ckpt, holdout=holdout, H=H, step=s_ + 1, n_arrive=n_arrive, arrive_jit=arrive_jit)
    return f


# ----------------------------------------------------------------------------- checks
def check(ckpt=None, n=3, n_dir=12, eps=1e-4):
    """(1) the explicit expert flow equals lerobot's sample_actions on the same noise;
    (2) dPhi/dz (reverse mode) equals central finite differences along random directions,
    with the expert in float64."""
    from env4 import make_scenes, START_JIT
    rng = np.random.default_rng(0)
    objs, _ = make_scenes(n, rng)
    pos = rng.uniform(-1, 1, (n, 2)) * START_JIT
    for dt in (torch.float32, torch.float64):
        f = load(ckpt, dtype=dt) if dt == torch.float32 else (f.__init__(f.pol, dtype=dt) or f)
        z = rng.standard_normal((n, H * D_ACT))
        prompt = "square"
        if dt == torch.float32:
            st = np.zeros((n, D_ACT)); st[:, :2] = state_vec(pos)
            img = render(pos, objs, f.dev) * 2 - 1
            tok, tm = f.tokenize([instruction(prompt)] * n)
            ref = f.m.sample_actions([img], [torch.ones(n, dtype=torch.bool, device=f.dev)], tok, tm,
                                     torch.as_tensor(st, dtype=torch.float32, device=f.dev),
                                     noise=f._z(z), num_steps=N_STEPS)[..., :A_TASK].flatten(1).double().cpu().numpy()
            mine = f.sample(z, pos, objs, prompt)
            print(f"explicit flow vs lerobot sample_actions: max abs diff {np.abs(mine - ref).max():.2e} "
                  f"(|a| max {np.abs(ref).max():.2f})", flush=True)
            continue
        a, J = f.jac_z(z, pos, objs, prompt, bs=1)          # same batch shape as the FD below
        U = rng.standard_normal((n_dir, H * D_ACT)); U /= np.linalg.norm(U, axis=1, keepdims=True)
        err = []
        with torch.no_grad():
            for i in range(n):
                # hold the conditioning fixed: the VLM's K/V depend slightly on the batch size
                # (2e-5 in float32, 6e-2 in bfloat16), which would swamp a finite difference
                c = f.cond(pos[i:i + 1], objs[i:i + 1], prompt)
                fd = (f.flow_np(z[i] + eps * U, c) - f.flow_np(z[i] - eps * U, c)) / (2 * eps)
                err.append(np.abs(U @ J[i].T - fd).max() / np.abs(fd).max())
        print(f"dPhi/dz (float64) vs central FD, {n_dir} random directions x {n} scenes: "
              f"max rel err {max(err):.2e} | FTLE {np.round(ftle(J), 3)}", flush=True)


if __name__ == "__main__":
    args = sys.argv[1:]
    if args[0] == "check":
        check(args[1] if len(args) > 1 else None)
    elif args[0] == "train":
        # python pi0flow.py train [holdout] [arrive] [steps]
        #   arrive: 6 arrived states per episode, jittered within contact distance (0.31)
        holdout, arrive = "holdout" in args, "arrive" in args
        nums = [int(x) for x in args[1:] if x.isdigit()]
        kw = dict(n_arrive=6, arrive_jit=0.31) if arrive else {}
        os.makedirs(os.path.join(HERE, "policies"), exist_ok=True)
        name = "pi0" + ("_arrive" if arrive else "") + ("_holdout" if holdout else "")
        path = os.path.join(HERE, "policies", f"{name}.pt")
        f = train(holdout, steps=nums[0] if nums else 5000, ckpt=path, **kw)
        f.save(path, holdout=holdout, H=H, step="final", **kw)
