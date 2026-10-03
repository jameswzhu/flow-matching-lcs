"""
Can FTLE reason over successive replans?  Composed (multi-chunk) sensitivity vs the
one-chunk FTLE, evaluated on the closed-loop flips of S3 (same 360 states x 24 seeds,
same perturbations, same future-noise stream).

    python mc_eval.py prep        # rebuild S3 cases, first-chunk Jacobians    (~5 min)
    python mc_eval.py oracle      # compose along the actual noise stream      (~11 min)
    python mc_eval.py zero        # compose with all-zero future seeds         (~11 min)
    python mc_eval.py rand1 rand2 # compose with independent random streams    (~11 min each)
    python mc_eval.py labels      # base / goal+- rollouts for A-vs-B flip split (~9 min)
    python mc_eval.py analyze     # AUROCs, figure, results_mc.json
"""
import json, os, sys, time
import numpy as np
from policy2 import Policy, A, ftle
from env2 import pending_column, route_of_traj, X_START, COLS, CORR
from multichunk import flow_obs_jac, composed, scores_from

STEPS = sys.argv[1:]
pol = Policy.load("policy2.npz")
R_REPLANS = 15
os.makedirs("mc", exist_ok=True)
t00 = time.time()


def load_prep():
    return dict(np.load("mc/prep.npz"))


if "prep" in STEPS:
    # replay eval2.py S3's random draws exactly
    import eval2_rollout as er
    rng = np.random.default_rng(51)
    n_ep = 150
    gE = rng.uniform(-0.9, 0.9, n_ep)
    pos0 = np.column_stack([np.full(n_ep, X_START), rng.uniform(-0.3, 0.3, n_ep)])
    _, states = er.rollout(pol, pos0, gE, seed=52)
    S = states.reshape(-1, 2); gS = np.repeat(gE, states.shape[1])
    keep = pending_column(S) >= 0
    S, gS = S[keep], gS[keep]
    pick = rng.choice(len(S), 360, replace=False); S, gS = S[pick], gS[pick]
    nz = 24
    Sp, gp = np.repeat(S, nz, 0), np.repeat(gS, nz)
    Z = rng.standard_normal((len(Sp), A))
    from eval2_rollout import obsn
    outs, P0s = [], []
    for i in range(0, len(Z), 1500):
        outs.append(pol.sample_jac(Z[i:i + 1500].astype(np.float32), obsn(Sp[i:i + 1500], gp[i:i + 1500]), 20))
        P0s.append(flow_obs_jac(pol, Z[i:i + 1500], Sp[i:i + 1500], gp[i:i + 1500])[1])
    a0 = np.concatenate([o[0] for o in outs]); F0 = np.concatenate([o[1] for o in outs]); P0 = np.concatenate(P0s)
    f1 = ftle(F0)
    ref = np.load("s3_rows.npz")
    print("prep: one-chunk FTLE reproduces S3:", float(np.abs(f1 - ref["ftle"]).max()), flush=True)
    np.savez_compressed("mc/prep.npz", Sp=Sp, gp=gp, Z=Z, a0=a0, F0=F0, P0=P0, ftle1=f1)
    print(f"prep done {time.time() - t00:.0f}s", flush=True)

for name in ["oracle", "zero", "rand1", "rand2"]:
    if name not in STEPS:
        continue
    P = load_prep(); N = len(P["Sp"])
    if name == "oracle":
        rng = np.random.default_rng(53)
        draws = [rng.standard_normal((N, A)).astype(np.float32) for _ in range(R_REPLANS)]
        stream = lambda k, n: draws[k]
    elif name == "zero":
        stream = lambda k, n: np.zeros((n, A), np.float32)
    else:
        rng = np.random.default_rng({"rand1": 61, "rand2": 62}[name])
        draws = [rng.standard_normal((N, A)).astype(np.float32) for _ in range(R_REPLANS)]
        stream = lambda k, n: draws[k]
    S, D, d, cD, cd, chs, chg = composed(pol, P["Sp"], P["gp"], P["Z"], P["F0"], P["P0"], P["a0"], stream, R_REPLANS)
    sc = scores_from(D, d, cD, cd, chs, chg)
    np.savez_compressed(f"mc/{name}.npz", states=S.astype(np.float32), **sc,
                        Dn=np.linalg.norm(D, axis=(2, 3)).astype(np.float32),
                        dn=np.linalg.norm(d, axis=2).astype(np.float32))
    print(f"{name} done {time.time() - t00:.0f}s", flush=True)

if "labels" in STEPS:
    import eval2_rollout as er
    P = load_prep()
    out = {}
    for nm, gg in [("base", P["gp"]), ("g+", P["gp"] + 0.05), ("g-", P["gp"] - 0.05)]:
        traj, st = er.rollout(pol, P["Sp"], gg, seed=53, first_noise=P["Z"])
        iA, iB, r = route_of_traj(traj)
        out[nm] = np.stack([iA, iB, r], 1)
        if nm == "base":
            np.save("mc/base_states.npy", st.astype(np.float32))
        print(f"labels {nm} {time.time() - t00:.0f}s", flush=True)
    np.savez_compressed("mc/labels.npz", **{k.replace("+", "p").replace("-", "m"): v for k, v in out.items()})

if "analyze" in STEPS:
    import plot_mc
    plot_mc.analyze()

print(f"done {time.time() - t00:.0f}s")
