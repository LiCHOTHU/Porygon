"""
Toy v2: critic-as-compass vs faithful drift-field update, in TWO regimes.

v1 lessons folded in:
  - FIELD now mirrors the real Porygon-C update: V_Q (tilted or top-k
    attraction over own particles, self-repulsion) + V_BC (attraction toward
    frozen-base samples, self-repulsion) + plain -lambda*r restore, total clip.
    v1 omitted V_BC and the cloud inflated off-manifold (41%) - V_BC is
    load-bearing.
  - EASY regime (2-D, 200 critic points) is the critic's best case; v1 showed
    the compass wins there. v2 adds a HARD regime (16-D actions, 60 critic
    points, overfit critic) where extrapolation error should be real.

Arms: GRAD_free, GRAD_bc (control-A analog), FIELD_tilted, FIELD_topk (real C).
"""
import json
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

assert torch.cuda.is_available(), "run on GPU"
DEV = "cuda"
OUT = sys.argv[1] if len(sys.argv) > 1 else "."
os.makedirs(OUT, exist_ok=True)

STEPS = 1500
LOG_EVERY = 25
N_SEEDS = 3
S_MODE = 0.12


class Regime:
    def __init__(self, dim, n_critic, critic_steps, name):
        self.dim, self.n_critic, self.critic_steps, self.name = dim, n_critic, critic_steps, name
        m = torch.zeros(dim, device=DEV)
        self.m1, self.m2 = m.clone(), m.clone()
        self.m1[0], self.m2[0] = -1.0, 1.0
        self.rw = 0.15 * math.sqrt(dim / 2)          # reward width ~ on-manifold noise
        self.radii = tuple(r * math.sqrt(dim / 2) for r in (0.1, 0.3, 0.9))
        self.off_thr = 0.5 * math.sqrt(dim / 2)

    def true_reward(self, a):
        r1 = torch.exp(-((a - self.m1) ** 2).sum(-1) / (2 * self.rw ** 2))
        r2 = 0.6 * torch.exp(-((a - self.m2) ** 2).sum(-1) / (2 * self.rw ** 2))
        return r1 + r2

    def sample_data(self, n):
        pick = (torch.rand(n, device=DEV) < 0.5)[:, None]
        return torch.where(pick, self.m1, self.m2) + S_MODE * torch.randn(n, self.dim, device=DEV)


def mlp(i, o):
    return nn.Sequential(nn.Linear(i, 128), nn.ReLU(),
                         nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, o))


def drift_field(query, pos, w_pos, neg, radii, mask_pos_self=False):
    V = torch.zeros_like(query)
    K = len(query)
    for R in radii:
        dq = torch.cdist(query, pos)
        if mask_pos_self and pos.shape == query.shape:
            dq = dq + torch.eye(K, device=DEV) * 1e6
        aff_p = (torch.softmax(-dq / R, 1) * torch.softmax(-dq / R, 0)).clamp_min(1e-9).sqrt() * w_pos[None, :]
        dn = torch.cdist(query, neg) + torch.eye(K, device=DEV) * 1e6
        aff_n = (torch.softmax(-dn / R, 1) * torch.softmax(-dn / R, 0)).clamp_min(1e-9).sqrt()
        V = V + (aff_p[:, :, None] * (pos[None] - query[:, None])).sum(1) \
              - (aff_n[:, :, None] * (neg[None] - query[:, None])).sum(1)
    return V


def metrics(rg, policy, critic, n=512):
    with torch.no_grad():
        a = policy(torch.randn(n, rg.dim, device=DEV))
        r = rg.true_reward(a)
        q = critic(a).squeeze(-1)
        d1 = (a - rg.m1).norm(dim=-1)
        d2 = (a - rg.m2).norm(dim=-1)
        offman = (torch.minimum(d1, d2) > rg.off_thr).float().mean().item()
        disp = torch.cdist(a[:256], a[:256]).mean().item()
        a16 = policy(torch.randn(16 * 64, rg.dim, device=DEV)).reshape(64, 16, rg.dim)
        q16 = critic(a16.reshape(-1, rg.dim)).reshape(64, 16)
        bon = rg.true_reward(a16[torch.arange(64), q16.argmax(1)]).mean().item()
    return dict(true_mean=r.mean().item(), q_mean=q.mean().item(),
                gap=q.mean().item() - r.mean().item(), offman=offman,
                disp=disp, m1_frac=(d1 < d2).float().mean().item(), bon16_true=bon)


def field_finetune(rg, fresh, base_frozen, critic, mode):
    pol = fresh()
    opt = torch.optim.Adam(pol.parameters(), 3e-4)
    q_step, bc_step, clip_n, lam = 0.2, 0.2, 0.15, 1.0
    hist = []
    for t in range(STEPS):
        z = torch.randn(64, rg.dim, device=DEV)
        with torch.no_grad():
            cur = pol(z)
            q = critic(cur).squeeze(-1)
            if mode == "tilted":
                adv = (q - q.mean()) / q.std().clamp_min(1e-6)
                w = torch.softmax(adv / 0.5, 0) * len(cur)
                VQ = drift_field(cur, cur, w, cur, rg.radii, mask_pos_self=True)
            else:  # topk (real C recipe: top-4-of-64 attractors)
                idx = q.topk(4).indices
                VQ = drift_field(cur, cur[idx], torch.ones(4, device=DEV), cur, rg.radii)
            bc = base_frozen(torch.randn(64, rg.dim, device=DEV))
            VBC = drift_field(cur, bc, torch.ones(64, device=DEV), cur, rg.radii)
            res = cur - base_frozen(z)
            delta = q_step * VQ + bc_step * VBC - lam * res
            dn = delta.norm(dim=-1, keepdim=True)
            tgt = cur + delta * torch.clamp(clip_n / (dn + 1e-8), max=1.0)
        l = F.mse_loss(pol(z), tgt)
        opt.zero_grad(); l.backward(); opt.step()
        if t % LOG_EVERY == 0:
            hist.append(metrics(rg, pol, critic))
    return pol, hist


def run_seed(rg, seed):
    torch.manual_seed(seed)
    critic = mlp(rg.dim, 1).to(DEV)
    da = rg.sample_data(rg.n_critic)
    dy = rg.true_reward(da)[:, None] + 0.02 * torch.randn(rg.n_critic, 1, device=DEV)
    opt = torch.optim.Adam(critic.parameters(), 1e-3)
    for _ in range(rg.critic_steps):
        i = torch.randint(0, rg.n_critic, (64,), device=DEV)
        l = F.mse_loss(critic(da[i]), dy[i])
        opt.zero_grad(); l.backward(); opt.step()
    for p in critic.parameters():
        p.requires_grad_(False)

    base = mlp(rg.dim, rg.dim).to(DEV)
    opt = torch.optim.Adam(base.parameters(), 1e-3)
    for _ in range(3000):
        z = torch.randn(64, rg.dim, device=DEV)
        with torch.no_grad():
            cur = base(z)
            V = drift_field(cur, rg.sample_data(64), torch.ones(64, device=DEV), cur, rg.radii)
            tgt = cur + 0.5 * V
        l = F.mse_loss(base(z), tgt)
        opt.zero_grad(); l.backward(); opt.step()
    base_sd = {k: v.clone() for k, v in base.state_dict().items()}
    base_frozen = mlp(rg.dim, rg.dim).to(DEV)
    base_frozen.load_state_dict(base_sd)
    for p in base_frozen.parameters():
        p.requires_grad_(False)

    def fresh():
        m = mlp(rg.dim, rg.dim).to(DEV)
        m.load_state_dict(base_sd)
        return m

    arms, hist = {}, {}
    for name, lam in [("GRAD_free", 0.0), ("GRAD_bc", 1.0)]:
        pol = fresh()
        opt = torch.optim.Adam(pol.parameters(), 3e-4)
        hist[name] = []
        for t in range(STEPS):
            z = torch.randn(64, rg.dim, device=DEV)
            a = pol(z)
            loss = -critic(a).mean() + lam * F.mse_loss(a, base_frozen(z))
            opt.zero_grad(); loss.backward(); opt.step()
            if t % LOG_EVERY == 0:
                hist[name].append(metrics(rg, pol, critic))
        arms[name] = pol
    for mode in ["tilted", "topk"]:
        arms[f"FIELD_{mode}"], hist[f"FIELD_{mode}"] = field_finetune(rg, fresh, base_frozen, critic, mode)

    finals = {"BASE": metrics(rg, base_frozen, critic)}
    finals.update({k: metrics(rg, m, critic) for k, m in arms.items()})
    return finals, hist


NAMES = ["BASE", "GRAD_free", "GRAD_bc", "FIELD_tilted", "FIELD_topk"]
results = {}
for rg in [Regime(2, 200, 4000, "EASY_2d"), Regime(16, 60, 8000, "HARD_16d")]:
    fs, hs = [], []
    for s in range(N_SEEDS):
        f, h = run_seed(rg, s)
        fs.append(f); hs.append(h)
        print(f"[{rg.name} seed {s}] " + " | ".join(
            f"{k}: true={v['true_mean']:.3f} gap={v['gap']:.3f} off={v['offman']:.2f} "
            f"bon16={v['bon16_true']:.3f}" for k, v in f.items()))
    agg = {n: {k: sum(f[n][k] for f in fs) / N_SEEDS for k in fs[0][n]} for n in NAMES}
    results[rg.name] = {"finals": fs, "agg": agg}
    print(f"\n=== {rg.name} MEAN ===")
    for n in NAMES:
        v = agg[n]
        print(f"{n:12s} true={v['true_mean']:.3f} q={v['q_mean']:.3f} gap={v['gap']:.3f} "
              f"offman={v['offman']:.2f} disp={v['disp']:.2f} m1={v['m1_frac']:.2f} "
              f"bon16={v['bon16_true']:.3f}")
    # curves
    t = [i * LOG_EVERY for i in range(len(hs[0]["FIELD_tilted"]))]
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    for ax, key, ttl in zip(axes, ["true_mean", "gap", "disp", "bon16_true"],
                            ["true reward", "critic-true gap", "dispersion", "best-of-16 true"]):
        for nm in NAMES[1:]:
            ax.plot(t, [sum(h[nm][i][key] for h in hs) / N_SEEDS for i in range(len(t))], label=nm)
        ax.axhline(agg["BASE"][key], color="k", ls="--", lw=0.8, label="base")
        ax.set_title(f"{rg.name}: {ttl}"); ax.legend(fontsize=7)
    fig.savefig(f"{OUT}/toy2_curves_{rg.name}.png", dpi=110, bbox_inches="tight")

json.dump({k: v["agg"] for k, v in results.items()},
          open(f"{OUT}/toy2_results.json", "w"), indent=1)
print(f"\nwrote {OUT}/toy2_results.json + curve plots")
