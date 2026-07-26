"""
Toy demonstration: critic-as-compass vs critic-as-judge (drift field) on a
2-D bimodal bandit.

Setup
  - Expert manifold: two Gaussian modes m1=(-1,0) (true reward 1.0) and
    m2=(+1,0) (true reward 0.6), sigma=0.12. True reward ~0 off-manifold.
  - Critic: small MLP regressed on true reward at ON-MANIFOLD actions only
    (200 samples) -> off-manifold values are pure extrapolation (BCQ's
    "extrapolation error"). We never fix the critic; both updates see the
    same one.
  - Base policy: one-step generator z->a pretrained with the kernel drift
    operator toward data samples (a miniature drifting model, bimodal).

Fine-tuning arms (same base, same critic, same #updates, same lr)
  GRAD_free : maximize Q(g(z)) by backprop through the critic (DDPG-style)
  GRAD_bc   : same + ||g(z)-g_base(z)||^2 anchor (control-A in miniature)
  FIELD     : tilted drift-field update - particles moved toward the
              softmax(adv/tau)-weighted version of their own cloud, clipped,
              dead-zone restore; policy regressed onto displaced particles.
              The critic is only ever EVALUATED at sampled actions.

Metrics (per update, averaged over seeds)
  true_mean   E[R_true(a)], a~policy           - did fine-tuning actually help?
  q_mean      E[Q(a)]                          - what the critic THINKS
  gap         q_mean - true_mean               - critic-exploitation gap
  offman      P(min dist to a mode > 0.5)      - off-manifold fraction
  disp        mean pairwise distance of samples- dispersion (diversity)
  modes       fraction of samples nearest m1 / m2 (mode coverage)
  bon16_true  true reward of best-of-16-by-critic - selection channel
"""
import json
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

M1 = torch.tensor([-1.0, 0.0], device=DEV)
M2 = torch.tensor([1.0, 0.0], device=DEV)
S_MODE = 0.12
RADII = (0.1, 0.3, 0.9)
STEPS = 1500
LOG_EVERY = 25
N_SEEDS = 3


def true_reward(a):
    r1 = torch.exp(-((a - M1) ** 2).sum(-1) / (2 * 0.15 ** 2))
    r2 = 0.6 * torch.exp(-((a - M2) ** 2).sum(-1) / (2 * 0.15 ** 2))
    return r1 + r2


def sample_data(n):
    pick = (torch.rand(n, device=DEV) < 0.5)[:, None]
    return torch.where(pick, M1, M2) + S_MODE * torch.randn(n, 2, device=DEV)


def mlp(i, o):
    return nn.Sequential(nn.Linear(i, 128), nn.ReLU(),
                         nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, o))


def drift_field(query, pos, w_pos, neg):
    """Multi-radius softmax-softmax attract/repel kernel field (toy version of
    compute_drift_field). query (K,2); pos (P,2) with weights w_pos (P,);
    neg (K,2) assumed == query (self-repulsion, diagonal masked)."""
    V = torch.zeros_like(query)
    for R in RADII:
        dq = torch.cdist(query, pos)
        aff_p = (torch.softmax(-dq / R, 1) * torch.softmax(-dq / R, 0)).clamp_min(1e-9).sqrt()
        aff_p = aff_p * w_pos[None, :]
        dn = torch.cdist(query, neg) + torch.eye(len(query), device=DEV) * 1e6
        aff_n = (torch.softmax(-dn / R, 1) * torch.softmax(-dn / R, 0)).clamp_min(1e-9).sqrt()
        attr = (aff_p[:, :, None] * (pos[None] - query[:, None])).sum(1)
        rep = -(aff_n[:, :, None] * (neg[None] - query[:, None])).sum(1)
        V = V + attr + rep
    return V


def metrics(policy, critic, n=512):
    with torch.no_grad():
        a = policy(torch.randn(n, 2, device=DEV))
        r = true_reward(a)
        q = critic(a).squeeze(-1)
        d1 = (a - M1).norm(dim=-1)
        d2 = (a - M2).norm(dim=-1)
        offman = (torch.minimum(d1, d2) > 0.5).float().mean().item()
        disp = torch.cdist(a[:256], a[:256]).mean().item()
        m1_frac = (d1 < d2).float().mean().item()
        # best-of-16 by critic (selection channel)
        a16 = policy(torch.randn(16 * 64, 2, device=DEV)).reshape(64, 16, 2)
        q16 = critic(a16.reshape(-1, 2)).reshape(64, 16)
        pick = q16.argmax(dim=1)
        bon = true_reward(a16[torch.arange(64), pick]).mean().item()
    return dict(true_mean=r.mean().item(), q_mean=q.mean().item(),
                gap=q.mean().item() - r.mean().item(), offman=offman,
                disp=disp, m1_frac=m1_frac, bon16_true=bon)


def run_seed(seed):
    torch.manual_seed(seed)
    # --- critic on limited on-manifold data ---
    critic = mlp(2, 1).to(DEV)
    da = sample_data(200)
    dy = true_reward(da)[:, None] + 0.02 * torch.randn(200, 1, device=DEV)
    opt = torch.optim.Adam(critic.parameters(), 1e-3)
    for _ in range(4000):
        i = torch.randint(0, 200, (64,), device=DEV)
        l = F.mse_loss(critic(da[i]), dy[i])
        opt.zero_grad(); l.backward(); opt.step()
    for p in critic.parameters():
        p.requires_grad_(False)

    # --- base policy pretrained with the drift operator toward data ---
    base = mlp(2, 2).to(DEV)
    opt = torch.optim.Adam(base.parameters(), 1e-3)
    for _ in range(3000):
        z = torch.randn(64, 2, device=DEV)
        with torch.no_grad():
            cur = base(z)
            data = sample_data(64)
            V = drift_field(cur, data, torch.ones(64, device=DEV), cur)
            tgt = cur + 0.5 * V
        l = F.mse_loss(base(z), tgt)
        opt.zero_grad(); l.backward(); opt.step()
    base_sd = {k: v.clone() for k, v in base.state_dict().items()}
    base_frozen = mlp(2, 2).to(DEV)
    base_frozen.load_state_dict(base_sd)
    for p in base_frozen.parameters():
        p.requires_grad_(False)

    def fresh():
        m = mlp(2, 2).to(DEV)
        m.load_state_dict(base_sd)
        return m

    arms, hist, finals = {}, {}, {}
    # --- GRAD arms: backprop Q through the actor ---
    for name, lam in [("GRAD_free", 0.0), ("GRAD_bc", 1.0)]:
        pol = fresh()
        opt = torch.optim.Adam(pol.parameters(), 3e-4)
        hist[name] = []
        for t in range(STEPS):
            z = torch.randn(64, 2, device=DEV)
            a = pol(z)
            loss = -critic(a).mean() + lam * F.mse_loss(a, base_frozen(z))
            opt.zero_grad(); loss.backward(); opt.step()
            if t % LOG_EVERY == 0:
                hist[name].append(metrics(pol, critic))
        arms[name] = pol

    # --- FIELD arm: tilted drift-field update, critic evaluated only ---
    pol = fresh()
    opt = torch.optim.Adam(pol.parameters(), 3e-4)
    q_step, clip, tau = 1.0, 0.2, 0.5
    rest_step, rest_rad = 1.0, 0.3
    hist["FIELD"] = []
    for t in range(STEPS):
        z = torch.randn(64, 2, device=DEV)
        with torch.no_grad():
            cur = pol(z)
            q = critic(cur).squeeze(-1)
            adv = (q - q.mean()) / q.std().clamp_min(1e-6)
            w = torch.softmax(adv / tau, 0) * len(cur)
            V = drift_field(cur, cur, w, cur)
            res = cur - base_frozen(z)
            rn = res.norm(dim=-1, keepdim=True)
            gate = (1 - rest_rad / rn.clamp_min(1e-8)).clamp(min=0.0)
            delta = q_step * V - rest_step * gate * res
            dn = delta.norm(dim=-1, keepdim=True)
            delta = delta * torch.clamp(clip / (dn + 1e-8), max=1.0)
            tgt = cur + delta
        l = F.mse_loss(pol(z), tgt)
        opt.zero_grad(); l.backward(); opt.step()
        if t % LOG_EVERY == 0:
            hist["FIELD"].append(metrics(pol, critic))
    arms["FIELD"] = pol

    finals["BASE"] = metrics(base_frozen, critic)
    for k, m in arms.items():
        finals[k] = metrics(m, critic)
    return critic, base_frozen, arms, hist, finals


all_hist, all_finals = [], []
keep = None
for s in range(N_SEEDS):
    critic, base, arms, hist, finals = run_seed(s)
    all_hist.append(hist)
    all_finals.append(finals)
    if s == 0:
        keep = (critic, base, arms)
    print(f"[seed {s}] " + " | ".join(
        f"{k}: true={v['true_mean']:.3f} gap={v['gap']:.3f} off={v['offman']:.2f} "
        f"disp={v['disp']:.2f} bon16={v['bon16_true']:.3f}" for k, v in finals.items()))

# --- aggregate ---
names = ["BASE", "GRAD_free", "GRAD_bc", "FIELD"]
agg = {n: {k: sum(f[n][k] for f in all_finals) / N_SEEDS for k in all_finals[0][n]}
       for n in names}
print("\n=== MEAN OVER SEEDS ===")
for n in names:
    v = agg[n]
    print(f"{n:9s} true={v['true_mean']:.3f} q={v['q_mean']:.3f} gap={v['gap']:.3f} "
          f"offman={v['offman']:.2f} disp={v['disp']:.2f} m1={v['m1_frac']:.2f} "
          f"bon16={v['bon16_true']:.3f}")
json.dump({"finals": all_finals, "agg": agg}, open(f"{OUT}/toy_results.json", "w"), indent=1)

# --- plots (seed 0) ---
critic, base, arms = keep
xs = torch.linspace(-2.5, 2.5, 200)
ys = torch.linspace(-2.0, 2.0, 160)
G = torch.stack(torch.meshgrid(xs, ys, indexing="xy"), -1).reshape(-1, 2).to(DEV)
with torch.no_grad():
    Q = critic(G).reshape(160, 200).cpu()
    R = true_reward(G).reshape(160, 200).cpu()

fig, axes = plt.subplots(1, 4, figsize=(22, 5))
for ax, (nm, mdl) in zip(axes, [("BASE", base)] + [(k, arms[k]) for k in ["GRAD_free", "GRAD_bc", "FIELD"]]):
    ax.imshow(Q, extent=[-2.5, 2.5, -2, 2], origin="lower", cmap="viridis", alpha=0.8)
    ax.contour(xs, ys, R, levels=[0.2, 0.5, 0.8], colors="w", linewidths=0.8)
    with torch.no_grad():
        a = mdl(torch.randn(400, 2, device=DEV)).cpu()
    ax.scatter(a[:, 0], a[:, 1], s=4, c="red")
    ax.set_title(nm)
fig.suptitle("critic heatmap (bg), true-reward contours (white), policy samples (red)")
fig.savefig(f"{OUT}/toy_samples.png", dpi=110, bbox_inches="tight")

fig, axes = plt.subplots(1, 4, figsize=(20, 4))
t = [i * LOG_EVERY for i in range(len(all_hist[0]["FIELD"]))]
for ax, key, ttl in zip(axes, ["true_mean", "gap", "disp", "bon16_true"],
                        ["true reward", "critic-true gap (exploitation)",
                         "dispersion", "best-of-16-by-critic true reward"]):
    for nm in ["GRAD_free", "GRAD_bc", "FIELD"]:
        cur = [sum(h[nm][i][key] for h in all_hist) / N_SEEDS for i in range(len(t))]
        ax.plot(t, cur, label=nm)
    ax.axhline(agg["BASE"][key], color="k", ls="--", lw=0.8, label="base")
    ax.set_title(ttl); ax.legend(fontsize=7)
fig.savefig(f"{OUT}/toy_curves.png", dpi=110, bbox_inches="tight")
print(f"wrote {OUT}/toy_results.json, toy_samples.png, toy_curves.png")
