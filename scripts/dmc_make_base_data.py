"""Stage 1-2 of the DMC experiment: produce a *partially trained* policy on the
DENSE task, then harvest a demonstration set from it.

Why this shape. Our method is a fine-tuner: it needs a frozen base to anchor to
and to bound movement from, so it cannot be run from scratch the way FPO is.
The benchmark setting that matches it is demos -> BC base -> sparse-reward RL
fine-tuning, which is exactly the robomimic/LIBERO protocol. DMC lets us build
that on a standard RL benchmark, with the added advantage that
cartpole swingup / swingup_sparse are the same dynamics under two reward
densities -- a controlled knob on critic quality.

SAC here is only a data generator, not part of the contribution: both the
field-target arm and the backprop control are later fine-tuned from the same
distilled base, so any weakness of this policy is shared by both.

Deliberately stopped early ("partially run RL") so the base has real headroom
left for the fine-tuner to recover.

Diversity note: a converged SAC policy is near-deterministic, and cloning it
would give a base with almost no multimodality -- which would quietly remove one
of the drifting model's advantages and understate our case. We therefore harvest
from SEVERAL checkpoints along training and inject exploration noise, so the
demo set contains genuinely different ways of solving the task.

Usage: python dmc_make_base_data.py <domain> <dense_task> <out_dir> [env_steps]
"""
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.expanduser("~/workspace/dice_rl_official/dice-rl"))

DOMAIN = sys.argv[1] if len(sys.argv) > 1 else "cartpole"
TASK = sys.argv[2] if len(sys.argv) > 2 else "swingup"
OUT = sys.argv[3] if len(sys.argv) > 3 else "."
TOTAL_STEPS = int(sys.argv[4]) if len(sys.argv) > 4 else 60000

assert torch.cuda.is_available(), "run on a GPU"
DEV = "cuda"
os.makedirs(OUT, exist_ok=True)
torch.manual_seed(0)
np.random.seed(0)

from dm_control import suite
from env.gym_utils.wrapper.dmc_lowdim import flatten_dmc_obs

env = suite.load(DOMAIN, TASK, task_kwargs={"random": 0})
spec = env.action_spec()
ADIM = int(np.prod(spec.shape))
A_LOW = np.asarray(spec.minimum, dtype=np.float32).reshape(-1)
A_HIGH = np.asarray(spec.maximum, dtype=np.float32).reshape(-1)
ODIM = flatten_dmc_obs(env.reset().observation).shape[0]
print(f"{DOMAIN}-{TASK}: obs_dim={ODIM} act_dim={ADIM}", flush=True)

H = 256
LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0


def mlp(i, o, out_act=None):
    layers = [nn.Linear(i, H), nn.ReLU(), nn.Linear(H, H), nn.ReLU(), nn.Linear(H, o)]
    if out_act is not None:
        layers.append(out_act)
    return nn.Sequential(*layers)


class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = mlp(ODIM, 2 * ADIM)

    def forward(self, s):
        mu, log_std = self.net(s).chunk(2, -1)
        return mu, log_std.clamp(LOG_STD_MIN, LOG_STD_MAX)

    def sample(self, s):
        mu, log_std = self(s)
        std = log_std.exp()
        x = mu + std * torch.randn_like(mu)
        a = torch.tanh(x)  # policy acts in [-1, 1]
        logp = (-0.5 * ((x - mu) / std) ** 2 - log_std - 0.5 * np.log(2 * np.pi)).sum(-1)
        logp = logp - torch.log(1 - a.pow(2) + 1e-6).sum(-1)
        return a, logp, torch.tanh(mu)


actor = Actor().to(DEV)
q1, q2 = mlp(ODIM + ADIM, 1).to(DEV), mlp(ODIM + ADIM, 1).to(DEV)
q1t, q2t = mlp(ODIM + ADIM, 1).to(DEV), mlp(ODIM + ADIM, 1).to(DEV)
q1t.load_state_dict(q1.state_dict())
q2t.load_state_dict(q2.state_dict())
opt_a = torch.optim.Adam(actor.parameters(), 3e-4)
opt_q = torch.optim.Adam(list(q1.parameters()) + list(q2.parameters()), 3e-4)
log_alpha = torch.zeros(1, requires_grad=True, device=DEV)
opt_alpha = torch.optim.Adam([log_alpha], 3e-4)
TARGET_ENT = -float(ADIM)

CAP = 200000
S = np.zeros((CAP, ODIM), np.float32)
A = np.zeros((CAP, ADIM), np.float32)
R = np.zeros((CAP, 1), np.float32)
S2 = np.zeros((CAP, ODIM), np.float32)
ptr = size = 0


def to_env(a):
    a = np.clip(a, -1, 1)
    return (a + 1) / 2 * (A_HIGH - A_LOW) + A_LOW


ts = env.reset()
s = flatten_dmc_obs(ts.observation)
ep_ret, ep_rets = 0.0, []
CKPT_AT = sorted({int(TOTAL_STEPS * f) for f in (0.35, 0.5, 0.7, 1.0)})
ckpts = []

for step in range(1, TOTAL_STEPS + 1):
    if step < 2000:
        a = np.random.uniform(-1, 1, ADIM).astype(np.float32)
    else:
        with torch.no_grad():
            a, _, _ = actor.sample(torch.as_tensor(s, device=DEV)[None])
        a = a[0].cpu().numpy()
    ts = env.step(to_env(a))
    s2 = flatten_dmc_obs(ts.observation)
    r = float(ts.reward or 0.0)
    S[ptr], A[ptr], R[ptr], S2[ptr] = s, a, r, s2
    ptr = (ptr + 1) % CAP
    size = min(size + 1, CAP)
    s, ep_ret = s2, ep_ret + r
    if ts.last():
        ep_rets.append(ep_ret)
        ts = env.reset()
        s, ep_ret = flatten_dmc_obs(ts.observation), 0.0

    if step >= 2000:
        idx = np.random.randint(0, size, 256)
        bs = torch.as_tensor(S[idx], device=DEV)
        ba = torch.as_tensor(A[idx], device=DEV)
        br = torch.as_tensor(R[idx], device=DEV)
        bs2 = torch.as_tensor(S2[idx], device=DEV)
        with torch.no_grad():
            a2, logp2, _ = actor.sample(bs2)
            qt = torch.min(q1t(torch.cat([bs2, a2], -1)), q2t(torch.cat([bs2, a2], -1)))
            y = br + 0.99 * (qt - log_alpha.exp() * logp2[:, None])
        sa = torch.cat([bs, ba], -1)
        lq = F.mse_loss(q1(sa), y) + F.mse_loss(q2(sa), y)
        opt_q.zero_grad(); lq.backward(); opt_q.step()

        an, logpn, _ = actor.sample(bs)
        qn = torch.min(q1(torch.cat([bs, an], -1)), q2(torch.cat([bs, an], -1)))
        la = (log_alpha.exp().detach() * logpn[:, None] - qn).mean()
        opt_a.zero_grad(); la.backward(); opt_a.step()
        lal = -(log_alpha.exp() * (logpn.detach() + TARGET_ENT)).mean()
        opt_alpha.zero_grad(); lal.backward(); opt_alpha.step()
        with torch.no_grad():
            for p, pt in zip(list(q1.parameters()) + list(q2.parameters()),
                             list(q1t.parameters()) + list(q2t.parameters())):
                pt.mul_(0.995).add_(0.005 * p)

    if step % 5000 == 0:
        recent = np.mean(ep_rets[-5:]) if ep_rets else 0.0
        print(f"step {step:6d}  recent_return {recent:8.2f}", flush=True)
    if step in CKPT_AT:
        ckpts.append({k: v.detach().clone() for k, v in actor.state_dict().items()})
        print(f"  -> stashed checkpoint at step {step} "
              f"(return {np.mean(ep_rets[-5:]) if ep_rets else 0:.1f})", flush=True)

torch.save({"actor": actor.state_dict(), "ckpts": ckpts,
            "obs_dim": ODIM, "act_dim": ADIM},
           os.path.join(OUT, f"sac_{DOMAIN}_{TASK}.pt"))

# ---- harvest demos from several checkpoints, with exploration noise --------
EP_LEN = 1000
N_EP_PER_CKPT = 25
NOISES = [0.0, 0.1, 0.2]
all_states, all_actions, all_rewards, all_ends = [], [], [], []
for ci, sd in enumerate(ckpts):
    actor.load_state_dict(sd)
    for ep in range(N_EP_PER_CKPT):
        noise = NOISES[ep % len(NOISES)]
        env_e = suite.load(DOMAIN, TASK, task_kwargs={"random": 1000 + 37 * ci + ep})
        ts = env_e.reset()
        so = flatten_dmc_obs(ts.observation)
        for _ in range(EP_LEN):
            with torch.no_grad():
                _, _, det = actor.sample(torch.as_tensor(so, device=DEV)[None])
            a = det[0].cpu().numpy()
            if noise > 0:
                a = np.clip(a + noise * np.random.randn(ADIM).astype(np.float32), -1, 1)
            ts = env_e.step(to_env(a))
            all_states.append(so)
            all_actions.append(a)
            all_rewards.append(float(ts.reward or 0.0))
            so = flatten_dmc_obs(ts.observation)
            if ts.last():
                break
        all_ends.append(len(all_states))

states = np.asarray(all_states, np.float32)
actions = np.asarray(all_actions, np.float32)
rewards = np.asarray(all_rewards, np.float32)
ends = np.asarray(all_ends, np.int64)
starts = np.concatenate([[0], ends[:-1]])
ep_returns = np.array([rewards[a:b].sum() for a, b in zip(starts, ends)])

np.savez(os.path.join(OUT, f"demos_{DOMAIN}_{TASK}.npz"),
         states=states, actions=actions, rewards=rewards,
         traj_lengths=(ends - starts).astype(np.int64))
np.savez(os.path.join(OUT, f"normalization_{DOMAIN}_{TASK}.npz"),
         obs_min=states.min(0), obs_max=states.max(0),
         action_min=np.full(ADIM, -1, np.float32),
         action_max=np.full(ADIM, 1, np.float32))
print(f"\ndemos: {len(ep_returns)} episodes, {len(states)} transitions")
print(f"demo return  mean {ep_returns.mean():.1f}  min {ep_returns.min():.1f}  "
      f"max {ep_returns.max():.1f}  std {ep_returns.std():.1f}")
print("wrote", OUT, flush=True)
