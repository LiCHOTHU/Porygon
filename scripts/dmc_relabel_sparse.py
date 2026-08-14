"""Re-harvest the demonstration set in the SPARSE task using the already-trained
SAC checkpoints.

Why this exists. The base policy is trained on the DENSE variant (that is where
learning signal exists), but fine-tuning happens on the SPARSE variant. RLPD
mixes expert transitions with online transitions in the same critic, so shipping
dense-labelled demos into a sparse-reward fine-tune would train one critic on two
different reward functions -- a silent confound that would invalidate the whole
dense-vs-sparse comparison.

Re-rolling the same policy in the sparse env gives exactly-labelled sparse
rewards rather than an approximate relabelling of the stored dense ones, and
costs nothing because the SAC checkpoints are already saved.

Usage: python dmc_relabel_sparse.py <domain> <dense_task> <sparse_task> <dir>
"""
import os
import sys
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.expanduser("~/workspace/dice_rl_official/dice-rl"))

DOMAIN = sys.argv[1] if len(sys.argv) > 1 else "cartpole"
DENSE = sys.argv[2] if len(sys.argv) > 2 else "swingup"
SPARSE = sys.argv[3] if len(sys.argv) > 3 else "swingup_sparse"
DIR = sys.argv[4] if len(sys.argv) > 4 else "."

assert torch.cuda.is_available(), "run on a GPU"
DEV = "cuda"
np.random.seed(0)
torch.manual_seed(0)

from dm_control import suite
from env.gym_utils.wrapper.dmc_lowdim import flatten_dmc_obs

blob = torch.load(os.path.join(DIR, f"sac_{DOMAIN}_{DENSE}.pt"), map_location=DEV,
                  weights_only=False)
ODIM, ADIM = blob["obs_dim"], blob["act_dim"]
ckpts = blob["ckpts"]
print(f"loaded {len(ckpts)} SAC checkpoints (obs {ODIM}, act {ADIM})", flush=True)

H = 256
LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0


class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(ODIM, H), nn.ReLU(), nn.Linear(H, H), nn.ReLU(),
            nn.Linear(H, 2 * ADIM))

    def det(self, s):
        mu, _ = self.net(s).chunk(2, -1)
        return torch.tanh(mu)


actor = Actor().to(DEV)

probe = suite.load(DOMAIN, SPARSE, task_kwargs={"random": 0})
spec = probe.action_spec()
A_LOW = np.asarray(spec.minimum, dtype=np.float32).reshape(-1)
A_HIGH = np.asarray(spec.maximum, dtype=np.float32).reshape(-1)


def to_env(a):
    a = np.clip(a, -1, 1)
    return (a + 1) / 2 * (A_HIGH - A_LOW) + A_LOW


EP_LEN = 1000
N_EP_PER_CKPT = 25
NOISES = [0.0, 0.1, 0.2]
states, actions, rewards, ends = [], [], [], []

for ci, sd in enumerate(ckpts):
    actor.load_state_dict(sd)
    for ep in range(N_EP_PER_CKPT):
        noise = NOISES[ep % len(NOISES)]
        # Same seeds as the dense harvest, so the two sets differ only in reward.
        env = suite.load(DOMAIN, SPARSE, task_kwargs={"random": 1000 + 37 * ci + ep})
        ts = env.reset()
        so = flatten_dmc_obs(ts.observation)
        for _ in range(EP_LEN):
            with torch.no_grad():
                a = actor.det(torch.as_tensor(so, device=DEV)[None])[0].cpu().numpy()
            if noise > 0:
                a = np.clip(a + noise * np.random.randn(ADIM).astype(np.float32), -1, 1)
            ts = env.step(to_env(a))
            states.append(so)
            actions.append(a)
            rewards.append(float(ts.reward or 0.0))
            so = flatten_dmc_obs(ts.observation)
            if ts.last():
                break
        ends.append(len(states))

states = np.asarray(states, np.float32)
actions = np.asarray(actions, np.float32)
rewards = np.asarray(rewards, np.float32)
ends = np.asarray(ends, np.int64)
starts = np.concatenate([[0], ends[:-1]])
ep_returns = np.array([rewards[a:b].sum() for a, b in zip(starts, ends)])

np.savez(os.path.join(DIR, f"demos_{DOMAIN}_{SPARSE}.npz"),
         states=states, actions=actions, rewards=rewards,
         traj_lengths=(ends - starts).astype(np.int64))
np.savez(os.path.join(DIR, f"normalization_{DOMAIN}_{SPARSE}.npz"),
         obs_min=states.min(0), obs_max=states.max(0),
         action_min=np.full(ADIM, -1, np.float32),
         action_max=np.full(ADIM, 1, np.float32))

nz = (ep_returns > 0).sum()
print(f"sparse demos: {len(ep_returns)} episodes, {len(states)} transitions")
print(f"sparse return  mean {ep_returns.mean():.1f}  min {ep_returns.min():.1f}  "
      f"max {ep_returns.max():.1f}")
print(f"episodes with ANY sparse reward: {nz}/{len(ep_returns)}")
if nz == 0:
    print("WARNING: no episode earned sparse reward -- the critic would have "
          "no signal to learn from and this task pair is unusable as set up.")
