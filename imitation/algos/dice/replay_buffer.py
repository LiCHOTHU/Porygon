"""Ring replay buffer for DICE-style off-policy RL on LIBERO.

Stores pre-encoded transitions:
    (cond, noise, action, reward, next_cond, done)
where `cond` is the BC encoder's output of shape (num_enc, hidden) -- pre-computed
once at insert time since the visual encoder is frozen. Saves the cost of re-encoding
during every grad step.

The buffer lives on CPU; sample() returns tensors moved to `device`. For the MVP we
implement online-only sampling. RLPD-style demo mixing is a TODO; the seam is the
`expert_dataset` arg + `expert_ratio` arg in sample().

Storage:
  cond_buf      : (max_size, num_enc, hidden) float32   ~ 10kB / row at num_enc=10, h=256
  noise_buf     : (max_size, horizon, action_dim) float32
  action_buf    : (max_size, horizon, action_dim) float32
  reward_buf    : (max_size, 1) float32
  next_cond_buf : (max_size, num_enc, hidden) float32
  done_buf      : (max_size, 1) float32
"""

from typing import Optional, Tuple

import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, max_size: int, cond_shape: Tuple[int, int],
                 horizon: int, action_dim: int, device: str = "cuda"):
        self.max_size = int(max_size)
        self.cond_shape = tuple(cond_shape)
        self.horizon = int(horizon)
        self.action_dim = int(action_dim)
        self.device = device

        ne, h = cond_shape
        self.cond = torch.zeros(self.max_size, ne, h, dtype=torch.float32)
        self.next_cond = torch.zeros(self.max_size, ne, h, dtype=torch.float32)
        self.noise = torch.zeros(self.max_size, horizon, action_dim, dtype=torch.float32)
        self.action = torch.zeros(self.max_size, horizon, action_dim, dtype=torch.float32)
        self.reward = torch.zeros(self.max_size, 1, dtype=torch.float32)
        self.done = torch.zeros(self.max_size, 1, dtype=torch.float32)

        self.ptr = 0
        self.size = 0

    def add(self, cond, noise, action, reward, next_cond, done):
        """Add ONE transition. All tensors are CPU; shapes:
            cond      (num_enc, hidden)
            noise     (horizon, action_dim)
            action    (horizon, action_dim)
            reward    scalar
            next_cond (num_enc, hidden)
            done      bool / 0|1
        """
        i = self.ptr
        self.cond[i].copy_(cond)
        self.next_cond[i].copy_(next_cond)
        self.noise[i].copy_(noise)
        self.action[i].copy_(action)
        self.reward[i, 0] = float(reward)
        self.done[i, 0] = float(done)
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size: int, expert_ratio: Optional[float] = None):
        """Online-only sample for now. `expert_ratio` reserved for RLPD."""
        idx = np.random.randint(0, self.size, size=batch_size)
        idx_t = torch.from_numpy(idx).long()
        d = self.device
        return {
            "cond": self.cond[idx_t].to(d),
            "noise": self.noise[idx_t].to(d),
            "action": self.action[idx_t].to(d),
            "reward": self.reward[idx_t].to(d),
            "next_cond": self.next_cond[idx_t].to(d),
            "done": self.done[idx_t].to(d),
            "data_source": torch.zeros(batch_size, 1, device=d),  # 0=online
        }

    def __len__(self):
        return self.size
