"""Replay buffer for DICE-style off-policy residual RL on LIBERO.

Stores pre-encoded chunk-decision transitions:
    (cond, noise, action, reward, next_cond, done, n_step, mc_return, data_source)
where:
    cond            : (num_enc, hidden) -- BC encoder output, pre-computed at insert
    noise           : (chunk, action_dim) -- noise used at sampling (kept for diagnostic)
    action          : (chunk, action_dim) -- the action chunk that was executed
    reward          : scalar -- accumulated reward over n_step lookahead (or 1-step)
    next_cond       : (num_enc, hidden) -- cond observed AFTER the action chunk
                      (or n_step chunks ahead under use_n_step=True)
    done            : 0/1 -- terminal flag at the lookahead boundary
    n_step          : int -- effective number of chunk-decisions aggregated by this
                      transition (1 if use_n_step=False; up to self.n_step otherwise)
    mc_return       : scalar -- discounted return-to-go from this chunk decision
                      to the end of the episode; used for q_overestimation
    data_source     : 0 = online, 1 = expert (RLPD)

Key methods:
    add_trajectory(trajectory): take a finished episode's list of per-decision
        dicts and write n_step-aggregated transitions + mc_return per row.
    add_online_decision(...)  : 1-step variant for backwards compat with the
        non-n-step path. Computes n_step=1 and mc_return=reward.
    sample(batch_size, expert_ratio=None): returns the official's tuple shape:
        cond, noise, action, reward, next_cond, done, n_steps, mc_return, data_source.

Trajectory format (passed to add_trajectory):
    [{"cond": (num_enc,hidden), "noise": (chunk,A), "action": (chunk,A),
      "reward": float, "done": bool}, ...]  -- chunk-decision granularity, in order.
The last entry's done should be True (terminal), reward = 1.0 on success else 0.0.
"""

from typing import List, Optional, Tuple

import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, max_size: int, cond_shape: Tuple[int, int],
                 horizon: int, action_dim: int, device: str = "cuda",
                 gamma: float = 0.99,
                 use_n_step: bool = False, n_step: int = 1,
                 use_rlpd: bool = False, expert_ratio: float = 0.5,
                 expert_dataset=None):
        self.max_size = int(max_size)
        self.cond_shape = tuple(cond_shape)
        self.horizon = int(horizon)
        self.action_dim = int(action_dim)
        self.device = device
        self.gamma = float(gamma)
        self.use_n_step = bool(use_n_step)
        self.n_step = int(n_step)
        # RLPD knobs (expert dataset wiring is a TODO; hook exists).
        self.use_rlpd = bool(use_rlpd)
        self.expert_ratio = float(expert_ratio)
        self.expert_dataset = expert_dataset

        ne, h = cond_shape
        self.cond = torch.zeros(self.max_size, ne, h, dtype=torch.float32)
        self.next_cond = torch.zeros(self.max_size, ne, h, dtype=torch.float32)
        self.noise = torch.zeros(self.max_size, horizon, action_dim, dtype=torch.float32)
        self.action = torch.zeros(self.max_size, horizon, action_dim, dtype=torch.float32)
        self.reward = torch.zeros(self.max_size, 1, dtype=torch.float32)
        self.done = torch.zeros(self.max_size, 1, dtype=torch.float32)
        self.n_steps = torch.ones(self.max_size, 1, dtype=torch.float32)
        self.mc_return = torch.zeros(self.max_size, 1, dtype=torch.float32)
        self.data_source = torch.zeros(self.max_size, 1, dtype=torch.float32)  # 0=online

        self.ptr = 0
        self.size = 0
        self.num_episodes = 0

    # ------------------------------------------------------------------
    # Single-decision insert (1-step). Kept for back-compat.
    # ------------------------------------------------------------------
    def add(self, cond, noise, action, reward, next_cond, done, data_source: int = 0):
        i = self.ptr
        self.cond[i].copy_(cond)
        self.next_cond[i].copy_(next_cond)
        self.noise[i].copy_(noise)
        self.action[i].copy_(action)
        self.reward[i, 0] = float(reward)
        self.done[i, 0] = float(done)
        self.n_steps[i, 0] = 1.0
        self.mc_return[i, 0] = float(reward)  # single-step return; replaced by add_trajectory if used
        self.data_source[i, 0] = float(data_source)
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    # ------------------------------------------------------------------
    # Trajectory insert (computes n-step returns + mc_return)
    # ------------------------------------------------------------------
    def add_trajectory(self, trajectory: List[dict], data_source: int = 0):
        """Aggregate a whole episode's chunk decisions into n_step-aggregated
        transitions, plus per-row mc_return."""
        T = len(trajectory)
        if T == 0:
            return
        # Per-decision base reward (already 0/1 at chunk granularity from the collector).
        rewards = np.array([float(d["reward"]) for d in trajectory], dtype=np.float32)
        dones = np.array([bool(d["done"]) for d in trajectory], dtype=bool)

        # MC return-to-go (discounted from each decision to end of episode).
        mc = np.zeros(T, dtype=np.float32)
        running = 0.0
        for t in range(T - 1, -1, -1):
            running = rewards[t] + self.gamma * running * (0.0 if dones[t] else 1.0)
            mc[t] = running

        # N-step lookahead returns + next-cond + done flag for each decision.
        n_eff = self.n_step if self.use_n_step else 1
        for t in range(T):
            # Accumulate up to n_eff future rewards (or until episode terminates).
            R = 0.0
            steps_used = 0
            terminated = False
            for k in range(n_eff):
                idx = t + k
                if idx >= T:
                    break
                R += (self.gamma ** k) * rewards[idx]
                steps_used = k + 1
                if dones[idx]:
                    terminated = True
                    break
            # next_cond points to the decision AFTER the last aggregated one.
            tail = t + steps_used
            if terminated or tail >= T:
                next_cond = torch.zeros(self.cond_shape, dtype=torch.float32)
                done_flag = True
            else:
                next_cond = trajectory[tail]["cond"]
                done_flag = False

            i = self.ptr
            self.cond[i].copy_(trajectory[t]["cond"])
            self.next_cond[i].copy_(next_cond)
            self.noise[i].copy_(trajectory[t]["noise"])
            self.action[i].copy_(trajectory[t]["action"])
            self.reward[i, 0] = R
            self.done[i, 0] = float(done_flag)
            self.n_steps[i, 0] = float(steps_used)
            self.mc_return[i, 0] = float(mc[t])
            self.data_source[i, 0] = float(data_source)
            self.ptr = (self.ptr + 1) % self.max_size
            self.size = min(self.size + 1, self.max_size)
        self.num_episodes += 1

    # ------------------------------------------------------------------
    # Sampling (RLPD expert mixing hook; expert_dataset is a TODO wiring)
    # ------------------------------------------------------------------
    def _gather(self, idx: np.ndarray) -> dict:
        idx_t = torch.from_numpy(idx).long()
        d = self.device
        return {
            "cond": self.cond[idx_t].to(d),
            "noise": self.noise[idx_t].to(d),
            "action": self.action[idx_t].to(d),
            "reward": self.reward[idx_t].to(d),
            "next_cond": self.next_cond[idx_t].to(d),
            "done": self.done[idx_t].to(d),
            "n_steps": self.n_steps[idx_t].to(d),
            "mc_return": self.mc_return[idx_t].to(d),
            "data_source": self.data_source[idx_t].to(d),
        }

    def sample(self, batch_size: int, expert_ratio: Optional[float] = None) -> dict:
        """Online-only when expert_dataset is None or use_rlpd=False; otherwise
        symmetric-sample (expert_ratio * B from expert, rest from online).
        For now, expert_dataset wiring is a TODO -- expert_ratio is accepted but
        ignored unless self.expert_dataset is hooked up by the caller."""
        if not self.use_rlpd or self.expert_dataset is None:
            idx = np.random.randint(0, self.size, size=batch_size)
            return self._gather(idx)

        # TODO: wire an expert_dataset.sample(n_exp) that returns the same dict shape.
        # Until then, fall back to online-only.
        idx = np.random.randint(0, self.size, size=batch_size)
        return self._gather(idx)

    def get_total_transitions(self) -> int:
        return self.size

    def __len__(self) -> int:
        return self.size
