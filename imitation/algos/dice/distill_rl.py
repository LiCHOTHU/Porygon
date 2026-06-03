"""Distilled RL model for LIBERO FM policy (port of DICE-RL-Robot's DistilledRLModel).

Single-process MVP. Uses a frozen BC FlowMatchingPolicy as both:
  (i) the visual encoder that produces `cond` = state embedding
 (ii) the teacher whose action is anchored against via BC-MSE in actor_loss.

Student actor is a flat MLP `(state_emb_flat, noise_flat) -> action_chunk_flat`.
Critic is an ensemble of `(state, noise, action) -> Q(s,z,a)` MLPs with target net + Polyak.

Loss shape (matches DICE-RL-Robot core branch, warmup-only / no soft-Q-filtering / no SI):
  L_actor  = -E_K[Q(s, z_k, a_student(s, z_k))]
             + bc_loss_weight * MSE(a_student(s, z_k), a_teacher(s, z_k))
  L_critic = TD(MSE/Huber/BCE) over ensemble vs target_q
             target_q = r + gamma * (1 - done) * target_critic(s', z', a_student(s', z'))

Skipped (faithful but minimal): dynamics, intrinsic reward, self-imitation, soft-Q
filtering, q-normalization, n-step, multi-z next-noise sampling. Trivial to re-add as
options later. The ladder is to first show the spine works.
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _mlp(dims: List[int], activation: str = "Mish", use_layernorm: bool = True,
         out_activation: str = "Identity") -> nn.Sequential:
    act_cls = {"Mish": nn.Mish, "ReLU": nn.ReLU, "GELU": nn.GELU}[activation]
    out_act = {"Identity": nn.Identity, "Tanh": nn.Tanh, "Sigmoid": nn.Sigmoid}[out_activation]
    layers: List[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            if use_layernorm:
                layers.append(nn.LayerNorm(dims[i + 1]))
            layers.append(act_cls())
    layers.append(out_act())
    return nn.Sequential(*layers)


class DistilledActor(nn.Module):
    """(state, noise) -> action chunk; flat MLP, normalized [-1, 1] action space."""

    def __init__(self, state_dim: int, action_dim: int, horizon_steps: int,
                 hidden_dims: List[int] = (512, 512, 512),
                 activation: str = "Mish", use_layernorm: bool = True):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.horizon_steps = horizon_steps
        in_dim = state_dim + horizon_steps * action_dim
        out_dim = horizon_steps * action_dim
        self.mlp = _mlp([in_dim] + list(hidden_dims) + [out_dim],
                        activation=activation, use_layernorm=use_layernorm,
                        out_activation="Tanh")  # actions live in [-1, 1]

    def forward(self, state: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        B = state.shape[0]
        x = torch.cat([state.view(B, -1), noise.view(B, -1)], dim=-1)
        a = self.mlp(x)
        return a.view(B, self.horizon_steps, self.action_dim)


class DistilledCritic(nn.Module):
    """Ensemble of Q(s, z, a) -> R. Conservative aggregation via min or LCB."""

    def __init__(self, state_dim: int, action_dim: int, horizon_steps: int,
                 hidden_dims: List[int] = (512, 512, 512),
                 ensemble_size: int = 2, q_depends_on_noise: bool = True,
                 conservative: str = "min", lcb_kappa: float = 0.1,
                 activation: str = "Mish", use_layernorm: bool = True):
        super().__init__()
        self.q_depends_on_noise = q_depends_on_noise
        self.conservative = conservative
        self.lcb_kappa = lcb_kappa
        in_extra = 2 if q_depends_on_noise else 1
        in_dim = state_dim + in_extra * horizon_steps * action_dim
        self.qs = nn.ModuleList([
            _mlp([in_dim] + list(hidden_dims) + [1], activation=activation,
                 use_layernorm=use_layernorm, out_activation="Identity")
            for _ in range(ensemble_size)
        ])

    def _features(self, state: torch.Tensor, noise: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        B = state.shape[0]
        parts = [state.view(B, -1)]
        if self.q_depends_on_noise:
            parts.append(noise.view(B, -1))
        parts.append(action.view(B, -1))
        return torch.cat(parts, dim=-1)

    def forward(self, state: torch.Tensor, noise: torch.Tensor, action: torch.Tensor,
                return_all: bool = False) -> torch.Tensor:
        x = self._features(state, noise, action)
        qs = [q(x) for q in self.qs]
        if return_all:
            return qs
        q_stack = torch.stack(qs, dim=0)
        if self.conservative == "min":
            return q_stack.min(dim=0).values
        if self.conservative == "lcb":
            return q_stack.mean(dim=0) - self.lcb_kappa * q_stack.std(dim=0)
        if self.conservative == "mean":
            return q_stack.mean(dim=0)
        raise ValueError(f"Unknown conservative method: {self.conservative}")


class DistilledRLModel(nn.Module):
    """Wraps DistilledActor + DistilledCritic ensemble + target critic + BC teacher.

    The visual encoder is the frozen BC FM policy: state = flatten(get_cond(data)).
    The teacher action a_teacher is obtained by running the BC FM Euler sampler from
    the SAME starting noise as the student's `noise` input -- so the BC-MSE anchor is
    a meaningful pointwise distance (both conditioned on the same x_0).

    External API:
      forward / get_action(state_emb, noise) -> a_student (B, H, A)
      loss(state_emb, noise, action, next_state_emb, reward, done, gamma) -> dict
      update_target_critic(tau)
    """

    def __init__(self, state_dim: int, action_dim: int, horizon_steps: int,
                 actor_hidden: List[int] = (512, 512, 512),
                 critic_hidden: List[int] = (512, 512, 512),
                 ensemble_size: int = 2, q_depends_on_noise: bool = True,
                 conservative: str = "min", lcb_kappa: float = 0.1,
                 td_loss: str = "huber",
                 bc_loss_weight: float = 100.0,
                 num_multi_z: int = 4,
                 clip_action: bool = True,
                 device: str = "cuda"):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.horizon_steps = horizon_steps
        self.bc_loss_weight = bc_loss_weight
        self.num_multi_z = num_multi_z
        self.td_loss = td_loss
        self.clip_action = clip_action
        self.device = device

        self.actor = DistilledActor(
            state_dim=state_dim, action_dim=action_dim, horizon_steps=horizon_steps,
            hidden_dims=actor_hidden, use_layernorm=True,
        ).to(device)
        self.critic = DistilledCritic(
            state_dim=state_dim, action_dim=action_dim, horizon_steps=horizon_steps,
            hidden_dims=critic_hidden, ensemble_size=ensemble_size,
            q_depends_on_noise=q_depends_on_noise,
            conservative=conservative, lcb_kappa=lcb_kappa,
        ).to(device)
        self.target_critic = DistilledCritic(
            state_dim=state_dim, action_dim=action_dim, horizon_steps=horizon_steps,
            hidden_dims=critic_hidden, ensemble_size=ensemble_size,
            q_depends_on_noise=q_depends_on_noise,
            conservative=conservative, lcb_kappa=lcb_kappa,
        ).to(device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        for p in self.target_critic.parameters():
            p.requires_grad = False

        # teacher_fn(state_emb, noise) -> a_teacher in [-1,1] normalized action space.
        # Attached externally so this module stays light. Expected: torch.no_grad inside.
        self._teacher_fn = None

    def attach_teacher(self, teacher_fn):
        """teacher_fn: callable(state_emb (B,S), noise (B,H,A)) -> action (B,H,A) under no_grad."""
        self._teacher_fn = teacher_fn

    def get_action(self, state: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        a = self.actor(state, noise)
        if self.clip_action:
            a = torch.clamp(a, -1.0, 1.0)
        return a

    # ---- losses ----
    def _critic_td_loss(self, q_pred: torch.Tensor, target_q: torch.Tensor) -> torch.Tensor:
        if self.td_loss == "mse":
            return F.mse_loss(q_pred, target_q)
        if self.td_loss == "huber":
            return F.smooth_l1_loss(q_pred, target_q, beta=1.0)
        if self.td_loss == "bce":
            assert (target_q >= 0).all() and (target_q <= 1).all(), \
                f"BCE TD requires targets in [0,1]; got [{target_q.min():.3f},{target_q.max():.3f}]"
            return F.binary_cross_entropy_with_logits(q_pred, target_q)
        raise ValueError(f"Unknown td_loss: {self.td_loss}")

    def critic_loss(self, state: torch.Tensor, noise: torch.Tensor, action: torch.Tensor,
                    target_q: torch.Tensor) -> Dict[str, torch.Tensor]:
        qs = self.critic(state, noise, action, return_all=True)
        per_q = [self._critic_td_loss(q, target_q) for q in qs]
        total = sum(per_q)
        out = {"critic_loss": total}
        for i, q in enumerate(qs):
            out[f"q{i}_loss"] = per_q[i]
            out[f"q{i}_mean"] = q.mean().detach()
        return out

    def actor_loss(self, state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Multi-z actor loss: sample K noise vectors per state, average -Q and BC-MSE.

        L_actor = -E_K[Q(s, z_k, a_student(s, z_k))]
                  + bc_weight * E_K[||a_student - a_teacher||^2]
        """
        assert self._teacher_fn is not None, "Call attach_teacher(teacher_fn) before loss()."
        B = state.shape[0]
        K = self.num_multi_z

        # Replicate state K times while preserving the unflat (B, num_enc, hidden) shape:
        # the teacher needs it, while actor/critic flatten internally via view(B, -1).
        if state.dim() == 3:
            ne, h = state.shape[1], state.shape[2]
            state_rep = state.unsqueeze(1).expand(-1, K, -1, -1).reshape(B * K, ne, h)
        else:
            state_rep = state.unsqueeze(1).expand(-1, K, -1).reshape(B * K, -1)
        noise_rep = torch.randn(B * K, self.horizon_steps, self.action_dim, device=self.device)

        a_student = self.get_action(state_rep, noise_rep)  # (B*K, H, A)
        with torch.no_grad():
            a_teacher = self._teacher_fn(state_rep, noise_rep)  # (B*K, H, A)

        q_pred = self.critic(state_rep, noise_rep, a_student)  # (B*K, 1)
        q_loss = -q_pred.mean()
        bc_mse = ((a_student - a_teacher) ** 2).mean()

        total = q_loss + self.bc_loss_weight * bc_mse
        with torch.no_grad():
            residual_norm = ((a_student - a_teacher) ** 2).mean().sqrt()
        return {
            "actor_total": total,
            "actor_q_loss": q_loss,
            "actor_bc_loss": bc_mse,
            "residual_norm": residual_norm,
            "q_actor_mean": q_pred.mean().detach(),
        }

    def loss(self, state: torch.Tensor, noise: torch.Tensor, action: torch.Tensor,
             next_state: torch.Tensor, reward: torch.Tensor, done: torch.Tensor,
             gamma: float = 0.99) -> Dict[str, torch.Tensor]:
        """One joint forward producing actor_total + critic_loss + diagnostics."""
        B = state.shape[0]

        # ----- target_q (no grad) -----
        with torch.no_grad():
            next_noise = torch.randn(B, self.horizon_steps, self.action_dim, device=self.device)
            next_action = self.get_action(next_state, next_noise)
            target_next_q = self.target_critic(next_state, next_noise, next_action)
            target_q = reward + gamma * (1.0 - done) * target_next_q

        critic_d = self.critic_loss(state, noise, action, target_q)
        actor_d = self.actor_loss(state)

        out = {**critic_d, **actor_d,
               "target_q_mean": target_q.mean().detach(),
               "reward_mean": reward.mean().detach(),
               "done_frac": done.mean().detach()}
        return out

    def update_target_critic(self, tau: float = 0.005):
        with torch.no_grad():
            for tp, p in zip(self.target_critic.parameters(), self.critic.parameters()):
                tp.data.mul_(1.0 - tau).add_(tau * p.data)
