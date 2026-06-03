import math

import numpy as np
import torch

from imitation.algos.fm_policy import FlowMatchingPolicy


class FlowMatchingPolicyRL(FlowMatchingPolicy):
    """Flow-matching policy with a stochastic (noised-Euler) sampler that exposes
    per-denoising-step Gaussian log-probabilities, enabling GRPO/PPO fine-tuning.

    The deterministic ODE step ``x_{k+1} = x_k + dt * v(x_k, t_k)`` is replaced by the
    SDE-like step ``x_{k+1} = mu_k + sigma * sqrt(dt) * eps`` with ``mu_k = x_k + dt * v``,
    so each step has a tractable density ``N(x_{k+1}; mu_k, sigma^2 * dt)``. Exploration
    comes from ``sigma``; dropout is disabled (eval mode) so that log-probs are
    reproducible between the sampling pass and the update pass.

    Log-probs live entirely in the normalized [-1, 1] action space. The downstream
    ``normalizer.unnormalize`` (affine) and rotation post-processing (identity for
    axis-angle / abs_action=False) are deterministic and identical for the old and new
    policy, so they cancel in the importance ratio and need no Jacobian term.
    """

    def __init__(self, rl_sigma: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.rl_sigma = rl_sigma

    @property
    def step_var(self):
        """Per-denoising-step Gaussian variance sigma^2 * dt."""
        dt = 1.0 / self.num_inference_steps
        return (self.rl_sigma ** 2) * dt

    @staticmethod
    def _gaussian_logp(x, mu, var):
        # x, mu: (..., chunk, A); var: float scalar. Returns log-density summed over the
        # last two dims (chunk, A) -> (...,). SUM, not mean, to be a valid log-density.
        ll = -0.5 * (((x - mu) ** 2) / var + math.log(2.0 * math.pi * var))
        return ll.sum(dim=(-1, -2))

    @torch.no_grad()
    def sample_actions_stochastic(self, data):
        """Stochastic rollout sampler. Returns the (normalized) action chunk plus all the
        state needed to recompute log-probs differentiably during the PPO update.

        Returns a dict of CPU tensors / arrays:
          action   : (B, chunk, A) np.float32, clamped, normalized space
          cond     : (B, num_enc, hidden)  detached encoder conditioning
          chain    : (B, K+1, chunk, A)    x_0 .. x_K
          mu_old   : (B, K, chunk, A)      behavior-policy means mu_k
          t_grid   : (K,)                  denoising timesteps t_k
          logp_old : (B, K)                behavior-policy per-step log-probs (diagnostic)
        """
        self.eval()
        data = self.preprocess_input(data, train_mode=False)
        cond = self.get_cond(data)
        B, device, dtype = cond.shape[0], cond.device, cond.dtype

        K = self.num_inference_steps
        dt = 1.0 / K
        var = self.step_var
        std = self.rl_sigma * (dt ** 0.5)

        enc_cache = self.velocity_net.forward_enc(cond)
        x = torch.randn(B, self.chunk_size, self.network_action_dim, device=device, dtype=dtype)

        chain, mus, t_grid, logp = [x], [], [], []
        t = torch.zeros(B, device=device, dtype=dtype)
        for _ in range(K):
            v = self.velocity_net.forward_dec(x, t, enc_cache)
            mu = x + dt * v
            x_next = mu + std * torch.randn_like(mu)
            logp.append(self._gaussian_logp(x_next, mu, var))
            chain.append(x_next)
            mus.append(mu)
            t_grid.append(t[0].clone())
            x = x_next
            t = t + dt

        action = torch.clamp(x, -1, 1)
        return {
            "action": action.to(torch.float32).cpu().numpy(),
            "cond": cond.detach().to(torch.float32).cpu(),
            "chain": torch.stack(chain, dim=1).to(torch.float32).cpu(),
            "mu_old": torch.stack(mus, dim=1).to(torch.float32).cpu(),
            "t_grid": torch.stack(t_grid).to(torch.float32).cpu(),
            "logp_old": torch.stack(logp, dim=1).to(torch.float32).cpu(),
        }

    def chain_logprob(self, cond, chain, t_grid):
        """Recompute per-step means and log-probs under the CURRENT policy parameters,
        on the STORED denoising chain. Differentiable through ``velocity_net`` only
        (``cond`` is treated as a fixed input -> the perception encoder is frozen).

        Args:
          cond   : (B, num_enc, hidden)
          chain  : (B, K+1, chunk, A)
          t_grid : (K,)
        Returns:
          logp_new : (B, K)
          mu_new   : (B, K, chunk, A)
        """
        self.eval()  # disable dropout; params still require grad
        K = self.num_inference_steps
        dt = 1.0 / K
        var = self.step_var
        B = cond.shape[0]

        enc_cache = self.velocity_net.forward_enc(cond)
        logps, mus = [], []
        for k in range(K):
            x_k = chain[:, k]
            x_next = chain[:, k + 1]
            t_k = torch.full((B,), float(t_grid[k]), device=cond.device, dtype=cond.dtype)
            v = self.velocity_net.forward_dec(x_k, t_k, enc_cache)
            mu = x_k + dt * v
            logps.append(self._gaussian_logp(x_next, mu, var))
            mus.append(mu)
        return torch.stack(logps, dim=1), torch.stack(mus, dim=1)
