"""Critic-free GRPO-style advantage estimation and a per-denoising-step PPO-clip loss
for the noised flow-matching policy.

Key choices (see plan):
  - Advantage: group-normalized outcome reward. Group = G rollouts from the SAME init
    state. Centered (R - mean_group) by default; optionally /(std + eps). Broadcast to
    every chunk-decision and every denoising step of the trajectory.
  - Ratio: computed per denoising step (DPPO trick) directly from the squared-error
    difference, so the Gaussian normalizing constant cancels exactly (no fp blow-up):
        log_ratio_k = (||x_next - mu_old||^2 - ||x_next - mu_new||^2) / (2 * var)
    then clipped to +/- log_ratio_clip before exp.
"""

import torch


def compute_grpo_advantages(rewards, group_ids, std_normalize=False, eps=1e-4):
    """Group-normalized outcome advantages.

    Args:
      rewards   : (N,) tensor, terminal return per trajectory (e.g. 0/1 success).
      group_ids : (N,) long tensor, the GRPO group (same-init-state) each trajectory belongs to.
      std_normalize : if True divide centered reward by (group std + eps).
    Returns:
      advantages : (N,) tensor.
    """
    rewards = rewards.float()
    advantages = torch.zeros_like(rewards)
    for g in torch.unique(group_ids):
        m = group_ids == g
        r = rewards[m]
        centered = r - r.mean()
        if std_normalize:
            centered = centered / (r.std(unbiased=False) + eps)
        advantages[m] = centered
    return advantages


def ppo_clip_loss(mu_new, mu_old, x_next, advantages, var,
                  clip_eps=0.2, log_ratio_clip=20.0, step_mask=None,
                  clip_ratio_low=None, clip_ratio_high=None):
    """Per-denoising-step PPO-clipped surrogate loss.

    Matches SimpleVLA-RL's compute_policy_loss (asymmetric clip): the lower bound
    is ``1 - clip_ratio_low`` and the upper bound is ``1 + clip_ratio_high``.
    When both are None we fall back to symmetric clipping at ``clip_eps`` (the
    original behavior, retained for back-compat).

    Args:
      mu_new          : (N, K, chunk, A) means under the current policy (requires grad).
      mu_old          : (N, K, chunk, A) means under the behavior policy (detached).
      x_next          : (N, K, chunk, A) realized next states x_{k+1} (detached).
      advantages      : (N,) per-trajectory advantage, broadcast over K steps.
      var             : float, per-step Gaussian variance sigma^2 * dt.
      clip_eps        : symmetric clip width (used iff clip_ratio_low/high are None).
      clip_ratio_low  : lower-side clip width; lower bound = 1 - clip_ratio_low.
      clip_ratio_high : upper-side clip width; upper bound = 1 + clip_ratio_high.
      step_mask       : optional (N, K) {0,1} mask (e.g. zero out steps past episode end).
    Returns:
      loss : scalar
      info : dict of diagnostics
    """
    if clip_ratio_low is None:
        clip_ratio_low = clip_eps
    if clip_ratio_high is None:
        clip_ratio_high = clip_eps

    sq_old = ((x_next - mu_old) ** 2).sum(dim=(-1, -2))   # (N, K)
    sq_new = ((x_next - mu_new) ** 2).sum(dim=(-1, -2))   # (N, K)
    log_ratio = (sq_old - sq_new) / (2.0 * var)            # (N, K)
    log_ratio = torch.clamp(log_ratio, -log_ratio_clip, log_ratio_clip)
    ratio = torch.exp(log_ratio)

    adv = advantages.to(log_ratio.dtype).unsqueeze(1)      # (N, 1) -> broadcast over K
    surr1 = ratio * adv
    surr2 = torch.clamp(ratio, 1.0 - clip_ratio_low, 1.0 + clip_ratio_high) * adv
    surrogate = torch.minimum(surr1, surr2)                # (N, K)

    # Bounded KL proxy 0.5 * E[log_ratio^2] (>=0), used for early-stopping the update.
    kl_per = 0.5 * log_ratio ** 2
    # Two-sided clipfrac: ratio fell out of [1-low, 1+high].
    clipped = (ratio < 1.0 - clip_ratio_low) | (ratio > 1.0 + clip_ratio_high)
    if step_mask is None:
        loss = -surrogate.mean()
        clipfrac = clipped.float().mean()
        approx_kl = kl_per.mean()
        ratio_mean = ratio.mean()
    else:
        m = step_mask.to(surrogate.dtype)
        denom = m.sum().clamp(min=1.0)
        loss = -(surrogate * m).sum() / denom
        clipfrac = (clipped.float() * m).sum() / denom
        approx_kl = (kl_per * m).sum() / denom
        ratio_mean = (ratio * m).sum() / denom

    info = {
        "loss": float(loss.detach()),
        "ratio_mean": float(ratio_mean.detach()),
        "approx_kl": float(approx_kl.detach()),
        "clipfrac": float(clipfrac.detach()),
        "log_ratio_abs_max": float(log_ratio.detach().abs().max()),
    }
    return loss, info
