"""Baseline regularizers for visuomotor policies: mixup, SAM.

These exist so the spectral penalty is compared against what a practitioner would
actually reach for, not only against an unregularized policy. On CIFAR-10 that
distinction mattered a great deal: our penalty beat plain training by 8.5 points
under label noise and *lost to Mixup by 5.9*, which only became visible once
Mixup was in the table.

Dropout needs nothing here -- it is already a config knob on the policy.
"""

from __future__ import annotations

import torch
from torch import Tensor

__all__ = ["maybe_mixup", "sam_ascent", "sam_descend"]


def _mix_tensor(x: Tensor, index: Tensor, lam: float) -> Tensor:
    return lam * x + (1.0 - lam) * x[index]


def maybe_mixup(data: dict, actions: Tensor, alpha: float):
    """Blend observations and action chunks with one shared coefficient.

    Note what this assumes. In classification, mixing inputs and *labels* is
    justified by the labels being a simplex -- a 70/30 blend is a valid soft
    target. For a policy the target is a continuous action chunk, and blending
    asserts the action is locally **linear** in the observation. Halfway between
    "reach left" and "reach right" is not generally a valid action; it may be a
    collision. So this is a stronger assumption here than in its original
    setting, and it is reported as the weakest of the baseline arms rather than
    presented as an equal comparison.

    Returns ``(data, actions, lam)`` with ``lam = 1`` when mixup is off.
    """
    if alpha <= 0:
        return data, actions, 1.0

    batch = actions.shape[0]
    lam = float(torch._sample_dirichlet(torch.tensor([alpha, alpha]))[0])
    index = torch.randperm(batch, device=actions.device)

    mixed = dict(data)
    obs = mixed.get("obs")
    if isinstance(obs, dict):
        mixed_obs = {}
        for key, value in obs.items():
            # only blend float observations; indices/embeddings are left alone
            mixed_obs[key] = (
                _mix_tensor(value, index, lam)
                if torch.is_floating_point(value) and value.shape[0] == batch
                else value
            )
        mixed["obs"] = mixed_obs
    return mixed, _mix_tensor(actions, index, lam), lam


@torch.no_grad()
def sam_ascent(params: list[torch.nn.Parameter], rho: float) -> list[Tensor]:
    """Step to the worst point in the rho-ball; returns the perturbations to undo.

    Requires ``.grad`` already populated by a backward on the clean loss. The
    caller re-evaluates the loss at the perturbed point, then calls
    :func:`sam_descend` before the optimizer step.
    """
    grads = [p.grad for p in params if p.grad is not None]
    if not grads:
        return []
    norm = torch.norm(torch.stack([g.norm(p=2) for g in grads]), p=2)
    scale = rho / (norm + 1e-12)
    perturbations: list[Tensor] = []
    for p in params:
        if p.grad is None:
            perturbations.append(torch.zeros((), device=p.device))
            continue
        e = p.grad * scale
        p.add_(e)
        perturbations.append(e)
    return perturbations


@torch.no_grad()
def sam_descend(params: list[torch.nn.Parameter], perturbations: list[Tensor]) -> None:
    """Undo the ascent, leaving the gradient measured at the perturbed point."""
    for p, e in zip(params, perturbations):
        if e.ndim > 0:
            p.sub_(e)
