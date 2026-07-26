"""PR2 — pure field utilities for the two-field residual actor update.

Two fields, both in the actor's NORMALIZED action space (the space the residual,
the critic, and the drifting-BC training all live in):

  * V_BC  (compute_drift_field): the drifting-model transport field. Same
    softmax-softmax affinity kernel, same multi-radius R_list, same numerical
    stabilizers as imitation/algos/rl/grd.drift_loss / PolicyDrifting._drift_loss
    (NOT a new generic RBF). Pulls the policy's particles toward the positive
    set (frozen BC) and repels from the negative set (current density).
  * V_Q   (compute_q_gradient_field): grad_a of the SAME scalar the DICE actor
    optimizes (conservative ensemble Q, optionally q-normalized), then norm-clipped
    (NOT direction-normalized — clipping preserves "small gradient => small step").

clip_field_norm caps a field's per-particle L2 norm. aggregate_q mirrors the
critic's conservative aggregation.

Field convention: transported = query + field (field already in action units).

Unit tests in scripts/test_drift_field.py:
  * P+ == P- (with self-handling)  => V_BC ~= 0   (catches self-term inconsistency)
  * translation / permutation equivariance
  * perturbed-distribution recovery: query=bc+delta => mean_i V_i . delta < 0
  * no gradient leaks through field construction
"""
from dataclasses import dataclass
from typing import Optional, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor

R_LIST_DEFAULT = (0.02, 0.05, 0.2)   # matches grd.R_LIST / the drift base


@dataclass
class DriftFieldOutput:
    field: Tensor            # (B, Kq, S) — transported = query + field
    positive_field: Tensor   # (B, Kq, S) attraction-only contribution (diagnostic)
    negative_field: Tensor   # (B, Kq, S) repulsion-only contribution (diagnostic)
    positive_norm: Tensor    # (B,) mean per-particle norm of positive_field
    negative_norm: Tensor    # (B,) mean per-particle norm of negative_field


@dataclass
class QFieldOutput:
    field: Tensor            # (B, Kq, S) clipped grad_a of (Q/q_scale)
    q_value: Tensor          # (B, Kq) conservative Q at the query actions
    raw_norm: Tensor         # (B, Kq) per-particle norm before clipping
    clipped_norm: Tensor     # (B, Kq) per-particle norm after clipping


def _flatten(a: Tensor) -> Tensor:
    """(B,K,H,A) or (B,K,S) -> (B,K,S)."""
    return a.reshape(a.shape[0], a.shape[1], -1) if a.dim() == 4 else a


def _cdist(x: Tensor, y: Tensor, eps: float = 1e-8) -> Tensor:
    xy = torch.einsum("bnd,bmd->bnm", x, y)
    xn = torch.einsum("bnd,bnd->bn", x, x).unsqueeze(-1)
    yn = torch.einsum("bmd,bmd->bm", y, y).unsqueeze(-2)
    return (xn + yn - 2 * xy).clamp_min(eps).sqrt()


def clip_field_norm(field: Tensor, max_norm: Optional[float], eps: float = 1e-8) -> Tensor:
    """Cap per-particle L2 norm at max_norm (no-op if None). field (B,K,S)."""
    if max_norm is None:
        return field
    norm = field.norm(dim=-1, keepdim=True)                       # (B,K,1)
    scale = torch.clamp(max_norm / (norm + eps), max=1.0)
    return field * scale


def aggregate_q(q_heads: Sequence[Tensor], mode: str, lcb_kappa: float = 1.0) -> Tensor:
    """q_heads: list of (N,1). Returns (N,1) conservative scalar. Mirrors DistilledCritic."""
    q = torch.stack(list(q_heads), dim=0)                         # (M, N, 1)
    if mode == "min":
        return q.min(dim=0).values
    if mode == "mean":
        return q.mean(dim=0)
    if mode == "lcb":
        return q.mean(dim=0) - lcb_kappa * q.std(dim=0)
    raise ValueError(f"unknown aggregation: {mode}")


@torch.no_grad()
def compute_drift_field(query_actions: Tensor,
                        positive_actions: Tensor,
                        negative_actions: Tensor,
                        radii: Sequence[float] = R_LIST_DEFAULT,
                        w_pos: Optional[Tensor] = None,
                        w_neg: Optional[Tensor] = None,
                        exclude_negative_self: bool = True,
                        exclude_positive_self: bool = False,
                        normalize_per_radius: bool = True,
                        eps: float = 1e-8) -> DriftFieldOutput:
    """Drifting-model transport field at the query particles.

    query_actions    : (B, Kq, H, A) or (B, Kq, S)  — points the field acts on
    positive_actions : (B, Kp, ...)  attractors  (detached), weights w_pos (B,Kp)
    negative_actions : (B, Kn, ...)  repulsors / current density, weights w_neg (B,Kn)
    exclude_negative_self : when Kn == Kq (negative is the query set itself), mask the
                            diagonal self-pair so a particle does not repel itself.

    Faithful port of grd.drift_loss's internal field V: same softmax-softmax affinity
    `sqrt(softmax(-d/R,-1) * softmax(-d/R,-2))`, same per-R unit-normalized force sum,
    same global distance scaling. Returns field s.t. transported = query + field.
    """
    q = _flatten(query_actions)
    pos = _flatten(positive_actions)
    neg = _flatten(negative_actions)
    B, Kq, S = q.shape
    Kp, Kn = pos.shape[1], neg.shape[1]
    dev, dt = q.device, q.dtype
    if w_pos is None:
        w_pos = q.new_ones(B, Kp)
    if w_neg is None:
        w_neg = q.new_ones(B, Kn)

    # targets ordered [negative (repel) | positive (attract)].
    targets = torch.cat([neg, pos], dim=1)                        # (B, Kn+Kp, S)
    tw = torch.cat([w_neg, w_pos], dim=1)                         # (B, Kn+Kp)
    dist = _cdist(q, targets)                                     # (B, Kq, Kn+Kp)

    # global distance scaling (matches grd.drift_loss)
    scale = (dist * tw.unsqueeze(1)).mean() / tw.mean().clamp_min(1e-8)
    scale_in = (scale / (S ** 0.5)).clamp_min(1e-3)
    q_s = q / scale_in
    tgt_s = targets / scale_in
    dn = dist / scale.clamp_min(1e-3)

    # exclude self-pairs when a target block IS the query set (symmetric handling:
    # masking only one block leaves uncancelled self-attraction -> nonzero field even
    # when positive and negative distributions are identical).
    if exclude_negative_self and Kn == Kq:
        eye = torch.eye(Kq, device=dev, dtype=dt)
        dn = dn + F.pad(eye, (0, Kp)).unsqueeze(0) * 100.0          # mask neg block diag
    if exclude_positive_self and Kp == Kq:
        eye = torch.eye(Kq, device=dev, dtype=dt)
        dn = dn + F.pad(eye, (Kn, 0)).unsqueeze(0) * 100.0          # mask pos block diag

    split = Kn                                                   # [:split]=neg, [split:]=pos
    V = torch.zeros_like(q_s)
    Vpos = torch.zeros_like(q_s)
    Vneg = torch.zeros_like(q_s)
    for R in radii:
        lg = -dn / R
        aff = (torch.softmax(lg, -1) * torch.softmax(lg, -2)).clamp_min(1e-6).sqrt()
        aff = aff * tw.unsqueeze(1)
        an, ap = aff[:, :, :split], aff[:, :, split:]            # neg / pos affinities
        rc_neg = -an * ap.sum(-1, keepdim=True)                  # (B,Kq,Kn) repel
        rc_pos = ap * an.sum(-1, keepdim=True)                   # (B,Kq,Kp) attract
        rc = torch.cat([rc_neg, rc_pos], dim=-1)
        force = torch.einsum("biy,byx->bix", rc, tgt_s) - rc.sum(-1, keepdim=True) * q_s
        # normalize_per_radius=True (faithful grd): unit-RMS force per scale -> the
        # field never vanishes (constant pull). False (raw): force keeps its true
        # magnitude -> a RESTORING field that -> 0 as the distributions match.
        nrm = (force ** 2).mean().clamp_min(1e-8).sqrt() if normalize_per_radius else 1.0
        V = V + force / nrm
        # diagnostic split (same normalizer so they add to ~V)
        Vneg = Vneg + (torch.einsum("biy,byx->bix", rc_neg, tgt_s[:, :split])
                       - rc_neg.sum(-1, keepdim=True) * q_s) / nrm
        Vpos = Vpos + (torch.einsum("biy,byx->bix", rc_pos, tgt_s[:, split:])) / nrm

    field = V * scale_in
    pos_field = Vpos * scale_in
    neg_field = Vneg * scale_in
    return DriftFieldOutput(
        field=field.reshape(query_actions.shape),
        positive_field=pos_field.reshape(query_actions.shape),
        negative_field=neg_field.reshape(query_actions.shape),
        positive_norm=pos_field.norm(dim=-1).mean(dim=1),
        negative_norm=neg_field.norm(dim=-1).mean(dim=1),
    )


def compute_q_gradient_field(critic,
                             states: Tensor,
                             noise: Tensor,
                             actions: Tensor,
                             q_scale,
                             max_norm: Optional[float] = None,
                             eps: float = 1e-8) -> QFieldOutput:
    """grad_a of the conservative critic scalar Q(states, noise, a), divided by
    q_scale (match the DICE actor's q-normalization), then norm-clipped.

    actions: (N, H, A). critic.forward(state, noise, action) returns the SAME
    conservative aggregation the DICE actor differentiates (min/lcb/mean).
    Critic params are frozen here (we only need grad wrt actions); caller restores
    requires_grad for critic training.
    """
    saved = [p.requires_grad for p in critic.parameters()]
    for p in critic.parameters():
        p.requires_grad_(False)
    try:
        a = actions.detach().clone().requires_grad_(True)
        q = critic(states, noise, a)                             # (N,1) conservative
        obj = (q / q_scale).sum()
        grad = torch.autograd.grad(obj, a, only_inputs=True)[0].detach()  # (N,H,A)
    finally:
        for p, s in zip(critic.parameters(), saved):
            p.requires_grad_(s)

    N = grad.shape[0]
    g = grad.reshape(N, 1, -1)                                   # (N,1,S) for clip helper
    raw_norm = g.norm(dim=-1)                                    # (N,1)
    clipped = clip_field_norm(g, max_norm, eps)
    clipped_norm = clipped.norm(dim=-1)
    return QFieldOutput(
        field=clipped.reshape(actions.shape),
        q_value=q.detach().reshape(N),
        raw_norm=raw_norm.reshape(N),
        clipped_norm=clipped_norm.reshape(N),
    )
