"""Spectral Information Bottleneck for visuomotor policies.

The penalty charges a spectral band of the visual representation only when it
carries variation the *action* does not explain **and** the policy's predicted
velocity moves with it:

    R = sum_k sqrt( S_k * V_k + eps )

    V_k = E|| P_k (Z - h_k(C)) ||^2     variation not explained by the conditioning
    S_k = E|| J_v(Z) P_k ||_F^2         how much the velocity head leans on band k

Why this task.  A visuomotor policy is free to steer on table texture, lighting,
or a distractor object rather than on object shape and position -- and those
nuisances are spatially band-localized, high frequency against the low-frequency
layout that actually determines the action.  Adding a distractor or shifting the
camera then breaks the policy.  That is a real, pre-existing generalization gap
(``distractor_objects`` / ``cam_shift`` in the task config), and unlike Mixup,
SAM or dropout, a band-resolved penalty can express "rely less on this band".

Two departures from the classification form, both forced by the setting:

* **No ``Q`` projector.**  ``Q = I - (1/|Y|)11^T`` removes the softmax gauge --
  adding a constant to every logit changes nothing.  Flow matching is a squared
  error on velocities and has no such invariance, so ``Q = I``.
* **``h_k(C)`` is learned, and conditions on the flow timestep.**  There are no
  class means for a continuous action chunk.  The conditioner is a cross-fitted
  linear map, and it takes ``t`` as well as the action: the representation is
  encoded from an observation whose relationship to the velocity target depends
  on ``t``, and a conditioner blind to ``t`` measures that schedule as though it
  were residual.  This is the same trap that made an earlier MAE attempt report
  ``V/E = 1.13`` -- the fitted mean predicting *worse* than the mean -- until the
  masking pattern was included in the conditioning.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor

__all__ = ["SpectralBands", "ActionConditioner", "BandEMA", "sib_penalty",
           "use_double_backward_safe_attention"]


def use_double_backward_safe_attention() -> None:
    """Force the math SDPA kernel, which supports second-order gradients.

    The velocity net is a DiT, and PyTorch's fused scaled-dot-product-attention
    kernels (flash / mem-efficient) implement no double backward.  The SIB penalty
    differentiates through a Jacobian, so training dies with "derivative for
    aten::_scaled_dot_product_efficient_attention_backward is not implemented".
    The math backend is slower but differentiable twice.
    """
    if hasattr(torch.backends.cuda, "enable_flash_sdp"):
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)


def dct_matrix(n: int, dtype: torch.dtype = torch.float64) -> Tensor:
    """Orthonormal DCT-II matrix: rows are frequencies, ``D @ D.T == I``."""
    freq = torch.arange(n, dtype=dtype).unsqueeze(1)
    pos = torch.arange(n, dtype=dtype).unsqueeze(0)
    mat = torch.cos(math.pi * (2.0 * pos + 1.0) * freq / (2.0 * n))
    mat *= math.sqrt(2.0 / n)
    mat[0] *= math.sqrt(0.5)
    return mat


class SpectralBands(nn.Module):
    """Orthonormal DCT and a radial equal-count band partition of the feature map.

    The projectors are exact by construction: disjoint 0/1 masks that sum to the
    identity.  Nothing here assumes the band coefficients are independent -- they
    are not -- only that the partition is geometric.
    """

    def __init__(self, height: int, width: int, n_bands: int = 8):
        super().__init__()
        self.height, self.width, self.n_bands = height, width, n_bands
        self.register_buffer("dct_h", dct_matrix(height).float())
        self.register_buffer("dct_w", dct_matrix(width).float())

        u = torch.arange(height, dtype=torch.float64).unsqueeze(1) / height
        v = torch.arange(width, dtype=torch.float64).unsqueeze(0) / width
        radius = torch.sqrt(u**2 + v**2).flatten()
        order = torch.argsort(radius, stable=True)
        band_of = torch.empty(height * width, dtype=torch.long)
        for band, chunk in enumerate(torch.tensor_split(order, n_bands)):
            band_of[chunk] = band
        masks = torch.zeros(n_bands, height * width)
        masks[band_of, torch.arange(height * width)] = 1.0
        self.register_buffer("masks", masks.reshape(n_bands, height, width))

    def dct(self, h: Tensor) -> Tensor:
        return torch.einsum("ij,bcjk,lk->bcil", self.dct_h, h, self.dct_w)

    def idct(self, z: Tensor) -> Tensor:
        return torch.einsum("ij,bcil,lk->bcjk", self.dct_h, z, self.dct_w)

    def band_energy(self, t: Tensor) -> Tensor:
        """``||P_k t||^2`` per sample, shape ``(batch, n_bands)``."""
        return torch.einsum("bchw,khw->bk", t * t, self.masks)


class ActionConditioner(nn.Module):
    """``h_k(C)`` for a continuous action chunk, fitted out of fold.

    ``C`` is a fixed low-dimensional projection of the action chunk together with
    the flow timestep.  Three choices keep this from zeroing the penalty for free:

    * the map is **linear** and the projection is **fixed** (seeded, never
      trained), so capacity is one number rather than whatever a network decides;
    * it is **solved in closed form**, so it never sees the encoder's gradient and
      cannot collude with it;
    * it is fitted **out of fold** -- solved on one half of the batch, applied to
      the other -- so nothing is ever scored by a fit that saw it.

    ``code_dim`` must stay well under the fold size or the fit interpolates and
    ``V_k`` collapses for reasons that have nothing to do with the representation.
    """

    def __init__(self, action_dim: int, code_dim: int = 8, ridge: float = 1e-2,
                 n_folds: int = 2, seed: int = 0):
        super().__init__()
        self.code_dim = code_dim
        self.ridge = ridge
        self.n_folds = n_folds
        generator = torch.Generator().manual_seed(seed)
        self.register_buffer(
            "projection", torch.randn(action_dim, code_dim, generator=generator) / math.sqrt(action_dim)
        )

    def code(self, actions: Tensor, t: Tensor) -> Tensor:
        """[projected action chunk, t, t^2, 1].

        ``t`` enters explicitly: the velocity target depends on the flow time, so a
        conditioner blind to ``t`` would score that dependence as residual variation.
        """
        a = actions.flatten(1) @ self.projection
        t = t.reshape(-1, 1).to(a.dtype)
        return torch.cat([a, t, t**2, torch.ones_like(t)], dim=1)

    def residual(self, z: Tensor, actions: Tensor, t: Tensor) -> Tensor:
        c = self.code(actions, t).detach()
        z_flat = z.flatten(1)
        prediction = torch.zeros_like(z_flat.detach())
        folds = torch.arange(z.shape[0], device=z.device) % self.n_folds
        for f in range(self.n_folds):
            held, fit_on = folds == f, folds != f
            if int(fit_on.sum()) <= c.shape[1] or not held.any():
                continue  # too few points to fit honestly; leave that fold's prediction at 0
            gram = c[fit_on].T @ c[fit_on]
            eye = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
            scale = torch.diagonal(gram).mean().clamp_min(1e-12)
            weight = torch.linalg.solve(
                gram + self.ridge * scale * eye, c[fit_on].T @ z_flat[fit_on].detach()
            )
            prediction[held] = c[held] @ weight
        return z - prediction.reshape(z.shape).detach()

    def saturation(self, batch_size: int) -> float:
        fold = max(batch_size * (self.n_folds - 1) // self.n_folds, 1)
        return (self.code_dim + 3) / fold


class BandEMA(nn.Module):
    """Debiased EMA of a per-band vector, used for the reliance factor."""

    def __init__(self, n_bands: int, decay: float = 0.995):
        super().__init__()
        self.decay = decay
        self.register_buffer("value", torch.zeros(n_bands))
        self.register_buffer("weight", torch.zeros(()))

    @torch.no_grad()
    def update(self, sample: Tensor) -> Tensor:
        self.value.mul_(self.decay).add_(sample.detach(), alpha=1.0 - self.decay)
        self.weight.mul_(self.decay).add_(1.0 - self.decay)
        return self.value / self.weight.clamp_min(1e-12)


def reliance_band_energy(zs: list[Tensor], velocity: Tensor, bands: SpectralBands,
                         n_probes: int = 1, create_graph: bool = False) -> Tensor:
    """``S_k = E||J_v(Z) P_k||_F^2`` by Hutchinson probes.

    For ``u ~ N(0, I)``, ``E||P_k J^T u||^2 = ||J P_k||_F^2``.  One
    vector-Jacobian product returns every band at once.  No ``Q``: squared error
    on velocities has no gauge direction to remove.

    ``zs`` is a **list**, one feature map per camera, and it must stay a list.
    Concatenating them first produces a tensor that is not itself a node in the
    velocity's graph, and ``autograd.grad`` then fails with "One of the
    differentiated Tensors appears to not have been used in the graph" -- the
    cat output is a leaf of a *new* graph, not the one the velocity came from.
    """
    total = torch.zeros(bands.n_bands, device=zs[0].device, dtype=zs[0].dtype)
    for _ in range(n_probes):
        u = torch.randn_like(velocity)
        grads = torch.autograd.grad(
            (velocity * u).sum(), zs, create_graph=create_graph, retain_graph=True
        )
        # average over cameras: one penalty for the whole visual stream rather
        # than a per-camera weighting we would then have to justify
        total = total + sum(bands.band_energy(g).mean(dim=0) for g in grads) / len(grads)
    return total / n_probes


def sib_penalty(zs: list[Tensor], velocity: Tensor, actions: Tensor, t: Tensor,
                bands: SpectralBands, conditioner: ActionConditioner, ema: BandEMA,
                n_probes: int = 1, eps: float = 1e-8, stop_grad: bool = False):
    """``sum_k sqrt(S_k V_k + eps)``, plus diagnostics.

    ``stop_grad=False`` by default, which is **not** the published Equation (12).
    With ``S`` detached the gradient of ``sqrt(sg(S) V)`` points toward smaller
    ``V`` unconditionally, and on CIFAR-10 that drove the representation onto its
    class prototypes -- ``V_k / E||Z_k||^2`` fell to 0.000 while the penalty
    happily reported success.  Differentiating through ``S`` restores the
    invariance of Proposition 4 and removes the degenerate direction, at the cost
    of a second backward.  The flag is kept so the published form can be run.
    """
    # The encoder flattens the frame stack: it sees (B*T, C, H, W) while the
    # action chunk and flow time are per-sample (B).  Each frame of a stack shares
    # the sample's action target, so the conditioning is repeat-interleaved to
    # match the encoder's row order ([b0t0, b0t1, b1t0, ...] from reshape(B*T,...)).
    reps = zs[0].shape[0] // actions.shape[0]
    if reps > 1:
        actions = actions.repeat_interleave(reps, dim=0)
        t = t.repeat_interleave(reps, dim=0)
    v_k = sum(
        bands.band_energy(conditioner.residual(z, actions, t)).mean(dim=0) for z in zs
    ) / len(zs)
    s_live = reliance_band_energy(zs, velocity, bands, n_probes, create_graph=not stop_grad)
    s_used = ema.update(s_live) if stop_grad else s_live
    penalty = torch.sqrt(s_used * v_k + eps).sum()

    with torch.no_grad():
        energy = sum(bands.band_energy(z.detach()).mean(dim=0) for z in zs) / len(zs)
        info = {
            "sib/penalty": float(penalty),
            # -> 0 signals the representation collapsing onto the conditional mean
            #    rather than the policy relying on it less
            "sib/v_over_e": float((v_k.detach() / energy.clamp_min(1e-12)).median()),
            "sib/S_max": float(s_live.detach().max()),
            "sib/V_max": float(v_k.detach().max()),
        }
    return penalty, info
