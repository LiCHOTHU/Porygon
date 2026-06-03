"""PolicyDrifting -- drifting-style BC on LIBERO.

Faithful PyTorch port of the JAX `drift_loss` from the official paper repo:
  https://github.com/lambertae/drifting/blob/main/drift_loss.py
applied per-state with G generator particles per state.

What is preserved from the parent FlowMatchingPolicy
---------------------------------------------------
The MODEL ARCHITECTURE, MODEL INPUT, and MODEL OUTPUT are unchanged. We
instantiate the same `velocity_net = DiTNoiseNet(...)` with the same kwargs,
the same encoder, the same `get_cond`, and call it with the same signature
`velocity_net(x, t, cond) -> (_, v)` where x is (B, chunk_size, action_dim),
t is (B,), cond is (B, num_enc, hidden). The exact same forward_enc /
forward_dec split is used at inference. The only thing that changes is the
training algorithm.

Algorithm (Algorithm 2, paper -- per state, G particles)
--------------------------------------------------------
Per state i in the batch:
  - Draw G noise tensors and query the velocity net at fixed t to get G
    generator particles: gen_i^g = x_0^g + v(x_0^g, t=drift_t, cond_i).
  - Positive anchor pool: own demo (C_p = 1).
  - Negative anchor pool: empty external + gen-as-negatives (the G
    particles themselves, diagonal self-masked). Mass conservation comes
    from the symmetric softmax over the G generator rows.

Drift field (1:1 with lambertae JAX impl):
  targets = concat([old_gen, fixed_neg, fixed_pos])                 # (B, T, S)
  dist = cdist(old_gen, targets)
  scale = mean(weighted_dist)                                        # scalar
  scale_inputs = max(scale / sqrt(S), 1e-3)                          # scalar
  gen_scaled = gen / scale_inputs                                    # grad-bearing
  targets_scaled = targets / scale_inputs                            # detached
  dist_normed = dist / max(scale, 1e-3)                              # O(1)
  block-mask the (gen_i, gen_i) diagonal with +100 so no self-repulsion
  For each R in (0.02, 0.05, 0.2):
    logits = -dist_normed / R                                        # very peaked
    A_row, A_col = softmax over targets / over rows
    A = sqrt(max(A_row * A_col, 1e-6)) * weights
    aff_neg = A[..., :C_g+C_n],   aff_pos = A[..., C_g+C_n:]
    R_coeff = concat([-aff_neg * sum(aff_pos), +aff_pos * sum(aff_neg)])
    V_R = R_coeff @ targets_scaled  -  sum(R_coeff) * old_gen_scaled  # f_self correction
    per-R normalize: V_R /= sqrt(mean(V_R^2))
    V_total += V_R
  goal_scaled = old_gen_scaled + V_total
  loss = mean((gen_scaled - goal_scaled)^2)

Inferenced action MSE (logged per epoch, not used in gradient)
--------------------------------------------------------------
The drift loss value is bounded (~num_R = 3) because V_R is per-tau
normalized -- it's not a faithful progress signal. We additionally compute
MSE(mean_over_G(a_pred), demos) in normalized action space and surface it
through info["action_mse"]; train.py averages and logs it per epoch.

Inference
---------
One-step generation by default: a = clamp(x_0 + v(x_0, drift_t, cond), -1, 1).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from imitation.algos.fm_policy import FlowMatchingPolicy


class PolicyDrifting(FlowMatchingPolicy):
    def __init__(
        self,
        drift_R_list=(0.02, 0.05, 0.2),
        drift_t: float = 0.0,
        drift_num_gen: int = 4,
        action_clamp: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.drift_R_list = tuple(float(r) for r in drift_R_list)
        self.drift_t = float(drift_t)
        self.drift_num_gen = int(drift_num_gen)
        assert self.drift_num_gen >= 2, "drift_num_gen must be >=2 for mass conservation"
        self.action_clamp = bool(action_clamp)

    @staticmethod
    def _batched_cdist(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        # x: (B, N, D), y: (B, M, D) -> (B, N, M). Matches the JAX cdist.
        xy = torch.einsum("bnd,bmd->bnm", x, y)
        xnorm = torch.einsum("bnd,bnd->bn", x, x).unsqueeze(-1)
        ynorm = torch.einsum("bmd,bmd->bm", y, y).unsqueeze(-2)
        sq = (xnorm + ynorm - 2.0 * xy).clamp_min(eps)
        return sq.sqrt()

    def _drift_loss(
        self,
        gen: torch.Tensor,                # (B, C_g, S) -- grad-bearing
        fixed_pos: torch.Tensor,          # (B, C_p, S) -- detached anchors
        fixed_neg: torch.Tensor = None,   # (B, C_n, S) or None
    ):
        """Port of lambertae/drifting/drift_loss.py:drift_loss to PyTorch."""
        B, C_g, S = gen.shape
        C_p = fixed_pos.shape[1]
        device = gen.device
        dtype = gen.dtype

        if fixed_neg is None:
            fixed_neg = gen.new_zeros(B, 0, S)
        C_n = fixed_neg.shape[1]

        weight_gen = gen.new_ones(B, C_g)
        weight_pos = gen.new_ones(B, C_p)
        weight_neg = gen.new_ones(B, C_n)

        old_gen = gen.detach()
        targets = torch.cat([old_gen, fixed_neg, fixed_pos], dim=1)              # (B, T, S)
        targets_w = torch.cat([weight_gen, weight_neg, weight_pos], dim=1)        # (B, T)

        with torch.no_grad():
            dist = self._batched_cdist(old_gen, targets)                          # (B, C_g, T)
            weighted_dist = dist * targets_w.unsqueeze(1)
            scale = weighted_dist.mean() / targets_w.mean().clamp_min(1e-8)
            scale_inputs = (scale / (S ** 0.5)).clamp_min(1e-3)                   # scalar

            old_gen_scaled = old_gen / scale_inputs
            targets_scaled = targets / scale_inputs

            dist_normed = dist / scale.clamp_min(1e-3)                            # O(1)

            mask_val = 100.0
            diag = torch.eye(C_g, device=device, dtype=dtype)
            block_mask = F.pad(diag, (0, C_n + C_p)).unsqueeze(0)                 # (1, C_g, T)
            dist_normed = dist_normed + block_mask * mask_val

            split_idx = C_g + C_n
            V_total = torch.zeros_like(old_gen_scaled)
            for R in self.drift_R_list:
                logits = -dist_normed / R
                aff_row = torch.softmax(logits, dim=-1)
                aff_col = torch.softmax(logits, dim=-2)
                affinity = (aff_row * aff_col).clamp_min(1e-6).sqrt()
                affinity = affinity * targets_w.unsqueeze(1)                      # (B, C_g, T)

                aff_neg = affinity[:, :, :split_idx]
                aff_pos = affinity[:, :, split_idx:]
                sum_pos = aff_pos.sum(dim=-1, keepdim=True)
                sum_neg = aff_neg.sum(dim=-1, keepdim=True)
                r_coeff_neg = -aff_neg * sum_pos
                r_coeff_pos = aff_pos * sum_neg
                R_coeff = torch.cat([r_coeff_neg, r_coeff_pos], dim=-1)           # (B, C_g, T)

                total_force = torch.einsum("biy,byx->bix", R_coeff, targets_scaled)
                total_coeffs = R_coeff.sum(dim=-1)                                # (B, C_g)
                total_force = total_force - total_coeffs.unsqueeze(-1) * old_gen_scaled

                f_norm = (total_force ** 2).mean().clamp_min(1e-8).sqrt()
                V_total = V_total + total_force / f_norm

            goal_scaled = old_gen_scaled + V_total                                 # detached

        gen_scaled = gen / scale_inputs
        diff = gen_scaled - goal_scaled
        loss = (diff ** 2).mean()

        info = {
            "drift_scale": float(scale),
            "drift_force_norm": float(V_total.pow(2).mean().sqrt()),
        }
        return loss, info

    def compute_loss(self, data):
        data = self.preprocess_input(data, train_mode=True)
        cond = self.get_cond(data)                                                # (B, num_enc, hidden)
        actions = data["abs_actions"] if self.abs_action else data["actions"]
        if self.action_clamp:
            actions = torch.clamp(actions, -1, 1)

        B = cond.shape[0]
        device = cond.device
        H, A = self.chunk_size, self.network_action_dim
        S = H * A
        G = self.drift_num_gen

        # Replicate cond G times so each state gets G independent noise draws.
        cond_g = cond.unsqueeze(1).expand(B, G, *cond.shape[1:]).reshape(B * G, *cond.shape[1:])
        x0 = torch.randn(B * G, H, A, device=device, dtype=actions.dtype)
        t = torch.full((B * G,), self.drift_t, device=device, dtype=actions.dtype)
        _, v = self.velocity_net(x0, t, cond_g)                                   # (B*G, H, A)
        a_pred = x0 + v                                                           # 1-step Euler

        gen = a_pred.reshape(B, G, S)
        pos = actions.reshape(B, 1, S)

        loss, info = self._drift_loss(gen, pos, fixed_neg=None)
        info["loss"] = float(loss.detach())

        # Inferenced action MSE in normalized action space (no extra forward pass).
        # This is the actual BC metric -- how close the model's outputs are to demos.
        with torch.no_grad():
            a_pred_avg = a_pred.reshape(B, G, H, A).mean(dim=1)                   # (B, H, A)
            info["action_mse"] = float(((a_pred_avg - actions) ** 2).mean())

        return loss, info

    def sample_actions(self, data):
        with torch.no_grad():
            data = self.preprocess_input(data, train_mode=False)
            cond = self.get_cond(data)
            B, device = cond.shape[0], cond.device

            x = torch.randn(B, self.chunk_size, self.network_action_dim, device=device)
            enc_cache = self.velocity_net.forward_enc(cond)

            K = max(1, int(self.num_inference_steps))
            delta_t = 1.0 / K
            t = torch.full((B,), self.drift_t, device=device, dtype=cond.dtype)
            for _ in range(K):
                v = self.velocity_net.forward_dec(x, t, enc_cache)
                x = x + delta_t * v
                t = t + delta_t

            return torch.clamp(x, -1, 1).cpu().numpy()
