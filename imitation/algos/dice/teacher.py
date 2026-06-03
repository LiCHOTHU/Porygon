"""Frozen-BC FM teacher: deterministic Euler from a given start noise.

Used inside DistilledRLModel.actor_loss for the BC-MSE anchor. Conditioning the
teacher on the same x0 as the student keeps the anchor pointwise (sample-level
target), matching DICE-RL-Robot's `get_action(state, noise)` usage of a single-step
DDIM teacher. Our 10-step flow makes the teacher deterministic given (cond, x0).
"""

import torch


class FMTeacher:
    """Wraps the BC FlowMatchingPolicy. Holds no parameters of its own; just calls
    forward_enc once per call and forward_dec K times. Assumed-frozen policy.

    Usage:
        teacher = FMTeacher(bc_policy)
        a = teacher(state_unflat, noise)
    state_unflat : (B, num_enc, hidden) -- the conditioning before flattening
    noise        : (B, chunk, action_dim)
    """

    def __init__(self, bc_policy):
        self.policy = bc_policy
        self.K = int(bc_policy.num_inference_steps)

    @torch.no_grad()
    def __call__(self, state_unflat: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        # If state arrived flat, reshape using policy's expected encoder out shape.
        if state_unflat.dim() == 2:
            B = state_unflat.shape[0]
            # We rely on caller passing unflat; if flat is provided, error loudly.
            raise ValueError("FMTeacher expects unflat state (B, num_enc, hidden).")
        was_training = self.policy.training
        self.policy.eval()
        x = noise
        delta_t = 1.0 / self.K
        t = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        enc_cache = self.policy.velocity_net.forward_enc(state_unflat)
        for _ in range(self.K):
            v = self.policy.velocity_net.forward_dec(x, t, enc_cache)
            x = x + delta_t * v
            t = t + delta_t
        if was_training:
            self.policy.train()
        return torch.clamp(x, -1, 1)
