"""Distilled residual-RL model for LIBERO FM policy.

Faithful port of real-stanford/dice-rl's DistillResidualRLModel (arXiv 2603.10263,
ICML 2026 -- "From Prior to Pro"). The sim repo's model is at:
    https://github.com/real-stanford/dice-rl/blob/main/model/rl/distill_residual_rl.py
and this file mirrors its loss(), actor_loss(), critic_loss(), get_exploration_action()
and the residual-on-teacher get_action() structure 1:1, with two LIBERO-specific
adaptations:

  (i)  The frozen pretrained policy is attached via `attach_teacher(teacher_fn)`
       (an FMTeacher wrapping our BC FlowMatchingPolicy) instead of being
       instantiated from a saved hydra config.
  (ii) Optional `zero_final_layer` zero-inits the final actor Linear so the
       initial residual is exactly 0 (a_student == a_teacher at iter 0).
       The official does NOT zero-init; we expose this as opt-in for the LIBERO
       use case where "from prior" should be exact at iter 0.

Everything else -- warmup vs post-warmup loss schedule, Q-normalization, soft
Q-filtering, self-imitation BC mask, multi-z next-noise target, n-step gamma,
expert/online data masks, Q-based exploration -- matches the official.

Action shape conventions (matched to LIBERO):
    state    : (B, num_enc, hidden)  -- cond from the BC encoder
    noise    : (B, horizon, A)       -- per-chunk Gaussian
    action   : (B, horizon, A)       -- normalized, NOT clamped inside the model
    a_student = a_teacher(state, noise) + residual_theta(state, noise)
"""

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
    """(state, noise) -> residual chunk. Unbounded MLP delta added on top of the
    frozen teacher's action. Output activation is Identity (the residual is NOT
    a normalized [-1, 1] action). If zero_final_layer=True, the last Linear is
    zero-initialized so the initial residual is exactly 0 (deviation from the
    official; opt-in for the LIBERO "from prior" property)."""

    def __init__(self, state_dim: int, action_dim: int, horizon_steps: int,
                 hidden_dims: List[int] = (1024, 1024, 1024),
                 activation: str = "Mish", use_layernorm: bool = True,
                 zero_final_layer: bool = False):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.horizon_steps = horizon_steps
        in_dim = state_dim + horizon_steps * action_dim
        out_dim = horizon_steps * action_dim
        self.mlp = _mlp([in_dim] + list(hidden_dims) + [out_dim],
                        activation=activation, use_layernorm=use_layernorm,
                        out_activation="Identity")
        if zero_final_layer:
            last_linear = None
            for m in self.mlp.modules():
                if isinstance(m, nn.Linear):
                    last_linear = m
            assert last_linear is not None, "no Linear layer found in actor MLP"
            nn.init.zeros_(last_linear.weight)
            if last_linear.bias is not None:
                nn.init.zeros_(last_linear.bias)

    def forward(self, state: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Returns the residual delta, NOT the full action."""
        B = noise.shape[0]
        x = torch.cat([state.reshape(B, -1), noise.reshape(B, -1)], dim=-1)
        delta = self.mlp(x)
        return delta.view(B, self.horizon_steps, self.action_dim)


class DistilledCritic(nn.Module):
    """Ensemble of Q(s, a) or Q(s, z, a) -> R. Conservative aggregation via min
    (official default) or LCB / mean for diagnostics."""

    def __init__(self, state_dim: int, action_dim: int, horizon_steps: int,
                 hidden_dims: List[int] = (1024, 1024, 1024),
                 ensemble_size: int = 10, q_depends_on_noise: bool = False,
                 conservative: str = "min", lcb_kappa: float = 0.1,
                 activation: str = "Mish", use_layernorm: bool = True,
                 td_loss: str = "mse"):
        super().__init__()
        self.q_depends_on_noise = q_depends_on_noise
        self.conservative = conservative
        self.lcb_kappa = lcb_kappa
        self.td_loss = td_loss
        in_extra = 2 if q_depends_on_noise else 1
        in_dim = state_dim + in_extra * horizon_steps * action_dim
        self.Q_ensemble = nn.ModuleList([
            _mlp([in_dim] + list(hidden_dims) + [1], activation=activation,
                 use_layernorm=use_layernorm, out_activation="Identity")
            for _ in range(ensemble_size)
        ])

    def _features(self, state: torch.Tensor, noise: Optional[torch.Tensor],
                  action: torch.Tensor) -> torch.Tensor:
        B = action.shape[0]
        parts = [state.reshape(B, -1)]
        if self.q_depends_on_noise:
            assert noise is not None, "q_depends_on_noise=True but noise=None"
            parts.append(noise.reshape(B, -1))
        parts.append(action.reshape(B, -1))
        return torch.cat(parts, dim=-1)

    def forward(self, state: torch.Tensor, noise: Optional[torch.Tensor],
                action: torch.Tensor, return_all: bool = False,
                return_mean: bool = False) -> torch.Tensor:
        x = self._features(state, noise, action)
        qs = [q(x) for q in self.Q_ensemble]
        # BCE: sigmoid before returning a single Q (loss path uses raw logits via return_all).
        if self.td_loss == "bce" and not return_all:
            qs = [torch.sigmoid(q) for q in qs]
        if return_all:
            return qs
        q_stack = torch.stack(qs, dim=0)
        if return_mean:
            return q_stack.mean(dim=0)
        if self.conservative == "min":
            return q_stack.min(dim=0).values
        if self.conservative == "lcb":
            return q_stack.mean(dim=0) - self.lcb_kappa * q_stack.std(dim=0)
        if self.conservative == "mean":
            return q_stack.mean(dim=0)
        raise ValueError(f"Unknown conservative method: {self.conservative}")


class NoiseHead(nn.Module):
    """ReinFlow-style learnable, state-conditioned, bounded output noise.

    Maps state -> per-(chunk_step, action_dim) noise scale `sigma_s`, squashed
    into `[sigma_min, sigma_max]` so it cannot diverge. Used by
    `DistilledRLModel.get_collection_action` to inject calibrated exploration
    noise *on top of* the deterministic K=1 student action, restoring the
    per-state action diversity that the K=1 teacher's noise->action map fails
    to produce.

    Init: the final Linear is zero'd so the initial pre-sigmoid logit is the
    bias only; with bias=`initial_logit` (default 0.0), `sigma_s` starts at the
    midpoint of [sigma_min, sigma_max]. Pass `initial_logit > 0` to start near
    `sigma_max` (more exploration) or `< 0` to start near `sigma_min`.

    NOT trained in Phase 1 (kept out of the actor optimizer) so the head acts
    as a state-conditioned but effectively constant noise floor; Phase 2 adds
    it to the optimizer with -Q + magnitude regularization.
    """

    def __init__(self, state_dim: int, action_dim: int, horizon_steps: int,
                 hidden_dim: int = 256, sigma_min: float = 0.01,
                 sigma_max: float = 0.1, initial_logit: float = 0.0):
        super().__init__()
        assert sigma_min > 0 and sigma_max > sigma_min, \
            f"need 0 < sigma_min < sigma_max, got [{sigma_min}, {sigma_max}]"
        self.action_dim = action_dim
        self.horizon_steps = horizon_steps
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        out_dim = horizon_steps * action_dim
        self.mlp = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, out_dim),
        )
        # Zero final layer, set bias = initial_logit -> sigma_s starts at
        # sigma_min + (sigma_max - sigma_min) * sigmoid(initial_logit).
        last = self.mlp[-1]
        nn.init.zeros_(last.weight)
        nn.init.constant_(last.bias, float(initial_logit))

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """state: (B, num_enc, hidden) or (B, state_dim). Returns sigma_s
        of shape (B, horizon_steps, action_dim)."""
        B = state.shape[0]
        x = state.reshape(B, -1)
        logit = self.mlp(x)
        span = self.sigma_max - self.sigma_min
        sigma_flat = self.sigma_min + span * torch.sigmoid(logit)
        return sigma_flat.view(B, self.horizon_steps, self.action_dim)


class DistilledRLModel(nn.Module):
    """Residual-RL student that adds an MLP delta on top of a frozen teacher.

    See module docstring for the exact loss formulas (mirrors official). External
    API: get_action, get_exploration_action, loss, update_target_networks, and
    the attach_teacher(teacher_fn) hook used in our LIBERO wiring.
    """

    def __init__(self, state_dim: int, action_dim: int, horizon_steps: int,
                 # Network configurations
                 actor_hidden: List[int] = (1024, 1024, 1024),
                 critic_hidden: List[int] = (1024, 1024, 1024),
                 ensemble_size: int = 10,
                 q_depends_on_noise: bool = False,
                 conservative: str = "min", lcb_kappa: float = 0.1,
                 td_loss: str = "mse",
                 # Loss coefficients
                 bc_loss_weight: float = 100.0,
                 critic_weight: float = 1.0,
                 # Multi-z sampling
                 num_multi_z: int = 8,
                 # Q-filtering / self-imitation (post-warmup)
                 use_soft_q_filtering: bool = False,
                 q_filtering_warmup_steps: int = 25000,
                 q_underestimation_threshold: float = -0.1,
                 # Exploration warmup (separate from Q-filtering)
                 replay_flow_warmup_steps: int = 1000,
                 # Q-normalization
                 use_q_normalization: bool = False,
                 # Multi-sample next-noise for stable target Q
                 multi_sample_next_noise: bool = False,
                 num_next_noise_samples: int = 4,
                 # N-step returns
                 use_n_step: bool = False,
                 n_step: int = 1,
                 # Expert / online data masks (RLPD)
                 disable_q_loss_for_expert_data: bool = False,
                 disable_td_loss_for_expert_data: bool = False,
                 always_retain_bc_loss_for_expert_data: bool = False,
                 # ReinFlow-style learnable noise head (Phase 1+).
                 # When True, get_collection_action samples a_exec = clamp(mu + sigma_s * eps)
                 # using a state-conditioned bounded sigma_s. The collector stores a_exec into
                 # replay so the critic learns Q around the noisy action.
                 use_noise_head: bool = False,
                 noise_sigma_min: float = 0.01,
                 noise_sigma_max: float = 0.1,
                 noise_head_hidden: int = 256,
                 noise_head_initial_logit: float = 0.0,
                 # CQL-style conservative penalty on the critic. When > 0, adds
                 #   cql_weight * ( mean Q(s,z,a_student) - mean Q(s,z,a_data) )
                 # to critic_loss. Pushes Q DOWN at the policy's own actions, UP at
                 # data actions; suppresses the monotonic Q-level inflation seen on
                 # long-horizon DICE runs (hard-8 z-cond probe: Q drifts -0.81->+0.41
                 # while ||residual||~3e-4). Default 0 -> off.
                 cql_weight: float = 0.0,
                 # GRD-style coverage repulsion across the K multi-z action samples.
                 # DICE's -Q term collapses all K samples onto the Q-argmax (the thin
                 # noise->action map that kills the max_q gain channel on K=1 drift).
                 # This adds an isolated SVGD/GRD repulsion kernel that pushes the K
                 # per-state samples apart; together with -Q it reconstructs an SVGD
                 # drift update with the LEARNED Q as the energy. repel_weight=0 -> off
                 # (exact original behaviour). bandwidth = median pairwise distance *
                 # repel_bandwidth_scale (classic SVGD median heuristic, recomputed per
                 # step so it self-limits once particles are ~1 bandwidth apart).
                 repel_weight: float = 0.0,
                 repel_bandwidth_scale: float = 1.0,
                 # ---- DFP (Drifting Field Policy) actor mode ----
                 # actor_mode="dfp" replaces the residual (-Q + bc-anchor) actor with the
                 # official DFP update (github.com/KAIST-Visual-AI-Group/DFP, agents/drift.py):
                 # sample dfp_n_samples candidate actions from the CURRENT policy, score them
                 # with the critic, keep the top dfp_top_k by Q, and DRIFT the policy's
                 # particles toward those top-K (improvement) plus a teacher/BC anchor
                 # (trust region, weight dfp_alpha_target), plus a direct -Q term (weight
                 # dfp_q_weight). The improvement signal is a RANKING -> robust to the loose,
                 # over-dispersed K=1 drift Q surface that pins a gradient-based residual.
                 # Reuses the multi-scale drift_loss from imitation/algos/rl/grd.py and trains
                 # the same residual-on-teacher actor as residual mode.
                 actor_mode: str = "residual",
                 dfp_n_samples: int = 32,
                 dfp_top_k: int = 4,
                 dfp_gen: int = 8,
                 dfp_alpha_target: float = 1.0,
                 dfp_q_weight: float = 1.0,
                 dfp_lam_rep: float = 1.0,
                 # Two-field residual actor (actor_mode in
                 # {field_pointwise, field_distributional}); see field_actor_loss.
                 field_cfg: Optional[dict] = None,
                 # DAWN (arXiv 2602.10539) layernorm ablation knobs. Defaults preserve
                 # the historical behaviour (LN on in both nets). DAWN Fig.14: critic
                 # LN restores sensitivity to the residual (scale mismatch); actor LN
                 # gives no benefit and can degrade — actor_layernorm=false to test.
                 actor_layernorm: bool = True,
                 critic_layernorm: bool = True,
                 # Misc
                 clip_action: bool = False,    # official sim does NOT clamp inside get_action
                 zero_final_layer: bool = False,
                 device: str = "cuda"):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.horizon_steps = horizon_steps
        self.bc_loss_weight = bc_loss_weight
        self.critic_weight = critic_weight
        self.num_multi_z = num_multi_z
        self.td_loss = td_loss
        self.clip_action = clip_action
        self.device = device
        # Q-filtering
        self.use_soft_q_filtering = use_soft_q_filtering
        self.q_filtering_warmup_steps = q_filtering_warmup_steps
        self.q_underestimation_threshold = q_underestimation_threshold
        # Exploration warmup
        self.replay_flow_warmup_steps = replay_flow_warmup_steps
        # Q-norm
        self.use_q_normalization = use_q_normalization
        # Multi-z target
        self.multi_sample_next_noise = multi_sample_next_noise
        self.num_next_noise_samples = num_next_noise_samples
        # N-step
        self.use_n_step = use_n_step
        self.n_step = n_step
        # Data-source masks
        self.disable_q_loss_for_expert_data = disable_q_loss_for_expert_data
        self.disable_td_loss_for_expert_data = disable_td_loss_for_expert_data
        self.always_retain_bc_loss_for_expert_data = always_retain_bc_loss_for_expert_data
        # CQL conservative penalty weight
        self.cql_weight = float(cql_weight)
        # GRD coverage repulsion across multi-z action samples
        self.repel_weight = float(repel_weight)
        self.repel_bandwidth_scale = float(repel_bandwidth_scale)
        # DFP actor mode (critic-ranked top-K drift)
        self.actor_mode = str(actor_mode)
        self.dfp_n_samples = int(dfp_n_samples)
        self.dfp_top_k = int(dfp_top_k)
        self.dfp_gen = int(dfp_gen)
        self.dfp_alpha_target = float(dfp_alpha_target)
        self.dfp_q_weight = float(dfp_q_weight)
        self.dfp_lam_rep = float(dfp_lam_rep)
        self.field_cfg = dict(field_cfg or {})
        # Unified-field adaptive BC weight (dual variable lambda); regulated toward
        # residual/trust-region target. Persists across gradient steps.
        self._field_lambda = float(self.field_cfg.get("lambda_init", 1.0))
        self._field_res_ema = None   # smoothed residual rms for the dual controller

        self.actor = DistilledActor(
            state_dim=state_dim, action_dim=action_dim, horizon_steps=horizon_steps,
            hidden_dims=actor_hidden, use_layernorm=bool(actor_layernorm),
            zero_final_layer=zero_final_layer,
        ).to(device)
        self.critic = DistilledCritic(
            state_dim=state_dim, action_dim=action_dim, horizon_steps=horizon_steps,
            hidden_dims=critic_hidden, ensemble_size=ensemble_size,
            q_depends_on_noise=q_depends_on_noise,
            conservative=conservative, lcb_kappa=lcb_kappa,
            td_loss=td_loss, use_layernorm=bool(critic_layernorm),
        ).to(device)
        self.target_critic = DistilledCritic(
            state_dim=state_dim, action_dim=action_dim, horizon_steps=horizon_steps,
            hidden_dims=critic_hidden, ensemble_size=ensemble_size,
            q_depends_on_noise=q_depends_on_noise,
            conservative=conservative, lcb_kappa=lcb_kappa,
            td_loss=td_loss, use_layernorm=bool(critic_layernorm),
        ).to(device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        for p in self.target_critic.parameters():
            p.requires_grad = False

        # ReinFlow-style learnable bounded noise head (collection-side stochasticity).
        # When `use_noise_head=False` (default) the head is None and the model behaves
        # exactly as before — protects FM K=10 runs from accidental drift.
        self.use_noise_head = bool(use_noise_head)
        if self.use_noise_head:
            self.noise_head = NoiseHead(
                state_dim=state_dim, action_dim=action_dim, horizon_steps=horizon_steps,
                hidden_dim=int(noise_head_hidden),
                sigma_min=float(noise_sigma_min), sigma_max=float(noise_sigma_max),
                initial_logit=float(noise_head_initial_logit),
            ).to(device)
        else:
            self.noise_head = None

        # teacher_fn(state_unflat (B,num_enc,hidden), noise (B,H,A)) -> a_teacher (B,H,A)
        # Returns the BC FM's deterministic Euler output (already clamped to [-1, 1]).
        # Computed under torch.no_grad in FMTeacher; we treat it as frozen.
        self._teacher_fn = None

    # ------------------------------------------------------------------
    # Teacher / action
    # ------------------------------------------------------------------
    def attach_teacher(self, teacher_fn):
        self._teacher_fn = teacher_fn

    def get_action(self, state: torch.Tensor, noise: torch.Tensor,
                   return_pretrained_actions: bool = False):
        """a_student = a_teacher(state, noise) + actor(state, noise).

        Mirrors real-stanford/dice-rl: total is UNCLAMPED inside the model
        (env-execution sites in collector / dice_train clamp before stepping).
        Set self.clip_action=True if you want internal clamping (matches the
        DICE-RL-Robot variant); we leave it off by default.
        """
        assert self._teacher_fn is not None, "call attach_teacher(teacher_fn) first."
        with torch.no_grad():
            a_teacher = self._teacher_fn(state, noise)
        residual = self.actor(state, noise)
        a_total = a_teacher + residual
        if self.clip_action:
            a_total = torch.clamp(a_total, -1.0, 1.0)
        if return_pretrained_actions:
            return a_total, a_teacher
        return a_total

    def get_collection_action(self, state: torch.Tensor,
                              noise: Optional[torch.Tensor] = None):
        """ReinFlow-style noisy rollout action for the K=1 collapse case.

        Computes mu = a_teacher + residual (the deterministic 1-step action),
        sigma_s = NoiseHead(state) (bounded in [sigma_min, sigma_max]), and
        returns the EXECUTED action a_exec = clamp(mu + sigma_s * eps, -1, 1)
        along with the auxiliaries needed by the collector to fill replay:

            returns dict {
                "a_exec":  (B, H, A)  -- executed, clamped, what hits the env
                "mu":      (B, H, A)  -- deterministic action (pre-noise, pre-clamp)
                "sigma_s": (B, H, A)  -- per-dim noise scale
                "eps":     (B, H, A)  -- standard-normal draw used
                "noise":   (B, H, A)  -- input noise z (so the buffer can recompute
                                       a_teacher + residual later for the actor pass)
            }

        Requires `use_noise_head=True`; raises otherwise. Runs no_grad — Phase 1
        does not train the head (it's a per-state bounded floor); Phase 2 will
        include `self.noise_head.parameters()` in an optimizer and train against
        -Q(s, a_exec) + magnitude regularizer.
        """
        assert self.use_noise_head, "get_collection_action requires use_noise_head=True"
        assert self._teacher_fn is not None, "call attach_teacher(teacher_fn) first."
        B = state.shape[0]
        H, A = self.horizon_steps, self.action_dim
        if noise is None:
            noise = torch.randn(B, H, A, device=state.device, dtype=state.dtype)
        with torch.no_grad():
            a_teacher = self._teacher_fn(state, noise)
            residual = self.actor(state, noise)
            mu = a_teacher + residual
            sigma_s = self.noise_head(state)
            eps = torch.randn(B, H, A, device=state.device, dtype=mu.dtype)
            a_exec = torch.clamp(mu + sigma_s * eps, -1.0, 1.0)
        return {
            "a_exec": a_exec,
            "mu": mu,
            "sigma_s": sigma_s,
            "eps": eps,
            "noise": noise,
        }

    # ------------------------------------------------------------------
    # Q-based exploration (matches official get_exploration_action)
    # ------------------------------------------------------------------
    def get_exploration_action(self, state: torch.Tensor, num_samples: int = 10,
                               exploration_strategy: str = "max_q_std",
                               training_step: int = 0):
        """Select a noise that maximises a Q-ensemble criterion. Mirrors the
        official's max_q_std / max_q_min / max_q_std_filtered_by_min strategies.
        Returns (selected_action, selected_noise), both (B, H, A)."""
        B = state.shape[0]
        device = state.device
        if training_step <= self.replay_flow_warmup_steps:
            # Warmup: single random sample.
            noise = torch.randn(B, self.horizon_steps, self.action_dim, device=device)
            with torch.no_grad():
                action = self.get_action(state, noise)
            return action, noise

        # Sample num_samples noise vectors per batch element.
        noise_samples = torch.randn(num_samples, B, self.horizon_steps, self.action_dim, device=device)
        state_rep = state.unsqueeze(0).expand(num_samples, *state.shape).reshape(
            num_samples * B, *state.shape[1:]
        )
        noise_flat = noise_samples.reshape(num_samples * B, self.horizon_steps, self.action_dim)
        with torch.no_grad():
            actions_flat = self.get_action(state_rep, noise_flat)
            q_all = self.critic(state_rep, noise_flat, actions_flat, return_all=True)
            # q_all: list of (num_samples*B, 1); stack into (ensemble, num_samples, B, 1)
            q_stacked = torch.stack(q_all, dim=0).view(len(q_all), num_samples, B, 1)
            if exploration_strategy == "max_q_min":
                q_min = q_stacked.min(dim=0)[0].squeeze(-1)              # (num_samples, B)
                sel = q_min.argmax(dim=0)                                 # (B,)
            elif exploration_strategy == "max_q_mean":
                # Rank candidates by the MEAN over the ensemble instead of the
                # min. min-over-K-heads is increasingly pessimistic as the
                # ensemble grows (K=10 here vs act_sim's 2): the single most
                # paranoid head decides the ranking and can bury the real
                # noise->action signal. mean is the least-conservative ranking.
                q_mean = q_stacked.mean(dim=0).squeeze(-1)               # (num_samples, B)
                sel = q_mean.argmax(dim=0)
            elif exploration_strategy == "max_q_lcb":
                # Middle ground: rank by mean - kappa*std (a soft, ensemble-size
                # invariant pessimism) rather than the hard min.
                q_lcb = (q_stacked.mean(dim=0) - self.critic.lcb_kappa * q_stacked.std(dim=0)).squeeze(-1)
                sel = q_lcb.argmax(dim=0)
            elif exploration_strategy == "max_q_std":
                q_std = q_stacked.std(dim=0).squeeze(-1)                  # (num_samples, B)
                sel = q_std.argmax(dim=0)
            elif exploration_strategy == "max_q_std_filtered_by_min":
                q_min = q_stacked.min(dim=0)[0].squeeze(-1)               # (num_samples, B)
                k = min(3, num_samples)
                top_q, top_idx = q_min.topk(k, dim=0)                      # (k, B)
                top_q_full = torch.stack([
                    q_stacked[:, top_idx[i], torch.arange(B)]              # (ensemble, B, 1)
                    for i in range(k)
                ], dim=1)                                                  # (ensemble, k, B, 1)
                top_std = top_q_full.std(dim=0).squeeze(-1)                # (k, B)
                sel_in_topk = top_std.argmax(dim=0)                        # (B,)
                sel = torch.gather(top_idx, 0, sel_in_topk.unsqueeze(0)).squeeze(0)
            else:
                raise ValueError(f"unknown exploration_strategy: {exploration_strategy}")

            actions_view = actions_flat.view(num_samples, B, self.horizon_steps, self.action_dim)
            sel_actions = torch.stack([actions_view[sel[b], b] for b in range(B)])
            sel_noise = torch.stack([noise_samples[sel[b], b] for b in range(B)])
        return sel_actions, sel_noise

    # ------------------------------------------------------------------
    # Losses (mirrors real-stanford/dice-rl)
    # ------------------------------------------------------------------
    @staticmethod
    def _q_norm_scale(q_current_samples: torch.Tensor,
                      online_mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """Return a no-grad scalar 1/mean(|Q|) to divide q_loss by, or None if
        the mean is tiny. Mirrors the official's online-masked normalization."""
        if online_mask is not None:
            K = q_current_samples.shape[1]
            mask = online_mask.unsqueeze(-1).expand(-1, K)
            n = mask.sum()
            if n > 0:
                q_abs_mean = ((q_current_samples * mask).abs().sum() / n).detach()
            else:
                q_abs_mean = q_current_samples.abs().mean().detach()
        else:
            q_abs_mean = q_current_samples.abs().mean().detach()
        if q_abs_mean > 1e-8:
            return 1.0 / q_abs_mean
        return None

    @staticmethod
    def _dispersion_repulsion(actions_samples: torch.Tensor,
                              bandwidth_scale: float = 1.0, eps: float = 1e-12):
        """Isolated SVGD/GRD repulsion across the K per-state action samples.

        actions_samples : (B, K, H, A) grad-bearing student actions (K multi-z draws
                          at the SAME state). Returns (repel_loss, dispersion):
          repel_loss  : mean off-diagonal RBF kernel overlap in [0,1]; MINIMIZING it
                        pushes the K samples apart (the SVGD repulsion gradient). The
                        RBF bandwidth is the per-state median pairwise distance (times
                        bandwidth_scale), detached and recomputed each call, so the
                        term self-limits once particles are ~1 bandwidth apart.
          dispersion  : (no-grad) RMS pairwise distance among the K samples -- the
                        metric we watch to confirm the noise->action map fattens.
        """
        B, K = actions_samples.shape[0], actions_samples.shape[1]
        a = actions_samples.reshape(B, K, -1)                      # (B, K, D)
        zero = a.new_zeros(())
        if K < 2:
            return zero, zero
        diff = a.unsqueeze(2) - a.unsqueeze(1)                     # (B, K, K, D)
        d2 = diff.pow(2).sum(-1)                                   # (B, K, K)
        eye = torch.eye(K, device=a.device, dtype=a.dtype).unsqueeze(0)
        offmask = 1.0 - eye
        denom = B * K * (K - 1)
        with torch.no_grad():
            big = d2 + eye * 1e12                                  # mask diagonal for median
            med = big.reshape(B, -1).median(dim=1).values         # (B,) median pairwise d^2
            h = (bandwidth_scale * med).clamp_min(eps).view(B, 1, 1)
            dispersion = ((d2 * offmask).sum() / denom).clamp_min(0).sqrt()
        k = torch.exp(-d2 / h)                                     # (B, K, K), diag=1
        repel_loss = (k * offmask).sum() / denom
        return repel_loss, dispersion

    def actor_loss(self, state: torch.Tensor,
                   training_step: int = 0,
                   q_overestimation: Optional[torch.Tensor] = None,
                   data_source: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Residual-RL actor loss, multi-z, with optional Q-filtering / self-imitation.

        Mirrors real-stanford/dice-rl actor_loss exactly:
          - Sample K noise vectors per state.
          - For each (s, z_k): compute a_student_k and a_teacher_k, Q(s, z_k, a_*).
          - During warmup: -mean Q + bc_weight * ||residual||^2 (uniform).
          - Post-warmup (use_soft_q_filtering=False, official default): HARD filter —
            BC term dropped per-sample whenever Q(student) > Q(teacher).
          - Post-warmup (use_soft_q_filtering=True): BC dropped only when better AND
            Q is underestimated vs the MC return -> guarded self-imitation.

        Args (data_source / q_overestimation are pass-through; safe to leave None):
          state           : (B, num_enc, hidden)
          training_step   : current global step (for warmup gate)
          q_overestimation: (B, 1) predicted_q - mc_return (None disables sQF)
          data_source     : (B, 1)  0=online, 1=expert (None disables masks)
        """
        assert self._teacher_fn is not None, "call attach_teacher(teacher_fn) first."
        B = state.shape[0]
        K = self.num_multi_z
        H = self.horizon_steps
        A = self.action_dim

        # Replicate state K times.
        if state.dim() == 3:
            ne, h = state.shape[1], state.shape[2]
            state_rep = state.unsqueeze(1).expand(-1, K, -1, -1).reshape(B * K, ne, h)
        else:
            state_rep = state.unsqueeze(1).expand(-1, K, -1).reshape(B * K, -1)
        noise_samples = torch.randn(B, K, H, A, device=self.device)
        noise_flat = noise_samples.reshape(B * K, H, A)

        # Compute student + teacher actions for each of B*K rows.
        actions_flat, pretrained_actions_flat = self.get_action(
            state_rep, noise_flat, return_pretrained_actions=True)

        actions_samples = actions_flat.reshape(B, K, H, A)
        pretrained_actions_samples = pretrained_actions_flat.reshape(B, K, H, A)

        # Q for current actions (grad flows to actor through Q).
        q_current_flat = self.critic(state_rep, noise_flat, actions_flat)  # (B*K, 1)
        q_current_samples = q_current_flat.reshape(B, K)

        # Q for pretrained actions (no grad).
        with torch.no_grad():
            q_pretrained_flat = self.critic(state_rep, noise_flat, pretrained_actions_flat)
            q_pretrained_samples = q_pretrained_flat.reshape(B, K)

        in_warmup = training_step <= self.q_filtering_warmup_steps

        # ---- Q loss ----
        q_loss_per_batch = q_current_samples.mean(dim=1)  # (B,)
        online_mask = None
        if self.disable_q_loss_for_expert_data and data_source is not None:
            # data_source: (B, 1) -> (B,)
            online_mask = (data_source == 0).float().squeeze(-1)
            q_loss_per_batch = q_loss_per_batch * online_mask
        q_loss = -q_loss_per_batch.mean()

        if self.use_q_normalization:
            scale = self._q_norm_scale(q_current_samples, online_mask)
            if scale is not None:
                q_loss = scale * q_loss

        # ---- BC / residual regularization with optional filtering ----
        # action_diff(B, K, H, A) -> mse_per_sample(B, K)
        action_diff = actions_samples - pretrained_actions_samples
        mse_per_timestep = (action_diff ** 2).mean(dim=-1)              # (B, K, H)
        mse_per_sample = mse_per_timestep.mean(dim=-1)                  # (B, K)

        if in_warmup:
            # Uniform average BC anchor; equivalent to ||residual||^2 (mean over B*K*H*A).
            filtered_bc_loss = mse_per_sample.mean(dim=1).mean()
            bc_filter = torch.ones(B, 1, device=self.device)
            better_percentage = torch.tensor(0.0, device=self.device)
            q_advantage = torch.zeros(B, 1, device=self.device)
        else:
            with torch.no_grad():
                q_adv_per_sample = q_current_samples - q_pretrained_samples       # (B, K)
                better_per_sample = (q_adv_per_sample > 0).float()                # (B, K)
                avg_q_curr = q_current_samples.mean(dim=1, keepdim=True)          # (B, 1)
                avg_q_pre = q_pretrained_samples.mean(dim=1, keepdim=True)        # (B, 1)
                q_advantage = avg_q_curr - avg_q_pre                              # (B, 1)
                better_percentage = better_per_sample.mean()
                if self.use_soft_q_filtering:
                    if q_overestimation is not None:
                        q_under = (q_overestimation < self.q_underestimation_threshold).float()  # (B, 1)
                        q_under_exp = q_under.expand(-1, K)
                        # Drop BC ONLY when better AND Q is underestimated (trust self-improvement).
                        should_filter = better_per_sample * q_under_exp           # (B, K)
                        bc_filter_exp = 1.0 - should_filter
                    else:
                        bc_filter_exp = 1.0 - better_per_sample
                else:
                    # UNGUARDED hard filter: drop the BC anchor wherever the critic
                    # scores the edited action above the base. NOTE this deliberately
                    # diverges from the official code, where False disables filtering
                    # entirely (bc_filter = ones) — the released-default setting that
                    # kept qfilt_keep pinned at 1.000 (residual ~0.003, DICE == BC)
                    # in all our pre-2026-07-04 runs. The paper's Eq. 6 filter is the
                    # guarded branch above (use_soft_q_filtering=True + q_overestimation).
                    bc_filter_exp = 1.0 - better_per_sample

                if self.always_retain_bc_loss_for_expert_data and data_source is not None:
                    expert_mask = (data_source == 1).float()                       # (B, 1)
                    expert_mask_exp = expert_mask.expand(-1, K)
                    bc_filter_exp = expert_mask_exp + (1.0 - expert_mask_exp) * bc_filter_exp

                bc_filter = bc_filter_exp.mean(dim=1, keepdim=True)               # (B, 1) -- log only

            uniform_weights = torch.ones(B, K, device=self.device) / K
            weighted_filtered_mse = uniform_weights * bc_filter_exp * mse_per_sample  # (B, K)
            filtered_bc_loss = weighted_filtered_mse.sum(dim=1).mean()

        total_loss = q_loss + self.bc_loss_weight * filtered_bc_loss

        # ---- GRD coverage repulsion across the K multi-z samples ----
        # Counteracts the -Q collapse that thins the noise->action map. Always
        # measure dispersion (cheap, no-grad); only add the gradient term when
        # repel_weight > 0 so repel_weight=0 is byte-identical to the original.
        repel_loss, action_dispersion = self._dispersion_repulsion(
            actions_samples, bandwidth_scale=self.repel_bandwidth_scale)
        if self.repel_weight > 0.0:
            total_loss = total_loss + self.repel_weight * repel_loss

        with torch.no_grad():
            residual_norm = ((actions_samples - pretrained_actions_samples) ** 2).mean().sqrt()
            avg_q_curr_log = q_current_samples.mean(dim=1, keepdim=True)
            avg_q_pre_log = q_pretrained_samples.mean(dim=1, keepdim=True)

        return {
            "actor_total": total_loss,
            "actor_q_loss": q_loss,
            "actor_residual_loss": filtered_bc_loss,
            "actor_bc_loss": filtered_bc_loss,
            "q_advantage_mean": q_advantage.mean() if not in_warmup else torch.tensor(0.0, device=self.device),
            "better_than_expert_percentage": better_percentage,
            "pretrained_q_mean": avg_q_pre_log.mean(),
            "current_q_mean": avg_q_curr_log.mean(),
            "residual_norm": residual_norm,
            "q_filtering_active": bc_filter.mean() if not in_warmup else torch.tensor(1.0, device=self.device),
            "repel_loss": repel_loss.detach(),
            "action_dispersion": action_dispersion.detach(),
        }

    def dfp_actor_loss(self, state: torch.Tensor, training_step: int = 0) -> Dict[str, torch.Tensor]:
        """Faithful DFP actor update (github.com/KAIST-Visual-AI-Group/DFP).

        Critic-RANKED top-K drift -- NOT a gradient ascent of Q through action
        selection. Pipeline (mirrors agents/drift.py `_actor_loss_online`):
          1) sample N=dfp_n_samples candidate actions from the current policy and
             score them with the critic; keep the top a=dfp_top_k by Q (no grad).
          2) generate b=dfp_gen grad-bearing particles from the current policy.
          3) DRIFT those particles toward the top-K (improvement) and toward the
             per-noise teacher action (BC/trust-region anchor, weight
             dfp_alpha_target), via the multi-scale drift_loss.
          4) add a direct -Q term on the fresh particles (weight dfp_q_weight).
        Because step 1 only RANKS candidates, it is robust to a loose/over-dispersed
        Q surface -- the exact K=1-drift failure mode that pins a gradient-based
        residual. Trains the same residual-on-teacher actor as residual mode.
        """
        assert self._teacher_fn is not None, "call attach_teacher(teacher_fn) first."
        # Lazy import so residual-mode DICE never depends on the drift-field module.
        from imitation.algos.rl.grd import drift_loss as _drift_field_loss
        B = state.shape[0]
        H, A, S = self.horizon_steps, self.action_dim, self.horizon_steps * self.action_dim
        N, Ktop, b = self.dfp_n_samples, max(1, min(self.dfp_top_k, self.dfp_n_samples)), self.dfp_gen
        dev = self.device

        def _rep(x, m):
            if x.dim() == 3:
                ne, h = x.shape[1], x.shape[2]
                return x.unsqueeze(1).expand(-1, m, -1, -1).reshape(B * m, ne, h)
            return x.unsqueeze(1).expand(-1, m, -1).reshape(B * m, -1)

        # 1) candidate pool: N samples from current policy, critic-ranked (no grad).
        with torch.no_grad():
            state_N = _rep(state, N)
            z_N = torch.randn(B * N, H, A, device=dev)
            a_N = self.get_action(state_N, z_N)                       # (B*N, H, A)
            q_N = self.critic(state_N, z_N, a_N).reshape(B, N)        # (B, N) conservative Q
            topk_idx = q_N.topk(Ktop, dim=1).indices                 # (B, Ktop)
            pos = torch.gather(a_N.reshape(B, N, S), 1,
                               topk_idx.unsqueeze(-1).expand(-1, -1, S))  # (B, Ktop, S)
            q_topk_mean = torch.gather(q_N, 1, topk_idx).mean()

        # 2) grad-bearing particles from current policy + their per-noise teacher anchor.
        state_b = _rep(state, b)
        z_b = torch.randn(B * b, H, A, device=dev)
        a_gen_flat = self.get_action(state_b, z_b)                   # (B*b, H, A) grad -> actor
        a_gen = a_gen_flat.reshape(B, b, S)
        with torch.no_grad():
            a_teacher_b = self._teacher_fn(state_b, z_b).reshape(B, b, S)   # BC-prior anchor

        # 3) drift toward top-K (improvement) + toward teacher (trust region).
        w_gen = a_gen.new_full((B, b), self.dfp_lam_rep)
        drift_pos = _drift_field_loss(a_gen, pos, a_gen.new_ones(B, Ktop), w_gen=w_gen)
        drift_tgt = _drift_field_loss(a_gen, a_teacher_b, a_gen.new_ones(B, b), w_gen=w_gen)

        # 4) direct -Q on the fresh particles (DFP keeps this term).
        q_gen = self.critic(state_b, z_b, a_gen_flat).reshape(B, b)  # grad -> actor
        q_loss = -q_gen.mean()
        if self.use_q_normalization:
            scale = self._q_norm_scale(q_gen, None)
            if scale is not None:
                q_loss = scale * q_loss

        total = drift_pos + self.dfp_alpha_target * drift_tgt + self.dfp_q_weight * q_loss

        with torch.no_grad():
            residual_norm = ((a_gen - a_teacher_b) ** 2).mean().sqrt()
            _, dispersion = self._dispersion_repulsion(a_gen.reshape(B, b, H, A))
        return {
            "actor_total": total,
            "actor_q_loss": q_loss.detach(),
            "actor_drift_pos": drift_pos.detach(),
            "actor_bc_loss": drift_tgt.detach(),
            "actor_residual_loss": drift_tgt.detach(),
            "q_advantage_mean": (q_gen.mean() - q_topk_mean).detach(),
            "better_than_expert_percentage": torch.tensor(0.0, device=dev),
            "pretrained_q_mean": q_topk_mean.detach(),
            "current_q_mean": q_gen.mean().detach(),
            "residual_norm": residual_norm,
            "q_filtering_active": torch.tensor(1.0, device=dev),
            "repel_loss": drift_pos.detach(),
            "action_dispersion": dispersion.detach(),
        }

    def field_actor_loss(self, state: torch.Tensor, *,
                         mode: str = "field_distributional",
                         q_source: str = "grad",
                         q_step: float = 0.01, bc_step: float = 0.01,
                         q_max_norm: Optional[float] = 1.0,
                         total_max_norm: Optional[float] = 1.0,
                         num_particles: Optional[int] = None,
                         num_bc_particles: int = 16,
                         radii=(0.02, 0.05, 0.2), topk: int = 4, q_temp: float = 1.0,
                         restore_step: float = 0.0, restore_radius: float = 0.0,
                         unit_normalize: bool = False, adaptive_bc: bool = False,
                         res_target: float = 0.2, lambda_lr: float = 0.1,
                         lambda_max: float = 20.0) -> Dict[str, torch.Tensor]:
        """Two-field residual actor update (replaces -Q + BC anchor in this mode).

        target r_i = sg[ r_i + q_step*V_Q + bc_step*V_BC ];  L = ||r_theta - target||^2.
          V_Q : reward field — grad_a of the SAME conservative q-normalized scalar the
                DICE actor optimizes (q_source='grad'), OR a zeroth-order drift toward
                the top-K-by-Q candidates (q_source='zeroth', survives a flat critic).
          V_BC: raw (un-normalized) drifting field toward independent frozen-BC particles
                — a RESTORING anchor that vanishes when the policy matches BC. For
                mode='field_pointwise' V_BC = -r (reproduces lambda||r||^2; equivalence check).
        Everything in normalized action space. Only the residual actor gets gradients.
        """
        from imitation.algos.dice.drift_field import (
            compute_drift_field, compute_q_gradient_field, clip_field_norm)
        assert self._teacher_fn is not None, "call attach_teacher(teacher_fn) first."
        B = state.shape[0]
        K = int(num_particles or self.num_multi_z)
        H, A, S = self.horizon_steps, self.action_dim, self.horizon_steps * self.action_dim
        dev = self.device

        def _rep(x, m):
            if x.dim() == 3:
                ne, h = x.shape[1], x.shape[2]
                return x.unsqueeze(1).expand(-1, m, -1, -1).reshape(B * m, ne, h)
            return x.unsqueeze(1).expand(-1, m, -1).reshape(B * m, -1)

        # ---- current detached particles a_i = a0(s,z_i) + r_theta(s,z_i) ----
        state_K = _rep(state, K)
        z = torch.randn(B * K, H, A, device=dev)
        with torch.no_grad():
            base = self._teacher_fn(state_K, z)                  # a0  (B*K,H,A)
            old_res = self.actor(state_K, z)                     # r_theta (detached source)
            cur = base + old_res                                 # current particles
            q_cur = self.critic(state_K, z, cur)                 # (B*K,1) conservative
            q_scale = (q_cur.abs().mean().clamp_min(1e-8)
                       if self.use_q_normalization else cur.new_tensor(1.0))

        # ---- reward field V_Q ----
        if q_source == "grad":
            qf = compute_q_gradient_field(self.critic, state_K, z, cur,
                                          q_scale=q_scale, max_norm=q_max_norm)
            q_field = qf.field                                   # (B*K,H,A)
            q_raw_norm = qf.raw_norm.mean()
        elif q_source == "zeroth":
            cur_b = cur.reshape(B, K, S)
            q_b = q_cur.reshape(B, K)
            Ktop = max(1, min(int(topk), K))
            top_idx = q_b.topk(Ktop, dim=1).indices
            pos = torch.gather(cur_b, 1, top_idx.unsqueeze(-1).expand(-1, -1, S))
            out = compute_drift_field(cur_b, pos, cur_b, radii=radii,
                                      exclude_negative_self=True, normalize_per_radius=False)
            q_field = clip_field_norm(out.field.reshape(B * K, 1, S), q_max_norm).reshape(B * K, H, A)
            q_raw_norm = out.field.norm(dim=-1).mean()
        elif q_source == "tilted":
            # exp(Q/tau)-tilted transport: ALL K particles attract with weight
            # softmax(adv/tau)*K (mean weight 1, matching the uniform repulsors),
            # so the field moves the cloud toward its own advantage-tilted density.
            cur_b = cur.reshape(B, K, S)
            q_b = q_cur.reshape(B, K)
            adv = (q_b - q_b.mean(dim=1, keepdim=True)) / q_b.std(dim=1, keepdim=True).clamp_min(1e-6)
            w_pos = torch.softmax(adv / max(q_temp, 1e-6), dim=1) * K
            out = compute_drift_field(cur_b, cur_b, cur_b, radii=radii, w_pos=w_pos,
                                      exclude_negative_self=True, exclude_positive_self=True,
                                      normalize_per_radius=False)
            q_field = clip_field_norm(out.field.reshape(B * K, 1, S), q_max_norm).reshape(B * K, H, A)
            q_raw_norm = out.field.norm(dim=-1).mean()
        else:
            raise ValueError(f"unknown q_source: {q_source}")

        # ---- BC field V_BC ----
        if mode == "field_pointwise":
            bc_field = -old_res                                  # pointwise: V_BC = -r
        elif mode == "field_distributional":
            state_bc = _rep(state, num_bc_particles)
            z_bc = torch.randn(B * num_bc_particles, H, A, device=dev)
            with torch.no_grad():
                bc_actions = self._teacher_fn(state_bc, z_bc).reshape(B, num_bc_particles, S)
            cur_b = cur.reshape(B, K, S)
            out_bc = compute_drift_field(cur_b, bc_actions, cur_b, radii=radii,
                                         exclude_negative_self=True, normalize_per_radius=False)
            bc_field = out_bc.field.reshape(B * K, H, A)
        else:
            raise ValueError(f"unknown field mode: {mode}")

        # ---- combine, clip total displacement, build detached residual target ----
        assert cur.shape == q_field.shape == bc_field.shape
        assert torch.isfinite(q_field).all() and torch.isfinite(bc_field).all()
        rnorm = old_res.reshape(B * K, -1).pow(2).mean().sqrt()   # current residual rms

        # Unified field: reward + BC as ONE velocity V = q_hat + lambda*bc_hat, both
        # optionally unit-RMS so lambda is a dimensionless PG/BC trade-off. lambda is
        # either fixed (bc_step) or a trust-region dual regulated toward res_target.
        if unit_normalize:
            def _unit(f):
                rms = f.reshape(B * K, -1).pow(2).mean(-1, keepdim=True).sqrt().clamp_min(1e-8)
                return f / rms.view(B * K, 1, 1)
            q_use, bc_use = _unit(q_field), _unit(bc_field)
        else:
            q_use, bc_use = q_field, bc_field

        # restore_step: pointwise -r pull that stays active at any range (the kernel
        # fields vanish beyond max(radii), so without it the target regression can
        # random-walk the residual off the data manifold). restore_radius > 0 makes
        # it a dead-zone restore: no damping inside the trust radius, pull back only
        # the excess norm beyond it.
        if restore_radius > 0.0:
            rn = old_res.reshape(B * K, -1).norm(dim=-1)
            gate = (1.0 - restore_radius / rn.clamp_min(1e-8)).clamp(min=0.0)
            restore_delta = -restore_step * gate.reshape(B * K, 1, 1) * old_res
        else:
            restore_delta = -restore_step * old_res

        lam = float(self._field_lambda) if adaptive_bc else float(bc_step / max(q_step, 1e-8))
        if unit_normalize or adaptive_bc:
            # single-step unified update, eta = q_step scales the whole field
            V = q_use + lam * bc_use
            q_delta = q_step * q_use
            bc_delta = (q_step * lam) * bc_use
            total_delta = clip_field_norm((q_step * V + restore_delta).reshape(B * K, 1, S),
                                          total_max_norm).reshape(B * K, H, A)
        else:
            # legacy: independent step sizes
            q_delta = q_step * q_use
            bc_delta = bc_step * bc_use
            total_delta = clip_field_norm((q_delta + bc_delta + restore_delta).reshape(B * K, 1, S),
                                          total_max_norm).reshape(B * K, H, A)
        residual_target = (old_res + total_delta).detach()

        # trust-region dual: MULTIPLICATIVE update on a SMOOTHED residual (scale-free,
        # stays positive) — raise lambda (more BC pull) when the residual overshoots
        # res_target, lower it when the policy is pinned too close to BC. EMA filters
        # the transient so the fast controller doesn't fight the slow residual plant.
        if adaptive_bc:
            with torch.no_grad():
                r_now = rnorm.item()
                # slow EMA (tc~100 steps) so the controller is slower than the residual
                # plant; tiny per-step multiplicative nudge with anti-windup clamp.
                self._field_res_ema = (r_now if self._field_res_ema is None
                                       else 0.99 * self._field_res_ema + 0.01 * r_now)
                rel = self._field_res_ema / max(res_target, 1e-6) - 1.0   # >0 => too far
                step = max(-0.02, min(0.02, lambda_lr * rel))            # gentle, clipped
                self._field_lambda = float(min(lambda_max, max(1e-3, lam * (2.718281828 ** step))))

        pred_res = self.actor(state_K, z)                        # grad -> actor
        loss = F.mse_loss(pred_res, residual_target)

        with torch.no_grad():
            qn = q_field.reshape(B * K, -1).norm(dim=-1)
            bn = bc_field.reshape(B * K, -1).norm(dim=-1)
            cos = F.cosine_similarity(q_field.reshape(B * K, -1),
                                      bc_field.reshape(B * K, -1), dim=-1)
            residual_norm = old_res.pow(2).mean().sqrt()
            _, dispersion = self._dispersion_repulsion(cur.reshape(B, K, H, A))
        metrics = {
            "actor_total": loss,
            "field_loss": loss.detach(),
            "field_q_norm": qn.mean(),
            "field_q_raw_norm": q_raw_norm.detach(),
            "field_bc_norm": bn.mean(),
            "field_q_delta_norm": q_delta.reshape(B * K, -1).norm(dim=-1).mean(),
            "field_bc_delta_norm": bc_delta.reshape(B * K, -1).norm(dim=-1).mean(),
            "field_total_delta_norm": total_delta.reshape(B * K, -1).norm(dim=-1).mean(),
            "field_q_bc_cosine": cos.mean(),
            "current_q_mean": q_cur.mean(),
            "residual_norm": residual_norm,
            "field_lambda": torch.tensor(float(self._field_lambda), device=dev),
            # compat keys read by dice_train.py logging
            "actor_q_loss": loss.detach(),
            "actor_bc_loss": bn.mean(),
            "actor_residual_loss": bn.mean(),
            "q_filtering_active": torch.tensor(1.0, device=dev),
            "action_dispersion": dispersion.detach(),
            "repel_loss": torch.tensor(0.0, device=dev),  # no separate repel term in field mode
        }
        return metrics

    def _critic_td_loss(self, q_pred: torch.Tensor, target_q: torch.Tensor,
                        online_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Per-Q TD loss with optional online-only masking. Matches official."""
        if self.td_loss == "mse":
            if online_mask is not None:
                per_sample = F.mse_loss(q_pred, target_q, reduction="none")  # (B, 1)
                return (per_sample * online_mask).mean()
            return F.mse_loss(q_pred, target_q)
        if self.td_loss == "huber":
            if online_mask is not None:
                per_sample = F.smooth_l1_loss(q_pred, target_q, reduction="none", beta=1.0)
                return (per_sample * online_mask).mean()
            return F.smooth_l1_loss(q_pred, target_q, beta=1.0)
        if self.td_loss == "bce":
            assert (target_q >= 0).all() and (target_q <= 1).all(), \
                f"BCE TD requires targets in [0,1]; got [{target_q.min():.3f},{target_q.max():.3f}]"
            if online_mask is not None:
                per_sample = F.binary_cross_entropy_with_logits(q_pred, target_q, reduction="none")
                return (per_sample * online_mask).mean()
            return F.binary_cross_entropy_with_logits(q_pred, target_q)
        raise ValueError(f"Unknown td_loss: {self.td_loss}")

    def critic_loss(self, state: torch.Tensor, noise: Optional[torch.Tensor],
                    action: torch.Tensor, target_q: torch.Tensor,
                    data_source: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        qs = self.critic(state, noise, action, return_all=True)
        online_mask = None
        if self.disable_td_loss_for_expert_data and data_source is not None:
            online_mask = (data_source == 0).float()  # (B, 1)
        per_q = [self._critic_td_loss(q, target_q, online_mask) for q in qs]
        td_total = sum(per_q)

        out = {}
        for i, q in enumerate(qs):
            if i < 3:
                out[f"q{i+1}_loss"] = per_q[i]
            out[f"q{i}_mean"] = q.mean().detach()
        out["critic_td_loss"] = td_total.detach()

        # CQL-style conservative penalty.
        # Discriminator probe on hard-8 z-cond run (iter 10 vs iter 40): residual
        # barely moves (~3e-4) but Q on the same (s,z,a) drifts up by ~1.2; eval
        # decays from 0.700 -> 0.588. The leak is Q-level inflation, not residual.
        # Penalty: push Q DOWN at the policy's current action, UP at the data
        # action. Policy action is computed under no_grad so this term only
        # gradients into critic params (the actor still gets its -Q signal in
        # actor_loss). Off when cql_weight == 0.
        if self.cql_weight > 0.0:
            with torch.no_grad():
                a_policy = self.get_action(state, noise)
            qs_at_policy = self.critic(state, noise, a_policy, return_all=True)
            q_policy_mean = sum(q.mean() for q in qs_at_policy) / len(qs_at_policy)
            q_data_mean = sum(q.mean() for q in qs) / len(qs)
            cql_term = self.cql_weight * (q_policy_mean - q_data_mean)
            total = td_total + cql_term
            out["critic_cql_term"] = cql_term.detach()
            out["critic_q_at_policy"] = q_policy_mean.detach()
            out["critic_q_at_data"] = q_data_mean.detach()
        else:
            total = td_total
        out["critic_loss"] = total
        return out

    # ------------------------------------------------------------------
    # End-to-end loss
    # ------------------------------------------------------------------
    def loss(self, state: torch.Tensor, noise: torch.Tensor, action: torch.Tensor,
             next_state: torch.Tensor, reward: torch.Tensor, done: torch.Tensor,
             gamma: float = 0.99,
             training_step: int = 0,
             q_overestimation: Optional[torch.Tensor] = None,
             n_steps: Optional[torch.Tensor] = None,
             data_source: Optional[torch.Tensor] = None,
             **kwargs) -> Dict[str, torch.Tensor]:
        """Compute joint actor + critic loss. Mirrors official.

        Note: the official RE-SAMPLES noise inside this method (the buffer's stored
        noise is effectively unused except for q_depends_on_noise=True). We accept
        the buffer's noise but it is only consumed by the critic when
        q_depends_on_noise=True; that path is OFF by default, matching official.
        """
        B = state.shape[0]
        H = self.horizon_steps
        A = self.action_dim

        # ---- target Q ----
        with torch.no_grad():
            if not self.multi_sample_next_noise:
                next_noise = torch.randn(B, H, A, device=self.device)
                next_actions = self.get_action(next_state, next_noise)
                target_next_q = self.target_critic(next_state, next_noise, next_actions)
            else:
                Kn = self.num_next_noise_samples
                next_noise_samples = torch.randn(Kn, B, H, A, device=self.device)
                if next_state.dim() == 3:
                    ne, h = next_state.shape[1], next_state.shape[2]
                    next_state_rep = next_state.unsqueeze(0).expand(Kn, -1, -1, -1).reshape(Kn * B, ne, h)
                else:
                    next_state_rep = next_state.unsqueeze(0).expand(Kn, -1, -1).reshape(Kn * B, -1)
                next_noise_flat = next_noise_samples.reshape(Kn * B, H, A)
                next_actions = self.get_action(next_state_rep, next_noise_flat)
                target_q_samples = self.target_critic(next_state_rep, next_noise_flat, next_actions)
                target_next_q = target_q_samples.reshape(Kn, B, 1).mean(dim=0)

            if self.use_n_step and n_steps is not None:
                gamma_effective = gamma ** n_steps.float()  # (B, 1)
            else:
                gamma_effective = gamma

            target_q = reward + gamma_effective * (1.0 - done.float()) * target_next_q

        # ---- actor + critic loss ----
        # NOTE: the official passes buffer noise to critic_loss, but with
        # q_depends_on_noise=False (the default) the noise is unused, so the
        # buffer's noise vs a fresh sample doesn't matter for the critic.
        if self.actor_mode == "dfp":
            actor_d = self.dfp_actor_loss(state, training_step=training_step)
        elif self.actor_mode in ("field_pointwise", "field_distributional"):
            fc = self.field_cfg
            actor_d = self.field_actor_loss(
                state, mode=self.actor_mode,
                q_source=fc.get("q_source", "grad"),
                q_step=fc.get("q_step_size", 0.01), bc_step=fc.get("bc_step_size", 0.01),
                q_max_norm=fc.get("q_max_norm", 1.0),
                total_max_norm=fc.get("total_max_norm", 1.0),
                num_particles=fc.get("num_particles", None),
                num_bc_particles=fc.get("num_bc_particles", 16),
                radii=tuple(fc.get("radii", (0.02, 0.05, 0.2))),
                topk=fc.get("topk", 4),
                q_temp=fc.get("q_temp", 1.0),
                restore_step=fc.get("restore_step_size", 0.0),
                restore_radius=fc.get("restore_radius", 0.0),
                unit_normalize=fc.get("unit_normalize", False),
                adaptive_bc=fc.get("adaptive_bc", False),
                res_target=fc.get("res_target", 0.2),
                lambda_lr=fc.get("lambda_lr", 0.1),
                lambda_max=fc.get("lambda_max", 20.0),
            )
        else:
            actor_d = self.actor_loss(
                state, training_step=training_step,
                q_overestimation=q_overestimation, data_source=data_source,
            )
        critic_d = self.critic_loss(state, noise, action, target_q, data_source=data_source)

        total = actor_d["actor_total"] + self.critic_weight * critic_d["critic_loss"]
        return {
            "total_loss": total,
            **actor_d,
            **critic_d,
            "target_q_mean": target_q.mean().detach(),
            "reward_mean": reward.float().mean().detach(),
            "done_frac": done.float().mean().detach(),
        }

    # ------------------------------------------------------------------
    # Target net update
    # ------------------------------------------------------------------
    def update_target_networks(self, tau: float = 0.005):
        with torch.no_grad():
            for tp, p in zip(self.target_critic.parameters(), self.critic.parameters()):
                tp.data.mul_(1.0 - tau).add_(tau * p.data)

    # Back-compat alias
    def update_target_critic(self, tau: float = 0.005):
        self.update_target_networks(tau=tau)
