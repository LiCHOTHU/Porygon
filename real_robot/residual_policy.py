"""Porygon DICE/CAST adapters for an executed prefix of a frozen FM chunk."""
import copy
import torch
import torch.nn as nn

from imitation.algos.dice.distill_rl import DistilledActor, DistilledCritic, DistilledRLModel
from real_robot.policy import RealRobotFlowPolicy


class PrefixActor(DistilledActor):
    def __init__(self, state_dim, base_config, rl_config):
        super().__init__(state_dim, base_config["action_dim"], base_config["chunk_size"],
                         hidden_dims=rl_config["actor_hidden"], zero_final_layer=True)
        mask = torch.zeros(1, base_config["chunk_size"], 1)
        mask[:, :rl_config["execution_horizon"]] = 1
        self.register_buffer("execution_mask", mask)

    def forward(self, state, noise):
        return super().forward(state, noise) * self.execution_mask


class PrefixCritic(DistilledCritic):
    def __init__(self, state_dim, action_dim, config):
        super().__init__(state_dim, action_dim, config["execution_horizon"],
                         hidden_dims=config["critic_hidden"], ensemble_size=config["ensemble_size"],
                         q_depends_on_noise=False, conservative="min")
        self.execution_horizon = config["execution_horizon"]

    def _features(self, state, noise, action):
        return super()._features(state, noise, action[:, :self.execution_horizon])


class FrozenTeacher:
    def __init__(self, base):
        self.base = base

    @torch.no_grad()
    def __call__(self, condition, noise):
        x = noise.clone()
        cache = self.base.velocity_net.forward_enc(condition)
        time = torch.zeros(len(x), device=x.device)
        dt = 1 / self.base.inference_steps
        for _ in range(self.base.inference_steps):
            x = x + dt * self.base.velocity_net.forward_dec(x, time, cache)
            time = time + dt
        return x.clamp(-1, 1)


def load_base(checkpoint, device="cuda"):
    state = checkpoint["model"]
    stats = {k: state[k] for k in ("action_min", "action_max", "proprio_min", "proprio_max")}
    base = RealRobotFlowPolicy(checkpoint["config"], stats, state["prompt_embedding"],
                               pretrained_backbone=False).to(device)
    base.load_state_dict(state)
    base.eval()
    base.requires_grad_(False)
    return base


def build_student(base, base_config, config, mode, state_dim, device="cuda"):
    student = DistilledRLModel(
        state_dim=state_dim, action_dim=base.action_dim, horizon_steps=base.chunk_size,
        actor_hidden=config["actor_hidden"], critic_hidden=config["critic_hidden"],
        ensemble_size=config["ensemble_size"], num_multi_z=config["num_particles"],
        bc_loss_weight=config["bc_loss_weight"], cql_weight=config["cql_weight"],
        use_q_normalization=True, q_depends_on_noise=False, clip_action=False,
        zero_final_layer=True, use_n_step=True, n_step=1,
        always_retain_bc_loss_for_expert_data=True,
        actor_mode="residual" if mode == "dice_rl" else "field_pointwise",
        field_cfg=config["field"], device=device)
    student.actor = PrefixActor(state_dim, base_config, config).to(device)
    student.critic = PrefixCritic(state_dim, base.action_dim, config).to(device)
    student.target_critic = copy.deepcopy(student.critic).requires_grad_(False)
    student.attach_teacher(FrozenTeacher(base))
    return student


class ResidualInferenceModel(nn.Module):
    """Only the frozen FM and residual are needed for deployment; no critic."""
    def __init__(self, checkpoint, device):
        super().__init__()
        self.base = load_base(checkpoint, device)
        self.actor = PrefixActor(checkpoint["state_dim"], checkpoint["config"], checkpoint["rl_config"]).to(device)
        self.actor.load_state_dict(checkpoint["residual_state"])
        self.chunk_size = self.base.chunk_size
        self.required_execution_horizon = checkpoint["rl_config"]["execution_horizon"]
        self.teacher = FrozenTeacher(self.base)
        self.eval()

    @property
    def proprio_min(self):
        return self.base.proprio_min

    @property
    def proprio_max(self):
        return self.base.proprio_max

    @torch.inference_mode()
    def sample_actions(self, batch):
        condition = self.base.get_conditioning(batch)
        noise = torch.randn(len(condition), self.base.chunk_size, self.base.action_dim, device=condition.device)
        actions = self.teacher(condition, noise) + self.actor(condition, noise)
        return self.base._unnormalize(actions, self.base.action_min, self.base.action_max)
