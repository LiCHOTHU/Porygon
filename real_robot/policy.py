import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18

from imitation.algos.utils.diffusion_policy_utils.dit_modules import DiTNoiseNet


class DualCameraEncoder(nn.Module):
    def __init__(self, hidden_dim, proprio_dim, dropout=0.1, pretrained=True):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = resnet18(weights=weights)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        self.visual_projection = nn.Conv2d(512, hidden_dim, 1)
        self.proprio_projection = nn.Sequential(
            nn.Linear(proprio_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.language_projection = nn.Sequential(
            nn.Linear(512, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "image_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "image_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def forward(self, chest_rgb, wrist_rgb, proprio, prompt_embedding):
        batch = chest_rgb.shape[0]
        images = torch.cat((chest_rgb, wrist_rgb), dim=0).float().div_(255.0)
        images = (images - self.image_mean) / self.image_std
        features = self.visual_projection(self.backbone(images))
        tokens = features.flatten(2).transpose(1, 2)
        chest_tokens, wrist_tokens = tokens[:batch], tokens[batch:]
        language = self.language_projection(prompt_embedding)
        visual = torch.cat((chest_tokens, wrist_tokens), dim=1) + language[:, None]
        proprio_token = self.proprio_projection(proprio)[:, None]
        return self.dropout(torch.cat((visual, proprio_token, language[:, None]), dim=1))


class RealRobotFlowPolicy(nn.Module):
    """Porygon conditional flow matching with joint-space actions."""

    def __init__(self, cfg, stats, prompt_embedding, pretrained_backbone=True):
        super().__init__()
        self.chunk_size = int(cfg["chunk_size"])
        self.action_dim = int(cfg["action_dim"])
        self.sigma_min = float(cfg["flow_sigma_min"])
        self.flow_alpha = float(cfg["flow_alpha"])
        self.flow_beta = float(cfg["flow_beta"])
        self.inference_steps = int(cfg["inference_steps"])
        hidden_dim = int(cfg["hidden_dim"])
        self.encoder = DualCameraEncoder(
            hidden_dim, int(cfg["proprio_dim"]), float(cfg["dropout"]), pretrained_backbone
        )
        self.velocity_net = DiTNoiseNet(
            ac_dim=self.action_dim,
            ac_chunk=self.chunk_size,
            time_dim=128,
            hidden_dim=hidden_dim,
            num_blocks=int(cfg["num_blocks"]),
            dim_feedforward=int(cfg["feedforward_dim"]),
            dropout=float(cfg["dropout"]),
            nhead=int(cfg["num_heads"]),
            activation="gelu",
        )
        for name in ("action_min", "action_max", "proprio_min", "proprio_max"):
            self.register_buffer(name, stats[name].clone().float())
        self.register_buffer("prompt_embedding", prompt_embedding.clone().float().view(1, 512))

    @staticmethod
    def _normalize(value, minimum, maximum):
        scale = (maximum - minimum).clamp_min(1e-6)
        return 2.0 * (value - minimum) / scale - 1.0

    @staticmethod
    def _unnormalize(value, minimum, maximum):
        return (value + 1.0) * 0.5 * (maximum - minimum) + minimum

    def get_conditioning(self, batch):
        proprio = self._normalize(batch["proprio"], self.proprio_min, self.proprio_max)
        prompt = self.prompt_embedding.expand(proprio.shape[0], -1)
        return self.encoder(batch["chest_rgb"], batch["wrist_rgb"], proprio, prompt)

    def _sample_time(self, batch_size, device):
        distribution = torch.distributions.Beta(self.flow_alpha, self.flow_beta)
        z = distribution.sample((batch_size,)).to(device)
        return (1.0 - self.sigma_min) * (1.0 - z)

    def compute_loss(self, batch):
        actions = self._normalize(batch["actions"], self.action_min, self.action_max).clamp(-1, 1)
        condition = self.get_conditioning(batch)
        t = self._sample_time(actions.shape[0], actions.device)
        x0 = torch.randn_like(actions)
        psi_t = (
            (1.0 - (1.0 - self.sigma_min) * t[:, None, None]) * x0
            + t[:, None, None] * actions
        )
        _, predicted_velocity = self.velocity_net(psi_t, t, condition)
        target_velocity = actions - (1.0 - self.sigma_min) * x0
        return torch.mean((predicted_velocity - target_velocity) ** 2)

    @torch.inference_mode()
    def sample_actions(self, batch):
        condition = self.get_conditioning(batch)
        batch_size = condition.shape[0]
        actions = torch.randn(
            batch_size, self.chunk_size, self.action_dim, device=condition.device
        )
        encoder_cache = self.velocity_net.forward_enc(condition)
        dt = 1.0 / self.inference_steps
        t = torch.zeros(batch_size, device=condition.device, dtype=condition.dtype)
        for _ in range(self.inference_steps):
            actions.add_(self.velocity_net.forward_dec(actions, t, encoder_cache), alpha=dt)
            t.add_(dt)
        actions.clamp_(-1, 1)
        return self._unnormalize(actions, self.action_min, self.action_max)
