from pathlib import Path

import numpy as np
import torch

from real_robot.policy import RealRobotFlowPolicy


class JigglypuffPolicy:
    """Deployment wrapper. Inputs are RGB HWC arrays and an 8D left-arm state."""

    def __init__(self, checkpoint, device="cuda"):
        if not isinstance(checkpoint, dict):
            checkpoint = torch.load(checkpoint, map_location=device, weights_only=False)
        cfg = checkpoint["config"]
        if checkpoint.get("policy_type") == "offline_residual":
            from real_robot.residual_policy import ResidualInferenceModel
            self.model = ResidualInferenceModel(checkpoint, device)
            self.device = torch.device(device)
            self.image_size = int(cfg["image_size"])
            return
        state = checkpoint["model"]
        stats = {name: state[name] for name in (
            "action_min", "action_max", "proprio_min", "proprio_max"
        )}
        prompt = state["prompt_embedding"].squeeze(0)
        self.model = RealRobotFlowPolicy(
            cfg, stats, prompt, pretrained_backbone=False
        ).to(device)
        self.model.load_state_dict(state)
        self.model.eval()
        self.device = torch.device(device)
        self.image_size = int(cfg["image_size"])

    def _image(self, rgb):
        import cv2

        rgb = cv2.resize(rgb, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        return torch.from_numpy(rgb.copy()).permute(2, 0, 1).unsqueeze(0)

    def predict_chunk(self, chest_rgb, wrist_rgb, left_joint_and_gripper):
        batch = {
            "chest_rgb": self._image(chest_rgb).to(self.device),
            "wrist_rgb": self._image(wrist_rgb).to(self.device),
            "proprio": torch.as_tensor(
                left_joint_and_gripper, dtype=torch.float32, device=self.device
            ).view(1, 8),
        }
        return self.model.sample_actions(batch)[0].cpu().numpy()
