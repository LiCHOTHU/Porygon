"""CPU tests of the replay/actor adapters; no real-robot interfaces."""
import unittest
import csv
import json
from pathlib import Path
import tempfile
import cv2
import numpy as np
import torch

from real_robot.prepare_offline_replay import blocks, episode
from real_robot.data import LEFT_COLUMNS
from real_robot.residual_policy import PrefixActor, PrefixCritic
from imitation.algos.dice.distill_rl import DistilledRLModel


class OfflineTests(unittest.TestCase):
    def test_terminal_reward_survives_sensor_shutdown_before_last_command(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            for name, times in (("follower_joint.csv", np.linspace(96, 99.995, 400)),
                                ("policy_commands.csv", np.linspace(97, 100, 48))):
                with (path / name).open("w") as f:
                    writer = csv.writer(f)
                    writer.writerow(["timestamp"] + LEFT_COLUMNS)
                    writer.writerows([[t] + [0.1]*8 for t in times])
            for view in ("chest", "left_wrist"):
                writer = cv2.VideoWriter(str(path / f"{view}_rgb_test.avi"),
                                         cv2.VideoWriter_fourcc(*"XVID"), 15, (64, 64))
                for _ in range(64):
                    writer.write(np.zeros((64, 64, 3), np.uint8))
                writer.release()
            (path / "take_meta.json").write_text(json.dumps({
                "ended_at_epoch_s": 100.1, "evaluation": {"policy_ended_at_epoch_s": 100.01}}))
            records = episode(path, False, True, 8)
            self.assertTrue(records[-1]["done"])
            self.assertEqual(records[-1]["reward"], 1)
            np.testing.assert_equal(records[-1]["proprio"][0], records[-1]["proprio"][1])

    def test_terminal_window_contains_only_real_commands(self):
        self.assertEqual(blocks(19, 8), [0, 8, 11])
        self.assertEqual(blocks(7, 8), [])

    def test_zero_residual_and_unexecuted_suffix(self):
        actor = PrefixActor(6, {"action_dim": 2, "chunk_size": 5},
                            {"actor_hidden": [8], "execution_horizon": 2})
        state, noise = torch.randn(3, 6), torch.randn(3, 5, 2)
        self.assertEqual(actor(state, noise).abs().sum(), 0)
        actor.mlp[-2].bias.data.fill_(1)
        self.assertEqual(actor(state, noise)[:, 2:].abs().sum(), 0)

    def test_critic_does_not_learn_from_unexecuted_suffix(self):
        critic = PrefixCritic(6, 2, {"execution_horizon": 2,
                                    "critic_hidden": [8], "ensemble_size": 2})
        state, noise, action = torch.randn(3, 6), torch.randn(3, 5, 2), torch.randn(3, 5, 2)
        other = action.clone()
        other[:, 2:] += 100
        torch.testing.assert_close(critic(state, noise, action), critic(state, noise, other))

    def test_cast_actor_backward_does_not_update_critic(self):
        model = DistilledRLModel(state_dim=6, action_dim=2, horizon_steps=5,
                                 actor_hidden=[8], critic_hidden=[8], ensemble_size=2,
                                 zero_final_layer=True, device="cpu")
        model.attach_teacher(lambda state, noise: noise.clamp(-1, 1))
        result = model.field_actor_loss(torch.randn(2, 6), mode="field_pointwise",
                                        num_particles=2, q_step=1, bc_step=.05,
                                        total_max_norm=.15, restore_step=1,
                                        restore_radius=.05, restore_radius_rms=True)
        result["actor_total"].backward()
        self.assertTrue(all(p.grad is None for p in model.critic.parameters()))
        self.assertLessEqual(float(result["field_total_delta_norm"]), .150001)
        self.assertTrue(any(p.grad is not None for p in model.actor.parameters()))


if __name__ == "__main__":
    unittest.main()
