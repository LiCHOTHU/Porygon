"""CPU tests of the replay/actor adapters; no real-robot interfaces."""
import unittest
import torch

from real_robot.prepare_offline_replay import blocks
from real_robot.residual_policy import PrefixActor, PrefixCritic
from imitation.algos.dice.distill_rl import DistilledRLModel


class OfflineTests(unittest.TestCase):
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
