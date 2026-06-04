"""DICE online rollout collector for LIBERO.

Drives one in-process env (DummyVectorEnv size 1). At each chunk boundary:
  - encode obs through the frozen BC encoder -> cond (1, num_enc, hidden)
  - sample noise (1, H, A); pick a noise via the configured exploration strategy
  - student.get_action(cond, noise) -> a_total = a_teacher + residual (unclamped)
  - clamp + unnormalize + execute chunk
  - buffer per-chunk-decision rows; at episode end push the whole trajectory to
    the replay (which computes n_step lookahead + mc_return + dones).

Exploration strategy is delegated to DistilledRLModel.get_exploration_action; we
just pass cond + training_step + strategy through.
"""

import numpy as np
import torch

import imitation.envs.libero.wrappers as lw


class DiceCollector:
    def __init__(self, env_runner, bc_policy, student_model, device,
                 max_episode_length=None,
                 use_teacher_for_collect: bool = False,
                 online_explore_strategy: str = "standard",
                 num_exploration_samples: int = 10):
        self.runner = env_runner
        self.bc = bc_policy
        self.student = student_model
        self.device = device

        self.benchmark = env_runner.benchmark
        self.env_factory = env_runner.env_factory
        self.env_names = env_runner.env_names
        self.frame_stack = env_runner.frame_stack
        self.max_episode_length = max_episode_length or env_runner.max_episode_length
        self.action_horizon = bc_policy.action_horizon
        self.chunk_size = bc_policy.chunk_size
        self.action_dim = bc_policy.network_action_dim

        self._env = None
        self._env_task = None
        self.use_teacher_for_collect = bool(use_teacher_for_collect)
        self.online_explore_strategy = str(online_explore_strategy)
        self.num_exploration_samples = int(num_exploration_samples)

    def _build_env(self, task_index: int):
        env_fn = lambda: lw.LiberoFrameStack(
            self.env_factory(task_id=task_index, benchmark=self.benchmark), self.frame_stack
        )
        self._env = lw.LiberoVectorWrapper(env_fn, 1)
        self._env_task = task_index

    def _task_emb(self, task_index: int):
        te = self.benchmark.get_task_emb(task_index)
        return {k: v.repeat(1, 1) for k, v in te.items()}

    @torch.no_grad()
    def _encode_cond(self, obs, task_index: int, task_emb):
        batch = self.bc._make_batch({k: v for k, v in obs.items()}, task_index, **task_emb)
        batch = self.bc.preprocess_input(batch, train_mode=False)
        cond = self.bc.get_cond(batch)  # (1, num_enc, hidden)
        return cond, batch

    @torch.no_grad()
    def _act(self, cond: torch.Tensor, training_step: int = 0):
        """Returns (a_norm, noise), both (1, chunk, A), on device.
        - During warmup collect: BC teacher's Euler output (matches official's
          warmstart phase that seeds the replay with prior data).
        - Standard strategy: random noise -> student.get_action.
        - Other strategies: delegate to student.get_exploration_action."""
        if self.use_teacher_for_collect:
            noise = torch.randn(1, self.chunk_size, self.action_dim, device=self.device)
            self.bc.eval()
            x = noise
            t = torch.zeros(1, device=self.device)
            dt = 1.0 / self.bc.num_inference_steps
            enc_cache = self.bc.velocity_net.forward_enc(cond)
            for _ in range(self.bc.num_inference_steps):
                v = self.bc.velocity_net.forward_dec(x, t, enc_cache)
                x = x + dt * v
                t = t + dt
            a_norm = torch.clamp(x, -1, 1)
            return a_norm, noise

        if self.online_explore_strategy == "standard":
            noise = torch.randn(1, self.chunk_size, self.action_dim, device=self.device)
            a_total = self.student.get_action(cond, noise)
        else:
            a_total, noise = self.student.get_exploration_action(
                cond, num_samples=self.num_exploration_samples,
                exploration_strategy=self.online_explore_strategy,
                training_step=training_step,
            )
        a_norm = torch.clamp(a_total, -1, 1)
        return a_norm, noise

    @torch.no_grad()
    def _unnormalize_and_execute_setup(self, a_norm: torch.Tensor, batch: dict) -> list:
        a = self.bc.normalizer.unnormalize({"actions": a_norm.cpu().numpy()})["actions"]
        batch["actions"] = torch.tensor(a, device=self.device)
        a = self.bc.postprocess_actions(batch)["actions"].cpu().numpy()
        a = np.transpose(a, (1, 0, 2))  # (chunk, 1, A)
        return list(a[: self.action_horizon])

    @torch.no_grad()
    def rollout_episode(self, task_index: int, init_idx: int, replay,
                        training_step: int = 0):
        """Run one episode, push the whole trajectory to replay at end.
        Returns: (success: bool, n_decisions: int).
        """
        if self._env is None or self._env_task != task_index:
            self._build_env(task_index)
        task_emb = self._task_emb(task_index)

        all_init = self.benchmark.get_task_init_states(task_index)
        init_states = all_init[np.array([init_idx % all_init.shape[0]])]
        obs, _ = self._env.reset(init_states=init_states)

        # Per-chunk-decision trajectory accumulator.
        traj = []
        success = False

        queue: list = []
        for step in range(self.max_episode_length):
            if len(queue) == 0:
                cond, batch = self._encode_cond(obs, task_index, task_emb)
                a_norm, noise = self._act(cond, training_step=training_step)
                queue = self._unnormalize_and_execute_setup(a_norm, batch)
                traj.append({
                    "cond": cond.squeeze(0).cpu(),     # (num_enc, hidden)
                    "noise": noise.squeeze(0).cpu(),   # (chunk, A)
                    "action": a_norm.squeeze(0).cpu(), # (chunk, A)
                    "reward": 0.0,                     # set below if this chunk terminates with success
                    "done": False,
                })

            act = torch.tensor(queue.pop(0))
            act = self.bc.final_postprocess_actions(act).to(torch.float32).cpu().numpy()
            obs, reward, terminated, truncated, info = self._env.step(act)

            now_success = bool(info[0]["success"])
            if now_success:
                success = True

            chunk_boundary = (len(queue) == 0)
            episode_done = success  # treat success as terminal (sparse-binary, matches LIBERO)

            if chunk_boundary or episode_done:
                # Close the current chunk decision with its terminal flag + reward.
                traj[-1]["done"] = bool(episode_done)
                traj[-1]["reward"] = 1.0 if (episode_done and success) else 0.0

            if episode_done:
                queue.clear()
                break

        # Push trajectory to replay (computes n_step + mc_return).
        replay.add_trajectory(traj, data_source=0)
        return bool(success), len(traj)
