"""DICE online rollout collector for LIBERO.

Drives one in-process env (DummyVectorEnv size 1). At each chunk boundary:
  - encode obs through the frozen BC encoder -> cond (1, num_enc, hidden)
  - sample noise (1, H, A)
  - get action chunk from the DistilledActor (or from the FM teacher during warm-up)
  - unnormalize + execute chunk
  - on next decision (or on episode end) write the transition to the replay buffer

Mirrors imitation.algos.rl.rollout_collector.FlowRLCollector for env setup.
"""

import numpy as np
import torch

import imitation.envs.libero.wrappers as lw


class DiceCollector:
    def __init__(self, env_runner, bc_policy, student_model, device,
                 max_episode_length=None, use_teacher_for_collect: bool = False):
        """
        env_runner       : LiberoRunner
        bc_policy        : frozen BC FlowMatchingPolicy (provides encoder + normalizer + teacher)
        student_model    : DistilledRLModel (provides actor for collection)
        device           : torch device
        use_teacher_for_collect : if True, act with the BC FM teacher (warm-up phase);
                                  otherwise act with the student actor.
        """
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
        """Build batch from raw obs, run BC preprocess + encoder -> cond (1, num_enc, hidden)."""
        batch = self.bc._make_batch({k: v for k, v in obs.items()}, task_index, **task_emb)
        batch = self.bc.preprocess_input(batch, train_mode=False)
        cond = self.bc.get_cond(batch)  # (1, num_enc, hidden)
        return cond, batch

    @torch.no_grad()
    def _act(self, cond: torch.Tensor) -> torch.Tensor:
        """Sample one normalized action chunk + the noise used. Returns (a_norm, noise)
        both shaped (1, chunk, A) on device. Uses student actor unless warm-up flag set."""
        noise = torch.randn(1, self.chunk_size, self.action_dim, device=self.device)
        if self.use_teacher_for_collect:
            # one Euler pass through the frozen FM
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
        else:
            a_norm = self.student.get_action(cond, noise)  # already clamped
        return a_norm, noise

    @torch.no_grad()
    def _unnormalize_and_execute_setup(self, a_norm: torch.Tensor, batch: dict) -> list:
        """Take a (1, chunk, A) normalized action, unnormalize+postprocess, return
        a list of env-ready np actions of length action_horizon."""
        a = self.bc.normalizer.unnormalize({"actions": a_norm.cpu().numpy()})["actions"]
        batch["actions"] = torch.tensor(a, device=self.device)
        a = self.bc.postprocess_actions(batch)["actions"].cpu().numpy()
        a = np.transpose(a, (1, 0, 2))  # (chunk, 1, A)
        return list(a[: self.action_horizon])

    @torch.no_grad()
    def rollout_episode(self, task_index: int, init_idx: int, replay):
        """Run one episode, push each chunk-decision transition into the replay buffer.

        Returns: (success: bool, n_decisions: int).
        """
        if self._env is None or self._env_task != task_index:
            self._build_env(task_index)
        task_emb = self._task_emb(task_index)

        all_init = self.benchmark.get_task_init_states(task_index)
        init_states = all_init[np.array([init_idx % all_init.shape[0]])]
        obs, _ = self._env.reset(init_states=init_states)

        # buffer of (cond, noise, action) pending writes -- written when we know s' or done.
        pending: list = []
        success = False
        first_success = np.inf

        queue: list = []
        step_in_chunk = 0
        for step in range(self.max_episode_length):
            if len(queue) == 0:
                cond, batch = self._encode_cond(obs, task_index, task_emb)
                a_norm, noise = self._act(cond)
                queue = self._unnormalize_and_execute_setup(a_norm, batch)
                step_in_chunk = 0
                # Stash pending (cond, noise, action) -- s' and reward filled at chunk boundary.
                pending.append({
                    "cond": cond.squeeze(0).cpu(),                       # (num_enc, hidden)
                    "noise": noise.squeeze(0).cpu(),                     # (chunk, A)
                    "action": a_norm.squeeze(0).cpu(),                   # (chunk, A)
                    "decision_step": step,
                })

            act = torch.tensor(queue.pop(0))
            act = self.bc.final_postprocess_actions(act).to(torch.float32).cpu().numpy()
            obs, reward, terminated, truncated, info = self._env.step(act)
            step_in_chunk += 1

            if info[0]["success"] and not success:
                first_success = step
            success = success or info[0]["success"]
            done = success

            # Close the current decision either at chunk boundary or on terminal.
            if (len(queue) == 0 or done) and pending:
                cur = pending.pop(0)
                if done:
                    next_cond = torch.zeros_like(cur["cond"])  # absorbed by (1 - done) anyway
                    r = 1.0 if success else 0.0
                    replay.add(cond=cur["cond"], noise=cur["noise"], action=cur["action"],
                               reward=r, next_cond=next_cond, done=True)
                else:
                    # Encode next obs as s'. We compute it once and reuse for the next pending.
                    next_cond, _ = self._encode_cond(obs, task_index, task_emb)
                    nc = next_cond.squeeze(0).cpu()
                    replay.add(cond=cur["cond"], noise=cur["noise"], action=cur["action"],
                               reward=0.0, next_cond=nc, done=False)

            if done:
                # Drop any stale pending records (they correspond to actions not executed).
                pending.clear()
                queue.clear()
                break

        return bool(success), step + 1
