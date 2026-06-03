"""On-policy rollout collector for GRPO fine-tuning of the noised flow-matching policy.

Reuses the LiberoRunner env construction. A GRPO *group* = ``group_size`` rollouts from
the SAME init state, so the group-normalized advantage is computed over rollouts that
differ only by the policy's sampling stochasticity.

Rollouts are run SEQUENTIALLY through a single in-process env (``env_num=1`` ->
DummyVectorEnv), matching the deterministic-eval path. This avoids the SubprocVectorEnv /
EGL-in-forked-subprocess fragility on a single GPU. Each rollout has batch dim 1.

Per chunk-decision (every ``action_horizon`` env steps) we store the denoising chain and
behavior means/log-probs needed to recompute the PPO ratio later. We break a rollout on
first success, so all recorded decisions are valid (the mask is kept for generality).
"""

import numpy as np
import torch

import imitation.envs.libero.wrappers as lw


class FlowRLCollector:
    def __init__(self, env_runner, policy, group_size, device, max_episode_length=None):
        self.runner = env_runner
        self.policy = policy
        self.G = group_size
        self.device = device
        self.benchmark = env_runner.benchmark
        self.env_factory = env_runner.env_factory
        self.env_names = env_runner.env_names
        self.frame_stack = env_runner.frame_stack
        self.max_episode_length = max_episode_length or env_runner.max_episode_length
        self.action_horizon = policy.action_horizon
        self._env = None
        self._env_task = None

    def _build_env(self, task_index):
        env_fn = lambda: lw.LiberoFrameStack(
            self.env_factory(task_id=task_index, benchmark=self.benchmark), self.frame_stack
        )
        self._env = lw.LiberoVectorWrapper(env_fn, 1)  # in-process DummyVectorEnv
        self._env_task = task_index

    def _task_emb(self, task_index):
        te = self.benchmark.get_task_emb(task_index)
        return {k: v.repeat(1, 1) for k, v in te.items()}  # batch dim 1

    @torch.no_grad()
    def _rollout(self, task_index, init_idx, task_emb):
        """One episode through the single env, reset to a specific init state."""
        policy = self.policy
        all_init = self.benchmark.get_task_init_states(task_index)
        init_states = all_init[np.array([init_idx % all_init.shape[0]])]  # (1, D)
        obs, _ = self._env.reset(init_states=init_states)

        success, first_success = False, np.inf
        queue, dec_records = [], []
        for step in range(self.max_episode_length):
            if len(queue) == 0:
                batch = policy._make_batch({k: v for k, v in obs.items()}, task_index, **task_emb)
                rec = policy.sample_actions_stochastic(batch)
                act = policy.normalizer.unnormalize({"actions": rec["action"]})["actions"]
                batch["actions"] = torch.tensor(act, device=self.device)
                act = policy.postprocess_actions(batch)["actions"].cpu().numpy()
                act = np.transpose(act, (1, 0, 2))  # (chunk, 1, A)
                queue = list(act[: self.action_horizon])
                rec["decision_step"] = step
                dec_records.append(rec)

            action = torch.tensor(queue.pop(0))
            action = policy.final_postprocess_actions(action).to(torch.float32).cpu().numpy()
            obs, reward, terminated, truncated, info = self._env.step(action)
            if info[0]["success"] and not success:
                first_success = step
            success = success or info[0]["success"]
            if success:
                break
        return dec_records, bool(success), first_success

    @torch.no_grad()
    def collect(self, task_index, init_indices):
        """Run len(init_indices) groups x G rollouts. Returns a flat buffer + stats."""
        if self._env is None or self._env_task != task_index:
            self._build_env(task_index)
        task_emb = self._task_emb(task_index)

        records, group_ids, rewards, group_succ = [], [], [], []
        t_grid = None
        for gi, init_idx in enumerate(init_indices):
            succ = []
            for _ in range(self.G):
                dec_records, success, first_success = self._rollout(task_index, init_idx, task_emb)
                succ.append(float(success))
                R = torch.tensor([float(success)], dtype=torch.float32)
                for rec in dec_records:
                    valid = 1.0 if rec["decision_step"] <= first_success else 0.0
                    records.append({
                        "cond": rec["cond"], "chain": rec["chain"], "mu_old": rec["mu_old"],
                        "valid": torch.tensor([valid], dtype=torch.float32),
                    })
                    group_ids.append(gi)
                    rewards.append(R)
                    t_grid = rec["t_grid"]
            group_succ.append(float(np.mean(succ)))

        buf = self._flatten(records, group_ids, rewards)
        buf["t_grid"] = t_grid
        stats = {
            "mean_success": float(np.mean(group_succ)),
            "n_groups": len(init_indices),
            "n_rows": int(buf["chain"].shape[0]) if len(records) else 0,
            "frac_valid": float(buf["valid"].mean()) if len(records) else 0.0,
        }
        return buf, stats

    @staticmethod
    def _flatten(records, group_ids, rewards):
        cond = torch.cat([r["cond"] for r in records], dim=0)
        chain = torch.cat([r["chain"] for r in records], dim=0)
        mu_old = torch.cat([r["mu_old"] for r in records], dim=0)
        valid = torch.cat([r["valid"] for r in records], dim=0)
        rewards = torch.cat(rewards, dim=0)
        group_ids = torch.tensor(group_ids, dtype=torch.long)
        return {"cond": cond, "chain": chain, "mu_old": mu_old,
                "valid": valid, "rewards": rewards, "group_ids": group_ids}
