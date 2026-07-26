"""Powered eval: 200 rollouts × 3 seeds, fixed init-state set across conditions.

Built to replace the 20-rollout × 3-seed re-eval whose ±13–18pp σ couldn't
distinguish real lift from init/policy noise.

Why this is "powered":
  - 200 rollouts halves the binomial σ from ~11pp (n=20) to ~3.5pp (n=200).
  - LIBERO runner indexes init states deterministically: indices = np.arange(
    count*env_num, (count+1)*env_num) % N. So at rollouts_per_env=200 every
    checkpoint sees the SAME 200 init states. The only residual variance is
    policy stochasticity, which is what the 3 seeds average over.
  - Each checkpoint loaded from its own baked-in config (BC, DICE, RL all
    save 'config' inside the .pth — we instantiate from that).

Usage via Hydra config_path=None:
  python scripts/powered_eval.py \
    ++targets='[["label1","/path/to/ckpt1.pth"], ["label2","/path/to/ckpt2.pth"]]' \
    ++task_indices=[32] ++n_seeds=3 ++base_seed=1000 ++rollouts_per_env=200 \
    ++out_json=/path/to/out.json

Output JSON:
  {
    "<label>": {ckpt, seeds, success_rates, mean, std_unbiased, min, max, range},
    "_gaps": {"<a>_vs_<b>": {delta, se, z}, ...},
    "_config": {task_indices, n_seeds, rollouts_per_env, ...},
  }
"""

import json
import os

import hydra
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

import imitation.utils.utils as utils


OmegaConf.register_new_resolver("eval", eval, replace=True)


def _make_student_sample_actions(bc_policy, student_model,
                                 eval_strategy="single", eval_num_samples=10):
    """Mirror of dice_train.py:_make_student_sample_actions — monkey-patch onto
    the BC policy so env_runner drives the DICE student through the BC encoder.

    eval_strategy "single" = one noise draw (legacy; residual~0 => ~BC).
    "max_q_min" etc. = the official/faithful protocol: draw eval_num_samples
    latents and execute the argmax under the Q-ensemble criterion — the same
    selection the training-loop eval and the official DICE-RL eval use."""
    @torch.no_grad()
    def sample_actions(data):
        was_training = bc_policy.training
        bc_policy.eval()
        data = bc_policy.preprocess_input(data, train_mode=False)
        cond = bc_policy.get_cond(data)
        B = cond.shape[0]
        if eval_strategy == "single":
            noise = torch.randn(B, bc_policy.chunk_size, bc_policy.network_action_dim,
                                device=cond.device, dtype=cond.dtype)
            a = student_model.get_action(cond, noise)
        else:
            # huge training_step so get_exploration_action skips its warmup
            # single-sample shortcut and actually runs the max_q selection
            a, _ = student_model.get_exploration_action(
                cond, num_samples=eval_num_samples,
                exploration_strategy=eval_strategy, training_step=10**9)
        a = torch.clamp(a, -1, 1)
        if was_training:
            bc_policy.train()
        return a.to(torch.float32).cpu().numpy()
    return sample_actions


def _build_dice_policy(dice_ck, env_runner, task_indices, device,
                       eval_strategy="single", eval_num_samples=10):
    """For a DICE checkpoint: build the frozen BC FlowMatchingPolicy from the
    baked-in cold_start, build the DistilledRLModel student with the baked-in
    DICE config, load student weights, monkey-patch sample_actions so the
    env_runner path drives the student. Returns the monkey-patched bc_policy."""
    from imitation.algos.dice.distill_rl import DistilledRLModel
    from imitation.algos.dice.teacher import FMTeacher
    import imitation.envs.libero.wrappers as lw

    cfg = OmegaConf.create(dice_ck["config"])
    shape_meta = cfg.task.shape_meta

    bc_policy = instantiate(cfg.algo.policy, shape_meta=shape_meta).to(device)
    bc_state = utils.load_checkpoint(cfg.cold_start_checkpoint)
    utils.soft_load_state_dict(bc_policy, bc_state["model"])
    bc_policy.normalizer.fit(dice_ck["norm_stats"])
    bc_policy.eval()
    for p in bc_policy.parameters():
        p.requires_grad = False

    task_id = int(task_indices[0])
    task_emb = {k: v.repeat(1, 1) for k, v in env_runner.benchmark.get_task_emb(task_id).items()}
    probe_env = env_runner.env_factory(task_id=task_id, benchmark=env_runner.benchmark)
    probe_env = lw.LiberoVectorWrapper(lambda: lw.LiberoFrameStack(probe_env, env_runner.frame_stack), 1)
    probe_obs, _ = probe_env.reset()
    probe_batch = bc_policy._make_batch({k: v for k, v in probe_obs.items()}, task_id, **task_emb)
    probe_batch = bc_policy.preprocess_input(probe_batch, train_mode=False)
    with torch.no_grad():
        probe_cond = bc_policy.get_cond(probe_batch)
    num_enc, hidden = int(probe_cond.shape[1]), int(probe_cond.shape[2])
    probe_env.close()

    d = cfg.dice
    student = DistilledRLModel(
        state_dim=num_enc * hidden,
        action_dim=bc_policy.network_action_dim,
        horizon_steps=bc_policy.chunk_size,
        actor_hidden=list(d.actor_hidden),
        critic_hidden=list(d.critic_hidden),
        ensemble_size=d.ensemble_size,
        q_depends_on_noise=d.q_depends_on_noise,
        conservative=d.conservative,
        lcb_kappa=d.lcb_kappa,
        td_loss=d.td_loss,
        bc_loss_weight=d.bc_loss_weight,
        critic_weight=d.get("critic_weight", 1.0),
        num_multi_z=d.num_multi_z,
        use_soft_q_filtering=d.get("use_soft_q_filtering", False),
        q_filtering_warmup_steps=d.get("q_filtering_warmup_steps", 25000),
        q_underestimation_threshold=d.get("q_underestimation_threshold", -0.1),
        replay_flow_warmup_steps=d.get("replay_flow_warmup_steps", 1000),
        use_q_normalization=d.get("use_q_normalization", False),
        multi_sample_next_noise=d.get("multi_sample_next_noise", False),
        num_next_noise_samples=d.get("num_next_noise_samples", 4),
        use_n_step=d.get("use_n_step", False),
        n_step=d.get("n_step", 1),
        disable_q_loss_for_expert_data=d.get("disable_q_loss_for_expert_data", False),
        disable_td_loss_for_expert_data=d.get("disable_td_loss_for_expert_data", False),
        always_retain_bc_loss_for_expert_data=d.get("always_retain_bc_loss_for_expert_data", False),
        clip_action=d.get("clip_action", True),
        zero_final_layer=d.get("zero_final_layer", False),
        use_noise_head=d.get("use_noise_head", False),
        noise_sigma_min=d.get("noise_sigma_min", 0.01),
        noise_sigma_max=d.get("noise_sigma_max", 0.1),
        noise_head_hidden=d.get("noise_head_hidden", 256),
        noise_head_initial_logit=d.get("noise_head_initial_logit", 0.0),
        cql_weight=d.get("cql_weight", 0.0),
        actor_layernorm=d.get("actor_layernorm", True),
        critic_layernorm=d.get("critic_layernorm", True),
        device=device,
    ).to(device)

    teacher = FMTeacher(bc_policy)
    student.attach_teacher(teacher)

    student.actor.load_state_dict(dice_ck["student_actor"])
    student.critic.load_state_dict(dice_ck["student_critic"])
    if "student_target_critic" in dice_ck:
        student.target_critic.load_state_dict(dice_ck["student_target_critic"])
    if student.use_noise_head and "student_noise_head" in dice_ck:
        student.noise_head.load_state_dict(dice_ck["student_noise_head"])
    student.eval()
    for p in student.parameters():
        p.requires_grad = False

    bc_policy.sample_actions = _make_student_sample_actions(
        bc_policy, student, eval_strategy=eval_strategy,
        eval_num_samples=eval_num_samples)
    return bc_policy, student


class _ResHead(torch.nn.Module):
    """Mirror of grd_train.ResHead: residual added to the one-step drift action."""
    def __init__(self, state_dim, chunk, adim, hidden=1024):
        super().__init__()
        self.chunk, self.adim = chunk, adim
        self.net = torch.nn.Sequential(
            torch.nn.Linear(state_dim + chunk * adim, hidden), torch.nn.Mish(),
            torch.nn.Linear(hidden, hidden), torch.nn.Mish(),
            torch.nn.Linear(hidden, chunk * adim))

    def forward(self, x, cond):
        b = x.shape[0]
        d = self.net(torch.cat([x.reshape(b, -1), cond.reshape(b, -1)], -1))
        return d.view(b, self.chunk, self.adim)


def _attach_grd_residual(policy, res_head_state, device):
    """Rebuild the GRD residual head from saved weights and fold it into the
    one-step drift sampler exactly as grd_train.sample_actions_res did:
        x_{k+1} = x_k + dt * v(x_k, t_k) + head(x_k, cond)
    Dims are inferred from the saved head weights so no probe env is needed."""
    chunk, adim = policy.chunk_size, policy.network_action_dim
    w0 = res_head_state["net.0.weight"]          # (hidden, state_dim + chunk*adim)
    hidden = int(w0.shape[0])
    state_dim = int(w0.shape[1]) - chunk * adim
    head = _ResHead(state_dim, chunk, adim, hidden=hidden).to(device)
    head.load_state_dict(res_head_state)
    head.eval()
    for p in head.parameters():
        p.requires_grad = False

    @torch.no_grad()
    def sample_actions_res(self, data):
        was_training = self.training
        self.eval()
        data = self.preprocess_input(data, train_mode=False)
        cond = self.get_cond(data)
        B, dev = cond.shape[0], cond.device
        x = torch.randn(B, self.chunk_size, self.network_action_dim, device=dev, dtype=cond.dtype)
        enc = self.velocity_net.forward_enc(cond)
        dt = 1.0 / self.num_inference_steps
        t = torch.zeros(B, device=dev, dtype=cond.dtype)
        for _ in range(self.num_inference_steps):
            x = x + dt * self.velocity_net.forward_dec(x, t, enc) + head(x, cond)
            t = t + dt
        if was_training:
            self.train()
        return torch.clamp(x, -1, 1).cpu().numpy()

    import types
    policy.sample_actions = types.MethodType(sample_actions_res, policy)
    return policy


def eval_checkpoint(ckpt_path, task_indices, rollouts_per_env, seeds, device,
                    dice_eval_strategy="single", dice_eval_samples=10):
    """Load a checkpoint, build its policy from baked-in config, run env eval
    at each seed. Returns list of per-seed overall_success_rate.

    Handles both BC/RL ckpts (state_dict['model']) and DICE ckpts
    (state_dict['student_actor'] + student_critic + cold_start reference).

    Init states are deterministic per (task, rollouts_per_env) — only the
    policy's action sampling is stochastic across seeds.
    """
    state_dict = utils.load_checkpoint(ckpt_path)
    cfg_dict = state_dict["config"]
    task_cfg = cfg_dict["task"]
    is_dice = "student_actor" in state_dict

    env_runner_cfg = OmegaConf.create(task_cfg)["env_runner"]
    env_runner_cfg["rollouts_per_env"] = int(rollouts_per_env)
    env_runner = instantiate(env_runner_cfg)

    if is_dice:
        print(f"    [DICE format] cold_start={cfg_dict['cold_start_checkpoint']}")
        print(f"    [DICE eval] strategy={dice_eval_strategy} samples={dice_eval_samples}")
        policy, _ = _build_dice_policy(state_dict, env_runner, task_indices, device,
                                       eval_strategy=dice_eval_strategy,
                                       eval_num_samples=dice_eval_samples)
    else:
        policy_cfg = OmegaConf.create(cfg_dict["algo"]["policy"])
        if "temporal_agg" in policy_cfg:
            policy_cfg["temporal_agg"] = False
        shape_meta = task_cfg["shape_meta"]
        policy = instantiate(policy_cfg, shape_meta=shape_meta).to(device)
        utils.soft_load_state_dict(policy, state_dict["model"])
        policy.normalizer.fit(state_dict["norm_stats"])
        policy.eval()
        for p in policy.parameters():
            p.requires_grad = False
        if "res_head" in state_dict:
            print("    [GRD residual] folding res_head into one-step drift sampler")
            _attach_grd_residual(policy, state_dict["res_head"], device)

    env_names = [env_runner.env_names[int(t)] for t in task_indices]
    print(f"    env_names: {env_names}  rollouts_per_env: {rollouts_per_env}")

    rates = []
    per_env_rates = {en: [] for en in env_names}
    for seed in seeds:
        torch.manual_seed(int(seed))
        np.random.seed(int(seed))
        out = env_runner.run(
            policy,
            env_names=env_names,
            do_tqdm=False,
        )
        rate = float(out["rollout"]["overall_success_rate"])
        rates.append(rate)
        for en in env_names:
            per_env_rates[en].append(float(out["rollout_success_rate"][en]))
        print(f"    seed {seed}: overall={rate:.3f}  per_env=" +
              " ".join(f"{en[:18]}:{out['rollout_success_rate'][en]:.3f}" for en in env_names))
    return rates, per_env_rates


def _pairwise_gap(results, a, b):
    ma = results[a]["mean"]; mb = results[b]["mean"]
    sa = results[a]["std_unbiased"]; sb = results[b]["std_unbiased"]
    na = len(results[a]["success_rates"]); nb = len(results[b]["success_rates"])
    se = (sa**2 / na + sb**2 / nb) ** 0.5
    delta = ma - mb
    return {"delta": delta, "se": se, "z": (delta / se) if se > 0 else None}


@hydra.main(config_path=None, version_base=None)
def main(cfg):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # targets: list of [label, ckpt_path] pairs (Hydra will pass as a ListConfig)
    targets_raw = list(cfg.targets)
    targets = [(str(t[0]), str(t[1])) for t in targets_raw]
    task_indices = list(cfg.task_indices)
    n_seeds = int(cfg.n_seeds)
    base_seed = int(cfg.base_seed)
    rollouts_per_env = int(cfg.rollouts_per_env)
    out_json = str(cfg.out_json)
    seeds = [base_seed + i for i in range(n_seeds)]

    print(f"=== Powered eval ===")
    print(f"  tasks:            {task_indices}")
    print(f"  rollouts_per_env: {rollouts_per_env}")
    print(f"  seeds:            {seeds}")
    print(f"  targets:          {len(targets)}")
    for label, ckpt in targets:
        print(f"    - {label}: {ckpt}")

    results = {}
    for label, ckpt in targets:
        print(f"\n=== {label} ===")
        if not os.path.exists(ckpt):
            print(f"  MISSING — skipping")
            results[label] = {"ckpt": ckpt, "error": "missing", "success_rates": [],
                              "mean": float("nan"), "std_unbiased": float("nan")}
            continue
        try:
            rates, per_env_rates = eval_checkpoint(
                ckpt, task_indices, rollouts_per_env, seeds, device,
                dice_eval_strategy=str(cfg.get("dice_eval_strategy", "single")),
                dice_eval_samples=int(cfg.get("dice_eval_samples", 10)))
            arr = np.array(rates, dtype=float)
            per_env_summary = {}
            for en, rs in per_env_rates.items():
                a = np.array(rs, dtype=float)
                per_env_summary[en] = {
                    "seeds": [float(x) for x in a],
                    "mean": float(a.mean()),
                    "std_unbiased": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
                }
            results[label] = {
                "ckpt": ckpt,
                "seeds": seeds,
                "success_rates": [float(x) for x in arr],
                "mean": float(arr.mean()),
                "std_unbiased": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
                "min": float(arr.min()),
                "max": float(arr.max()),
                "range": float(arr.max() - arr.min()),
                "per_env": per_env_summary,
            }
            print(f"  -> mean={arr.mean():.3f}  std={results[label]['std_unbiased']:.3f}  range={results[label]['range']:.3f}")
        except Exception as e:
            print(f"  ERROR: {e}")
            results[label] = {"ckpt": ckpt, "error": str(e), "success_rates": [],
                              "mean": float("nan"), "std_unbiased": float("nan")}

        # incremental save after each ckpt
        os.makedirs(os.path.dirname(out_json), exist_ok=True)
        with open(out_json, "w") as f:
            json.dump(results, f, indent=2)

    # pairwise gaps: every non-baseline vs the first target (treated as baseline)
    valid = [lbl for lbl, _ in targets if "error" not in results.get(lbl, {})]
    if len(valid) >= 2:
        baseline = valid[0]
        gaps = {}
        for lbl in valid[1:]:
            gaps[f"{lbl}_vs_{baseline}"] = _pairwise_gap(results, lbl, baseline)
        results["_gaps"] = gaps
        print(f"\n=== Pairwise gaps vs {baseline} ===")
        for k, v in gaps.items():
            z = v["z"]
            sig = "***" if z is not None and abs(z) > 2.0 else (" *" if z is not None and abs(z) > 1.0 else "  ")
            print(f"  {sig} {k}: delta={v['delta']:+.3f}  se={v['se']:.3f}  z={v['z']:+.2f}" if z is not None else
                  f"     {k}: delta={v['delta']:+.3f}  se={v['se']:.3f}  z=None")

    results["_config"] = {
        "task_indices": list(task_indices),
        "n_seeds": n_seeds,
        "base_seed": base_seed,
        "rollouts_per_env": rollouts_per_env,
    }

    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out_json}")


if __name__ == "__main__":
    main()
