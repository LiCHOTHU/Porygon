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


def _make_student_sample_actions(bc_policy, student_model, best_of_k=1,
                                 det_base=False, base_seed=0,
                                 sel_strategy="max_q_min"):
    """Mirror of dice_train.py:_make_student_sample_actions — monkey-patch onto
    the BC policy so env_runner drives the DICE student through the BC encoder.

    best_of_k: number of noise candidates the DICE critic scores by max_q_min.
    K=1 reproduces the single-sample deploy (no selection).

    det_base: if True, the K=1 deploy uses a FIXED seeded noise (shared across
    states), so the FM teacher base AND the residual are deterministic — matching
    act_sim's FrozenImitationBase._fixed_noise (same base_seed). This removes the
    base-sampling variance that otherwise penalizes DICE at K=1."""
    K = int(best_of_k)
    _fixed = {}  # cache: (device,dtype) -> (1, chunk, A) fixed noise
    def _fixed_noise(B, device, dtype):
        key = (str(device), str(dtype))
        if key not in _fixed:
            g = torch.Generator(device="cpu").manual_seed(int(base_seed))
            n = torch.randn(1, bc_policy.chunk_size, bc_policy.network_action_dim,
                            generator=g)
            _fixed[key] = n.to(device=device, dtype=dtype)
        return _fixed[key].expand(B, -1, -1).clone()
    @torch.no_grad()
    def sample_actions(data):
        was_training = bc_policy.training
        bc_policy.eval()
        data = bc_policy.preprocess_input(data, train_mode=False)
        cond = bc_policy.get_cond(data)
        B = cond.shape[0]
        if K <= 1:
            if det_base:
                noise = _fixed_noise(B, cond.device, cond.dtype)
            else:
                noise = torch.randn(B, bc_policy.chunk_size, bc_policy.network_action_dim,
                                    device=cond.device, dtype=cond.dtype)
            a = student_model.get_action(cond, noise)
        else:
            a, _ = student_model.get_exploration_action(
                cond, num_samples=K, exploration_strategy=sel_strategy,
                training_step=10**9)
        a = torch.clamp(a, -1, 1)
        if was_training:
            bc_policy.train()
        return a.to(torch.float32).cpu().numpy()
    return sample_actions


def _build_dice_policy(dice_ck, env_runner, task_indices, device, best_of_k=1,
                       det_base=False, sel_strategy="max_q_min", base_noise_seed=0,
                       return_student=False):
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
        bc_policy, student, best_of_k, det_base=det_base, base_seed=base_noise_seed,
        sel_strategy=sel_strategy)
    if return_student:
        return bc_policy, student
    return bc_policy


def _build_actsim_policy(actsim_ck, cold_start_path, env_runner, task_indices, device, best_of_k=1,
                         rand_base=False):
    """For an act_sim residual checkpoint (keys actor/critic/target_critic/meta,
    NO baked-in config): build the frozen FM base from the given cold_start ckpt,
    build the ResidualRLModel, load actor/critic, and monkey-patch sample_actions
    so env_runner drives base+residual with best-of-K critic selection (max_q_min).
    Mirrors collector._select_action(mode='policy') exactly."""
    import sys, os
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(ROOT, "act_sim_ppo_v6_gtsim"))
    from rl_algorithm.residual.residual_rl import ResidualRLModel
    from integrations.libero_imitation.frozen_base import FrozenImitationBase
    import imitation.envs.libero.wrappers as lw

    cs = utils.load_checkpoint(cold_start_path)
    cfg = OmegaConf.create(cs["config"])
    shape_meta = cfg.task.shape_meta
    policy_cfg = OmegaConf.create(cfg.algo.policy)
    if "temporal_agg" in policy_cfg:
        policy_cfg["temporal_agg"] = False
    bc_policy = instantiate(policy_cfg, shape_meta=shape_meta).to(device)
    utils.soft_load_state_dict(bc_policy, cs["model"])
    bc_policy.normalizer.fit(cs["norm_stats"])
    bc_policy.eval()
    for p in bc_policy.parameters():
        p.requires_grad = False

    # probe cond dim -> state_feat_dim S = num_enc * hidden
    task_id = int(task_indices[0])
    task_emb = {k: v.repeat(1, 1) for k, v in env_runner.benchmark.get_task_emb(task_id).items()}
    probe_env = env_runner.env_factory(task_id=task_id, benchmark=env_runner.benchmark)
    probe_env = lw.LiberoVectorWrapper(lambda: lw.LiberoFrameStack(probe_env, env_runner.frame_stack), 1)
    probe_obs, _ = probe_env.reset()
    probe_batch = bc_policy._make_batch({k: v for k, v in probe_obs.items()}, task_id, **task_emb)
    probe_batch = bc_policy.preprocess_input(probe_batch, train_mode=False)
    with torch.no_grad():
        probe_cond = bc_policy.get_cond(probe_batch)
    S = int(probe_cond.shape[1]) * int(probe_cond.shape[2])
    probe_env.close()

    base = FrozenImitationBase(bc_policy, device, state_feat_dim=S, base_seed=0)
    model = ResidualRLModel(
        frozen_act=base, action_dim=bc_policy.network_action_dim,
        n_exec=bc_policy.chunk_size, residual_scale=0.1, critic_ensemble_size=2,
        bc_loss_weight=10.0, use_soft_q_filtering=False, q_filtering_warmup_steps=2000,
        num_multi_z_for_actor_loss=4, device=device,
    ).to(device)
    model.actor.load_state_dict(actsim_ck["actor"])
    model.critic.load_state_dict(actsim_ck["critic"])
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    K = int(best_of_k)

    @torch.no_grad()
    def _rand_base_chunk(cond):
        """CONTROL: faithful mirror of base.base_chunk_from_cond but with FRESH
        noise resampled each decision (no _fixed_noise). Tests whether act_sim's
        edge is the fixed base — if it degrades like old DICE did, the base is the
        lever; if it stays high, the base is NOT the main difference."""
        B = cond.shape[0]
        x = torch.randn(B, base.chunk_size, base.action_dim,
                        device=cond.device, dtype=cond.dtype)
        enc = bc_policy.velocity_net.forward_enc(cond)
        Ksteps = base.num_inference_steps
        dt = 1.0 / Ksteps
        t0 = base.drift_t if base.is_drift else 0.0
        t = torch.full((B,), t0, device=cond.device, dtype=cond.dtype)
        for _ in range(Ksteps):
            v = bc_policy.velocity_net.forward_dec(x, t, enc)
            x = x + dt * v
            t = t + dt
        return torch.clamp(x, -1, 1)

    @torch.no_grad()
    def sample_actions(data):
        was_training = bc_policy.training
        bc_policy.eval()
        data = bc_policy.preprocess_input(data, train_mode=False)
        cond = bc_policy.get_cond(data)
        sf = base.state_feat_from_cond(cond)
        ba = _rand_base_chunk(cond) if rand_base else base.base_chunk_from_cond(cond)  # base chunk (det vs resampled)
        action, _ = model.get_exploration_action(
            sf, ba, num_samples=max(1, K), strategy="max_q_min",
            training_step=10**9, warmup_steps=0)
        action = torch.clamp(action, -1, 1)
        if was_training:
            bc_policy.train()
        return action.to(torch.float32).cpu().numpy()

    bc_policy.sample_actions = sample_actions
    return bc_policy


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
                    best_of_k=1, actsim_cold_start=None, dice_det_base=False,
                    dice_sel_strategy="max_q_min", actsim_rand_base=False,
                    dice_base_noise_seed=0, init_offset=0):
    """Load a checkpoint, build its policy from baked-in config, run env eval
    at each seed. Returns list of per-seed overall_success_rate.

    Handles BC/RL ckpts (state_dict['model']), DICE ckpts
    (state_dict['student_actor']), and act_sim residual ckpts
    (state_dict['actor']+'critic', NO baked config -> needs actsim_cold_start).

    best_of_k: critic best-of-K selection for DICE / act_sim (matched protocol).
    Init states are deterministic per (task, rollouts_per_env) — only the
    policy's action sampling is stochastic across seeds.
    """
    state_dict = utils.load_checkpoint(ckpt_path)
    is_dice   = "student_actor" in state_dict
    is_actsim = (not is_dice) and ("actor" in state_dict) and ("critic" in state_dict) \
                and ("config" not in state_dict)

    if is_actsim:
        # act_sim ckpt carries no config; build env_runner from the FM cold_start.
        cs = utils.load_checkpoint(actsim_cold_start)
        task_cfg = cs["config"]["task"]
    else:
        cfg_dict = state_dict["config"]
        task_cfg = cfg_dict["task"]

    env_runner_cfg = OmegaConf.create(task_cfg)["env_runner"]
    env_runner_cfg["rollouts_per_env"] = int(rollouts_per_env)
    env_runner_cfg["init_offset"] = int(init_offset)
    env_runner = instantiate(env_runner_cfg)

    if is_actsim:
        print(f"    [act_sim residual] best_of_k={best_of_k} cold_start={actsim_cold_start}")
        print(f"    [act_sim residual] rand_base={actsim_rand_base}")
        policy = _build_actsim_policy(state_dict, actsim_cold_start, env_runner,
                                      task_indices, device, best_of_k,
                                      rand_base=actsim_rand_base)
    elif is_dice:
        print(f"    [DICE format] best_of_k={best_of_k} det_base={dice_det_base} "
              f"sel_strategy={dice_sel_strategy} "
              f"cold_start={cfg_dict['cold_start_checkpoint']}")
        policy = _build_dice_policy(state_dict, env_runner, task_indices, device,
                                    best_of_k, det_base=dice_det_base,
                                    sel_strategy=dice_sel_strategy,
                                    base_noise_seed=dice_base_noise_seed)
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
    best_of_k = int(cfg.get("best_of_k", 1))
    actsim_cold_start = cfg.get("actsim_cold_start", None)
    if actsim_cold_start is not None:
        actsim_cold_start = str(actsim_cold_start)
    dice_det_base = bool(cfg.get("dice_det_base", False))
    dice_sel_strategy = str(cfg.get("dice_sel_strategy", "max_q_min"))
    actsim_rand_base = bool(cfg.get("actsim_rand_base", False))
    dice_base_noise_seed = int(cfg.get("dice_base_noise_seed", 0))
    init_offset = int(cfg.get("init_offset", 0))
    seeds = [base_seed + i for i in range(n_seeds)]

    print(f"=== Powered eval (matched best-of-K) ===")
    print(f"  tasks:            {task_indices}")
    print(f"  best_of_k:        {best_of_k}")
    print(f"  dice_det_base:    {dice_det_base}")
    print(f"  dice_sel_strategy:{dice_sel_strategy}")
    print(f"  actsim_rand_base: {actsim_rand_base}")
    print(f"  dice_base_noise_seed: {dice_base_noise_seed}")
    print(f"  init_offset:      {init_offset}")
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
            rates, per_env_rates = eval_checkpoint(ckpt, task_indices, rollouts_per_env, seeds, device,
                                                   best_of_k=best_of_k, actsim_cold_start=actsim_cold_start,
                                                   dice_det_base=dice_det_base,
                                                   dice_sel_strategy=dice_sel_strategy,
                                                   actsim_rand_base=actsim_rand_base,
                                                   dice_base_noise_seed=dice_base_noise_seed,
                                                   init_offset=init_offset)
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
