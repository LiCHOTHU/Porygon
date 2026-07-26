"""GOODCOV / BADENT finetunability metrics (DICE-RL paper Sec 5.2) for our bases.

Answers: WHY does the one-step drift base gain more from DICE than the FM base?
The paper's diagnostic: a base is finetunable when its z-conditioned samples
cover good modes (GOODCOV high) and its bad-mass actions are concentrated
rather than diffuse (BADENT low).

Protocol (ours, stated in the output JSON):
  - states s = BC-encoder conds visited by the BASE policy itself (fresh
    rollouts through DiceCollector's use_teacher_for_collect path, which builds
    reward/done/mc_return exactly as DICE training did: sparse success reward,
    gamma per chunk decision).
  - G_hat(s)  = mc_return from those rollouts (0 for failed episodes).
  - Q         = the finished DICE run's critic (iter-100 ckpt), min over the
                ensemble, evaluated at a = pi_pre(s, z) for num_z fresh z.
  - GOODCOV_a = E_{s,z}[ 1( Q(s, pi_pre(s,z)) >= a * G_hat(s) ) ]
  - BADENT_a  = mean over action coords of histogram entropy (nbins on [-1,1])
                of the actions failing the threshold.
  Caveat: the critic differs per arm (trained on its own base's data) — the
  metric is a property of (base, its critic), same as in the paper.

Usage:
  python scripts/finetunability_metrics.py \
    ++targets='[["ff_drift_s10000","/path/dice_iter_0100.pth"]]' \
    ++task_indices=[8,21,32,53,65,73,75,81] \
    ++episodes_per_task=25 ++num_z=16 ++alphas=[0.5,0.75,0.9] \
    ++out_json=/path/out.json [++dice.actor_layernorm=false]
"""

import json
import os
import sys

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from powered_eval import _build_dice_policy  # noqa: E402

import imitation.utils.utils as utils  # noqa: E402

OmegaConf.register_new_resolver("eval", eval, replace=True)


@torch.no_grad()
def compute_metrics(student, replay, num_z, alphas, nbins, device, batch_states=128):
    """Stream over replay rows; accumulate coverage counts and per-coord
    histograms of bad-mass actions for each alpha."""
    n = len(replay)
    chunk, adim = replay.action.shape[1], replay.action.shape[2]
    ncoord = chunk * adim

    good = {a: 0 for a in alphas}
    total = {a: 0 for a in alphas}
    good_s = {a: 0 for a in alphas}     # restricted to G_hat > 0 states
    total_s = {a: 0 for a in alphas}
    bad_hist = {a: torch.zeros(ncoord, nbins, dtype=torch.float64) for a in alphas}
    disp_sum, qstd_sum, qmean_sum, nstates = 0.0, 0.0, 0.0, 0

    for lo in range(0, n, batch_states):
        hi = min(lo + batch_states, n)
        B = hi - lo
        state = replay.cond[lo:hi].to(device)                    # (B, ne, h)
        G = replay.mc_return[lo:hi, 0].to(device)                # (B,)

        state_rep = state.unsqueeze(0).expand(num_z, *state.shape).reshape(
            num_z * B, *state.shape[1:])
        noise = torch.randn(num_z * B, chunk, adim, device=device)
        actions = student._teacher_fn(state_rep, noise)          # (K*B, chunk, A) base-only
        q_all = student.critic(state_rep, noise, actions, return_all=True)
        q_min = torch.stack(q_all, 0).min(0)[0].view(num_z, B)   # (K, B)

        a_flat = actions.view(num_z, B, ncoord)
        disp_sum += float(a_flat.std(dim=0).mean() * B)
        qstd_sum += float(q_min.std(dim=0).mean() * B)
        qmean_sum += float(q_min.mean() * B)
        nstates += B
        succ_mask = (G > 0)                                      # (B,)

        for a in alphas:
            ok = q_min >= (a * G).unsqueeze(0)                   # (K, B)
            good[a] += int(ok.sum()); total[a] += num_z * B
            good_s[a] += int(ok[:, succ_mask].sum())
            total_s[a] += num_z * int(succ_mask.sum())
            bad = a_flat.reshape(num_z * B, ncoord)[~ok.reshape(-1)]
            if bad.numel():
                idx = ((bad + 1.0) / 2.0 * nbins).clamp(0, nbins - 1).long()  # (nbad, ncoord)
                oh = torch.zeros(ncoord, nbins, device=device, dtype=torch.float64)
                oh.scatter_add_(1, idx.t(), torch.ones_like(idx.t(), dtype=torch.float64))
                bad_hist[a] += oh.cpu()

    out = {"n_states": nstates,
           "action_dispersion_over_z": disp_sum / max(nstates, 1),
           "qmin_std_over_z": qstd_sum / max(nstates, 1),
           "qmin_mean": qmean_sum / max(nstates, 1),
           "num_z": num_z, "nbins": nbins, "per_alpha": {}}
    for a in alphas:
        h = bad_hist[a]
        tot = h.sum(dim=1, keepdim=True).clamp(min=1.0)
        p = h / tot
        ent = -(p * p.clamp(min=1e-12).log()).sum(dim=1)         # (ncoord,) nats
        out["per_alpha"][str(a)] = {
            "goodcov": good[a] / max(total[a], 1),
            "goodcov_success_states": good_s[a] / max(total_s[a], 1),
            "n_bad": int(h.sum().item() / max(ncoord, 1)),
            "badent_nats": float(ent.mean()),
        }
    return out


@hydra.main(config_path=None, version_base=None)
def main(cfg):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    targets = [(str(t[0]), str(t[1])) for t in cfg.targets]
    task_indices = [int(t) for t in cfg.task_indices]
    episodes_per_task = int(cfg.get("episodes_per_task", 25))
    num_z = int(cfg.get("num_z", 16))
    alphas = [float(a) for a in cfg.get("alphas", [0.5, 0.75, 0.9])]
    nbins = int(cfg.get("nbins", 50))
    out_json = str(cfg.out_json)

    from imitation.algos.dice.collector import DiceCollector
    from imitation.algos.dice.replay_buffer import ReplayBuffer
    from hydra.utils import instantiate

    results = {"_protocol": {
        "states": "base-policy rollouts (use_teacher_for_collect)",
        "G_hat": "mc_return, sparse success reward, per-chunk gamma (as training)",
        "Q": "finished DICE run's critic, min over ensemble, at a=pi_pre(s,z)",
        "episodes_per_task": episodes_per_task, "task_indices": task_indices,
        "num_z": num_z, "alphas": alphas, "nbins": nbins,
    }}
    for label, ckpt in targets:
        print(f"\n=== {label}: {ckpt}")
        state_dict = utils.load_checkpoint(ckpt)
        run_cfg = OmegaConf.create(state_dict["config"])
        # allow ++dice.* overrides (e.g. actor_layernorm for aln0 ckpts)
        if "dice" in cfg:
            run_cfg.dice = OmegaConf.merge(run_cfg.dice, cfg.dice)
            state_dict["config"] = OmegaConf.to_container(run_cfg, resolve=False)

        env_runner = instantiate(OmegaConf.create(run_cfg.task)["env_runner"])
        bc_policy, student = _build_dice_policy(
            state_dict, env_runner, task_indices, device)

        gamma = float(run_cfg.dice.gamma)

        # cond shape: probe once through the collector's encoder path
        collector = DiceCollector(
            env_runner, bc_policy, student, device,
            max_episode_length=int(run_cfg.dice.max_episode_length),
            use_teacher_for_collect=True)
        te = collector._task_emb(task_indices[0])
        collector._build_env(task_indices[0])
        obs, _ = collector._env.reset()
        cond0, _ = collector._encode_cond(obs, task_indices[0], te)
        cond_shape = (int(cond0.shape[1]), int(cond0.shape[2]))

        replay = ReplayBuffer(
            max_size=500_000, cond_shape=cond_shape,
            horizon=bc_policy.chunk_size, action_dim=bc_policy.network_action_dim,
            device=device, gamma=gamma)

        per_task_succ = {}
        for ti in task_indices:
            succs = []
            for ep in range(episodes_per_task):
                s, _ = collector.rollout_episode(ti, init_idx=ep, replay=replay)
                succs.append(float(s))
            per_task_succ[str(ti)] = float(np.mean(succs))
            print(f"  task {ti}: base success {np.mean(succs):.2f}  replay={len(replay)}")

        m = compute_metrics(student, replay, num_z, alphas, nbins, device)
        m["base_success_per_task"] = per_task_succ
        m["base_success_mean"] = float(np.mean(list(per_task_succ.values())))
        m["gamma"] = gamma
        results[label] = m
        for a in alphas:
            pa = m["per_alpha"][str(a)]
            print(f"  alpha={a}: GOODCOV={pa['goodcov']:.3f} "
                  f"(succ-states {pa['goodcov_success_states']:.3f})  "
                  f"BADENT={pa['badent_nats']:.3f} nats  n_bad={pa['n_bad']}")
        print(f"  dispersion_over_z={m['action_dispersion_over_z']:.4f}  "
              f"qmin_std_over_z={m['qmin_std_over_z']:.4f}")

        os.makedirs(os.path.dirname(out_json), exist_ok=True)
        with open(out_json, "w") as f:
            json.dump(results, f, indent=2)

    print(f"\nWrote {out_json}")


if __name__ == "__main__":
    main()
