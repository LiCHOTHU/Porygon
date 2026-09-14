"""Matched offline DICE-RL / CAST optimization; no environment or ROS commands."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from real_robot.prepare_comparison import OUTPUT
from real_robot.residual_policy import load_base, build_student


def atomic_json(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2))
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=["dice_rl", "cast"])
    parser.add_argument("--experiment-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    directory = args.experiment_dir
    experiment = json.loads((directory / "experiment.json").read_text())
    cfg = experiment["rl"]
    root = directory / args.mode
    root.mkdir(exist_ok=True)
    if (root / "metrics.jsonl").exists():
        raise RuntimeError("Refusing to overwrite an existing experiment")
    torch.set_num_threads(4)
    torch.manual_seed(experiment["seed"])
    rng = np.random.default_rng(experiment["seed"])
    source = torch.load(experiment["base_checkpoint"], map_location="cpu", weights_only=False)
    base = load_base(source)
    replay = torch.load(directory / "replay.pt", map_location="cpu", weights_only=False)
    state_dim = int(np.prod(replay["cond"].shape[1:]))
    student = build_student(base, source["config"], cfg, args.mode, state_dim)
    actor_opt = torch.optim.AdamW(student.actor.parameters(), lr=cfg["actor_lr"], weight_decay=0)
    critic_opt = torch.optim.AdamW(student.critic.parameters(), lr=cfg["critic_lr"], weight_decay=0)
    experts = np.where(replay["data_source"].numpy().ravel() == 1)[0]
    online = np.where(replay["data_source"].numpy().ravel() == 0)[0]
    # Precompute the matched batch-index schedule, independent of actor RNG usage.
    batches = []
    n_expert = int(cfg["batch_size"] * cfg["expert_ratio"])
    for _ in range(cfg["updates"]):
        indices = np.r_[rng.choice(experts, n_expert),
                        rng.choice(online, cfg["batch_size"] - n_expert)]
        rng.shuffle(indices)
        batches.append(indices)
    np.save(root / "batch_indices.npy", np.stack(batches))
    atomic_json(root / "config.json", {"mode": args.mode, "rl": cfg,
                                       "experiment": str(directory / "experiment.json")})

    def checkpoint(step, filename):
        temp = root / (filename + ".tmp")
        torch.save({
            "policy_type": "offline_residual", "actor_mode": args.mode,
            "model": base.state_dict(), "residual_state": student.actor.state_dict(),
            "critic_state": student.critic.state_dict(), "target_critic_state": student.target_critic.state_dict(),
            "actor_optimizer": actor_opt.state_dict(), "critic_optimizer": critic_opt.state_dict(),
            "config": source["config"], "rl_config": cfg, "state_dim": state_dim,
            "rl_update": step, "epoch": -1, "prompt": source["prompt"],
            "action_order": source["action_order"], "parent_checkpoint": experiment["base_checkpoint"],
            "parent_epoch": source["epoch"] + 1,
            "replay_manifest": str(directory / "experiment.json"),
            "checkpoint_selection": "fixed update budget; not selected by training Q or unmeasured task success",
        }, temp)
        temp.replace(root / filename)

    started = time.monotonic()
    status = {"state": "running", "mode": args.mode, "updates": 0, "total_updates": cfg["updates"]}
    atomic_json(root / "status.json", status)
    print(json.dumps(status), flush=True)
    try:
        for step, indices in enumerate(batches, 1):
            b = {k: replay[k][indices].cuda() for k in
                 ("cond", "next_cond", "action", "reward", "done", "n_steps", "data_source")}
            z = torch.randn(len(indices), base.chunk_size, base.action_dim, device="cuda")
            with torch.no_grad():
                next_z = torch.randn_like(z)
                next_action = student.get_action(b["next_cond"], next_z)
                target_q = b["reward"] + cfg["gamma"]**b["n_steps"] * (1 - b["done"]) * student.target_critic(
                    b["next_cond"], next_z, next_action)
            critic = student.critic_loss(b["cond"], z, b["action"], target_q, b["data_source"])
            actor = None
            if step > cfg["critic_warmup"] and step % cfg["actor_frequency"] == 0:
                if args.mode == "dice_rl":
                    actor = student.actor_loss(b["cond"], training_step=step, data_source=b["data_source"])
                else:
                    f = cfg["field"]
                    actor = student.field_actor_loss(
                        b["cond"], mode="field_pointwise", q_source="grad",
                        q_step=f["q_step_size"], bc_step=f["bc_step_size"],
                        q_max_norm=f["q_max_norm"], total_max_norm=f["total_max_norm"],
                        num_particles=f["num_particles"], restore_step=f["restore_step_size"],
                        restore_radius=f["restore_radius"], restore_radius_rms=True)
                if not torch.isfinite(actor["actor_total"]):
                    raise RuntimeError("Non-finite actor loss")
                actor_opt.zero_grad(set_to_none=True)
                actor["actor_total"].backward()
                torch.nn.utils.clip_grad_norm_(student.actor.parameters(), 1.0, error_if_nonfinite=True)
                actor_opt.step()
            if not torch.isfinite(critic["critic_loss"]):
                raise RuntimeError("Non-finite critic loss")
            critic_opt.zero_grad(set_to_none=True)
            critic["critic_loss"].backward()
            torch.nn.utils.clip_grad_norm_(student.critic.parameters(), 1.0, error_if_nonfinite=True)
            critic_opt.step()
            student.update_target_networks(cfg["tau"])
            if step == 1 or step % 100 == 0:
                row = {"update": step, "critic_loss": float(critic["critic_loss"].detach()),
                       "elapsed_s": time.monotonic() - started,
                       "actor_loss": float(actor["actor_total"].detach()) if actor else None,
                       "residual_rms": float(actor["residual_norm"]) if actor else None,
                       "mean_replay_reward": float(b["reward"].mean())}
                with (root / "metrics.jsonl").open("a") as out:
                    out.write(json.dumps(row) + "\n")
                status.update(updates=step, **row)
                atomic_json(root / "status.json", status)
                print(json.dumps(row), flush=True)
            if step % 500 == 0:
                checkpoint(step, "latest.pt")
        checkpoint(cfg["updates"], "final.pt")
        status["state"] = "completed"
    except BaseException as exc:
        status.update(state="failed", error=str(exc))
        atomic_json(root / "status.json", status)
        raise
    atomic_json(root / "status.json", status)


if __name__ == "__main__":
    main()
