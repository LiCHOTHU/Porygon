"""Freeze the dataset manifest for three offline real-robot refinements."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from real_robot.prepare_data import read_joint_csv, VALIDATION_TAKES


PROJECT = Path(__file__).resolve().parents[1]
ROOT = Path("/home/pair/Desktop/openarm_exp/openarm_ws")
OUTPUT = PROJECT / "real_robot/runs/jigglypuff_four_way"


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if (OUTPUT / "experiment.json").exists():
        raise RuntimeError("Comparison manifest already exists; reuse it rather than resample demos")
    base = PROJECT / "real_robot/runs/jigglypuff/best.pt"
    source = torch.load(base, map_location="cpu", weights_only=False)
    candidates, excluded = [], []
    supplementary = ROOT / "data_collection_jigglypuff_supplementary"
    for take in sorted(p for p in supplementary.iterdir() if p.is_dir()):
        try:
            times, _ = read_joint_csv(take / "leader_joint.csv")
            if times[-1] - times[0] < 2:
                raise ValueError("Too short for a 30-step chunk")
            if not list(take.glob("chest_rgb_*.avi")) or not list(take.glob("left_wrist_rgb_*.avi")):
                raise ValueError("Missing RGB video")
            candidates.append(take)
        except Exception as exc:
            excluded.append({"path": str(take), "reason": str(exc)})
    if len(candidates) < 30:
        raise RuntimeError("Fewer than 30 usable supplementary demonstrations")
    selected = sorted(np.random.default_rng(42).choice(len(candidates), 30, replace=False))
    experts = [candidates[i] for i in selected]
    online = []
    for take in sorted((ROOT / "policy_evaluations/jigglypuff").iterdir()):
        if not take.is_dir() or take.name.startswith("."):
            continue
        try:
            meta = json.loads((take / "take_meta.json").read_text())
            outcome = meta["evaluation"]["success"]
            if type(outcome) is not bool or meta["evaluation"]["review_status"] != "saved":
                raise ValueError("Missing reviewed binary label")
            if meta.get("success", outcome) != outcome:
                raise ValueError("Conflicting success labels")
            if meta["evaluation"].get("recording_errors"):
                raise ValueError("Recorder reported errors")
            read_joint_csv(take / "follower_joint.csv")
            with (take / "policy_commands.csv").open() as f:
                commands = list(csv.DictReader(f))
            if len(commands) < 8:
                raise ValueError("Fewer than eight executed commands")
            if meta["evaluation"]["execute_steps"] != 8:
                raise ValueError("Executed prefix differs from eight")
            online.append({"path": str(take), "success": outcome,
                           "behavior_checkpoint": meta["evaluation"]["checkpoint"]["path"],
                           "command_count": len(commands)})
        except Exception as exc:
            excluded.append({"path": str(take), "reason": str(exc)})
    if not online or not any(x["success"] for x in online) or all(x["success"] for x in online):
        raise RuntimeError("RL comparison requires reviewed successes AND failures")
    config = dict(source["config"])
    config.update(
        data_roots=[str(ROOT / "data_collection_jigglypuff"), str(supplementary)],
        include_takes=[p.name for p in experts] + sorted(VALIDATION_TAKES),
        validation_takes=sorted(VALIDATION_TAKES),
        cache_dir=str(PROJECT / "real_robot/cache/four_way_bc30"),
        output_dir=str(OUTPUT / "bc_refinement"), initial_checkpoint=str(base),
        freeze_normalizer=True, epochs=40, patience=40, learning_rate=3e-5,
        validation_seed=1729, action_eval_every=5)
    (OUTPUT / "bc_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    experiment = {
        "base_checkpoint": str(base), "base_sha256": hashlib.sha256(base.read_bytes()).hexdigest(),
        "base_epoch": source["epoch"] + 1, "seed": 42,
        "expert_demonstrations": [str(p) for p in experts], "policy_rollouts": online,
        "excluded": excluded, "bc_config": str(OUTPUT / "bc_config.yaml"),
        "validation": "original eight held-out demonstrations; robot success trials not yet run",
        "regime": "offline refinement on existing data, no new robot interaction",
        "rl": {"updates": 6000, "critic_warmup": 500, "actor_frequency": 2,
               "batch_size": 32, "actor_hidden": [256, 256], "critic_hidden": [256, 256],
               "ensemble_size": 5, "actor_lr": 1e-4, "critic_lr": 1e-4,
               "gamma": .99, "tau": .01, "expert_ratio": .5,
               "num_particles": 16, "execution_horizon": 8, "bc_loss_weight": 100,
               "cql_weight": .1, "field": {"q_source": "grad", "num_particles": 16,
               "q_step_size": 1.0, "bc_step_size": .05, "q_max_norm": 1.,
               "total_max_norm": .15, "restore_step_size": 1., "restore_radius": .05,
               "restore_radius_rms": True, "unit_normalize": False, "adaptive_bc": False}},
    }
    (OUTPUT / "experiment.json").write_text(json.dumps(experiment, indent=2))
    print(json.dumps({"expert_demos": len(experts), "rollouts": len(online),
                      "successes": sum(x["success"] for x in online),
                      "failures": sum(not x["success"] for x in online), "excluded": excluded}, indent=2))


if __name__ == "__main__":
    main()
