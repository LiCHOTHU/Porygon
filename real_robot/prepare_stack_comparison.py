"""Stacking: success-only BC and matched DICE/CAST from frozen stack v2."""
import csv
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import torch
import yaml

from real_robot.prepare_comparison import PROJECT, ROOT
from real_robot.prepare_data import decode_video, read_joint_csv, interpolate


OUTPUT = PROJECT / "real_robot/runs/stack_four_way"


def cache_success(take, cache, horizon=30):
    meta = json.loads((take / "take_meta.json").read_text())
    joint_t, joint = read_joint_csv(take / "follower_joint.csv")
    command_t, command = read_joint_csv(take / "policy_commands.csv")
    videos = {k: decode_video(next(take.glob(pattern)), 128)
              for k, pattern in (("chest", "chest_rgb_*.avi"), ("wrist", "left_wrist_rgb_*.avi"))}
    ended = meta["ended_at_epoch_s"]
    times = {k: ended - (len(v) - 1 - np.arange(len(v))) / 15 for k, v in videos.items()}
    low = max(joint_t[0], command_t[0], *(v[0] for v in times.values()))
    high = min(joint_t[-1], command_t[-1], meta["evaluation"]["policy_ended_at_epoch_s"])
    indices = np.flatnonzero((times["chest"] >= low) & (times["chest"] + horizon / 15 <= high))
    if not len(indices):
        raise ValueError("No complete future command chunks in successful rollout")
    obs_t = times["chest"][indices]
    wrist_indices = np.searchsorted(times["wrist"], obs_t, side="right") - 1
    target_t = obs_t[:, None] + np.arange(1, horizon + 1)[None] / 15
    # Commands are held between publications; never interpolate fictitious actions.
    command_indices = np.searchsorted(command_t, target_t, side="right") - 1
    destination = cache / take.name
    destination.mkdir(parents=True, exist_ok=False)
    for key, values in videos.items():
        np.save(destination / f"{key}_rgb.npy", values)
    np.save(destination / "chest_indices.npy", indices.astype(np.int32))
    np.save(destination / "wrist_indices.npy", wrist_indices.astype(np.int32))
    np.save(destination / "proprio.npy", interpolate(joint_t, joint, obs_t).astype(np.float32))
    np.save(destination / "actions.npy", command[command_indices].astype(np.float32))
    return {"name": take.name, "source_root": str(take.parent), "split": "train",
            "samples": len(indices), "state_source": "follower_joint.csv",
            "action_source": "policy_commands.csv, zero-order hold", "success": True}


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if (OUTPUT / "experiment.json").exists() or (OUTPUT / "base.pt").exists():
        raise RuntimeError("Stacking experiment already exists; refusing to overwrite it")
    source_path = PROJECT / "real_robot/runs/stack_continued_v2/best.pt"
    source = torch.load(source_path, map_location="cpu", weights_only=False)
    if source["prompt"] != "stack the blocks":
        raise ValueError("Not the expected stacking base")
    parent_cache = Path(source["config"]["cache_dir"])
    parent_manifest = json.loads((parent_cache / "manifest.json").read_text())
    rollouts, excluded = [], []
    for take in sorted((ROOT / "policy_evaluations/stack").iterdir()):
        if not take.is_dir() or take.name.startswith("."):
            continue
        try:
            meta = json.loads((take / "take_meta.json").read_text())
            evaluation = meta["evaluation"]
            success = evaluation["success"]
            if type(success) is not bool or meta.get("success", success) != success:
                raise ValueError("Missing or conflicting binary outcome")
            if evaluation["review_status"] != "saved" or evaluation.get("recording_errors"):
                raise ValueError("Unreviewed or recording-error trajectory")
            if meta["fps"] != 15 or evaluation["execute_steps"] != 8:
                raise ValueError("Incompatible control rate/prefix")
            read_joint_csv(take / "follower_joint.csv")
            ct, _ = read_joint_csv(take / "policy_commands.csv")
            if len(ct) < 8:
                raise ValueError("Fewer than eight executed commands")
            if not list(take.glob("chest_rgb_*.avi")) or not list(take.glob("left_wrist_rgb_*.avi")):
                raise ValueError("Missing RGB recordings")
            rollouts.append({"path": str(take), "success": success, "command_count": len(ct),
                             "behavior_checkpoint": evaluation["checkpoint"]["path"],
                             "behavior_epoch": evaluation["checkpoint"].get("epoch"),
                             "termination_reason": evaluation["termination_reason"]})
        except Exception as exc:
            excluded.append({"path": str(take), "reason": str(exc)})
    if not any(r["success"] for r in rollouts) or not any(not r["success"] for r in rollouts):
        raise RuntimeError("Need usable successes and failures for the three-way refinement")
    train_demos = [e for e in parent_manifest["episodes"] if e["split"] == "train"]
    chosen = np.random.default_rng(42).choice(len(train_demos), 30, replace=False)
    experts = [str(Path(train_demos[i]["source_root"]) / train_demos[i]["name"]) for i in sorted(chosen)]
    cache = PROJECT / "real_robot/cache/stack_four_way_bc"
    cache.mkdir(parents=True, exist_ok=True)
    episodes = []
    for rollout in rollouts:
        if rollout["success"]:
            episodes.append(cache_success(Path(rollout["path"]), cache))
    for ep in parent_manifest["episodes"]:
        if ep["split"] == "val":
            episodes.append(dict(ep, cache_path=str(Path(ep.get("cache_path", parent_cache / ep["name"])).resolve())))
    stats = {k: source["model"][k].tolist() for k in
             ("action_min", "action_max", "proprio_min", "proprio_max")}
    (cache / "stats.json").write_text(json.dumps(stats, indent=2))
    (cache / "manifest.json").write_text(json.dumps({
        "data_root": str(ROOT / "policy_evaluations/stack"), "image_size": 128,
        "fps": 15, "chunk_size": 30, "episodes": episodes,
        "alignment": "Legacy video end anchoring; follower proprioception and future held commands",
        "excluded_episodes": excluded}, indent=2))
    frozen_base = OUTPUT / "base.pt"
    shutil.copy2(source_path, frozen_base)
    cfg = dict(source["config"])
    for key in ("export_dir", "include_takes", "data_roots", "preparation_exclusions"):
        cfg.pop(key, None)
    cfg.update(cache_dir=str(cache), output_dir=str(OUTPUT / "bc_refinement"),
               initial_checkpoint=str(frozen_base), freeze_normalizer=True,
               epochs=40, patience=40, learning_rate=1e-5,
               refinement_data_root=str(ROOT / "policy_evaluations/stack"),
               refinement_action_source="successful rollout policy_commands.csv",
               plot_title="Stacking success-only BC refinement from frozen v2")
    (OUTPUT / "bc_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    # Match the already-validated real-robot DICE/CAST recipe from the first study.
    rl = json.loads((PROJECT / "real_robot/runs/jigglypuff_four_way/experiment.json").read_text())["rl"]
    experiment = {
        "base_checkpoint": str(frozen_base), "base_source": str(source_path),
        "base_sha256": hashlib.sha256(frozen_base.read_bytes()).hexdigest(),
        "base_epoch": source["epoch"] + 1, "seed": 42,
        "expert_demonstrations": experts, "policy_rollouts": rollouts, "excluded": excluded,
        "bc_training": "Only successful saved policy evaluations; no expert replay in BC",
        "bc_success_episodes": sum(r["success"] for r in rollouts),
        "bc_config": str(OUTPUT / "bc_config.yaml"), "rl": rl,
        "validation": "Same 27 held-out demonstration episodes as base v2; not used for RL replay",
        "export_dir": str(ROOT / "policy_checkpoints/stack_four_way"),
        "regime": "Offline refinement with shared 30-demo expert replay for the two RL arms",
    }
    (OUTPUT / "experiment.json").write_text(json.dumps(experiment, indent=2))
    print(json.dumps({"base_epoch": experiment["base_epoch"], "rollouts": len(rollouts),
                      "successes": sum(r["success"] for r in rollouts),
                      "failures": sum(not r["success"] for r in rollouts),
                      "bc_train_samples": sum(e["samples"] for e in episodes if e["split"] == "train"),
                      "rl_expert_demos": len(experts), "excluded": excluded}, indent=2))


if __name__ == "__main__":
    main()
