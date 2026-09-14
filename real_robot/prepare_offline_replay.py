"""Build episode-safe, timestamped executed-action macro transitions for offline RL."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from real_robot.prepare_comparison import OUTPUT
from real_robot.prepare_data import decode_video, read_joint_csv, interpolate, LEFT_COLUMNS
from real_robot.residual_policy import load_base


def blocks(count, horizon):
    if count < horizon:
        return []
    # Last full window preserves terminal labels without inventing padded controls.
    return sorted(set(range(0, count - horizon + 1, horizon)) | {count - horizon})


def episode(path, expert, success, horizon):
    path = Path(path)
    meta = json.loads((path / "take_meta.json").read_text())
    jt, q = read_joint_csv(path / ("leader_joint.csv" if expert else "follower_joint.csv"))
    frames = {}
    ft = {}
    for key, pattern in (("chest", "chest_rgb_*.avi"), ("wrist", "left_wrist_rgb_*.avi")):
        frames[key] = decode_video(next(path.glob(pattern)), 128)
        ft[key] = meta["ended_at_epoch_s"] - (len(frames[key]) - 1 - np.arange(len(frames[key]))) / 15
    low = max(jt[0], *(t[0] for t in ft.values()))
    high = min(jt[-1], *(t[-1] for t in ft.values()))
    episode_end = high
    if expert:
        command_times = np.arange(low + 1/15, high, 1/15)
        actions = interpolate(jt, q, command_times)
    else:
        table = np.genfromtxt(path / "policy_commands.csv", delimiter=",", names=True)
        command_times = table["timestamp"]
        actions = np.stack([table[k] for k in LEFT_COLUMNS], -1)
        if not np.isfinite(actions).all() or not np.all(np.diff(command_times) > 0):
            raise ValueError(f"Invalid commands: {path}")
        episode_end = float(meta["evaluation"]["policy_ended_at_epoch_s"])
        high = min(high, episode_end)
    records = []
    for i in blocks(len(actions), horizon):
        terminal = i + horizon == len(actions)
        t = command_times[i] - 1/15
        nt = episode_end if terminal else command_times[i + horizon] - 1/15
        limit = episode_end if terminal else high
        if t < low or t > high or nt > limit or nt <= t or command_times[i + horizon - 1] > limit:
            continue
        images = []
        # Terminal next-state is an absorbing placeholder: TD never bootstraps
        # from it. This preserves the final reward when the last sensor sample
        # precedes the last command by a few milliseconds, without fabricating a
        # measured post-command observation or dropping successful terminals.
        next_observation_time = t if terminal else nt
        for stamp in (t, next_observation_time):
            images.append({k: frames[k][max(0, int(np.searchsorted(ft[k], stamp, side="right")) - 1)]
                           for k in frames})
        records.append({"images": images, "proprio": interpolate(jt, q, [t, next_observation_time]).astype(np.float32),
                        "action": actions[i:i+horizon].astype(np.float32),
                        "reward": float(success and terminal), "done": terminal,
                        "n_steps": float((nt - t) * 15), "data_source": int(expert),
                        "episode": path.name, "time": t, "next_time": nt})
    if not records or not records[-1]["done"]:
        raise ValueError(f"Cannot preserve terminal transition for {path}")
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", type=Path, default=OUTPUT)
    directory = parser.parse_args().experiment_dir
    manifest = json.loads((directory / "experiment.json").read_text())
    output = directory / "replay.pt"
    if output.exists():
        raise RuntimeError("Replay already exists; reuse the frozen dataset")
    torch.set_num_threads(4)
    source = torch.load(manifest["base_checkpoint"], map_location="cpu", weights_only=False)
    base = load_base(source)
    records = []
    H = manifest["rl"]["execution_horizon"]
    for path in manifest["expert_demonstrations"]:
        records.extend(episode(path, True, True, H))
    for rollout in manifest["policy_rollouts"]:
        records.extend(episode(rollout["path"], False, rollout["success"], H))
    conditions = []
    with torch.inference_mode():
        for offset in range(0, len(records), 16):
            rows = records[offset:offset+16]
            batch = {f"{view}_rgb": torch.from_numpy(np.stack([
                r["images"][j][view] for r in rows for j in (0, 1)])).permute(0, 3, 1, 2).cuda()
                     for view in ("chest", "wrist")}
            batch["proprio"] = torch.from_numpy(np.stack([r["proprio"][j] for r in rows for j in (0, 1)])).cuda()
            conditions.append(base.get_conditioning(batch).cpu().reshape(len(rows), 2, -1, 256))
    condition = torch.cat(conditions)
    action = torch.zeros(len(records), base.chunk_size, base.action_dim)
    action[:, :H] = base._normalize(torch.from_numpy(np.stack([r["action"] for r in records])).cuda(),
                                    base.action_min, base.action_max).cpu()
    replay = {"cond": condition[:, 0].contiguous(), "next_cond": condition[:, 1].contiguous(),
              "action": action,
              **{k: torch.tensor([r[k] for r in records], dtype=torch.float32).view(-1, 1)
                 for k in ("reward", "done", "n_steps", "data_source")},
              "episode": [r["episode"] for r in records],
              "time": [r["time"] for r in records], "next_time": [r["next_time"] for r in records]}
    torch.save(replay, output)
    report = {"transitions": len(records), "expert_transitions": int(replay["data_source"].sum()),
              "terminal_transitions": int(replay["done"].sum()),
              "positive_terminals": int(replay["reward"].sum()),
              "execution_horizon": H, "base_horizon": base.chunk_size,
              "image_alignment": "Approximate end-anchored video times; no per-frame timestamps in saved AVIs",
              "transition_definition": "8 executed commands per macro transition; final full window may overlap",
              "terminal_definition": "Operator-adjudicated attempt end, including timeout failures",
              "terminal_next_state": "Current-state placeholder, never bootstrapped because done=1",
              "noise": "Behavior latent noise unavailable; critic does not condition on it",
              "behavior_checkpoints": sorted({r["behavior_checkpoint"] for r in manifest["policy_rollouts"]})}
    (directory / "replay_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
