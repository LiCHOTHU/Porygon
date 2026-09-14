import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import yaml

from real_robot.data import LEFT_COLUMNS


VALIDATION_TAKES = {
    "20260910_162335", "20260910_162649", "20260910_162924",
    "20260910_163439", "20260910_163709", "20260910_164653",
    "20260910_165346", "20260910_165705",
}


def decode_video(path, image_size):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {path}")
    frames = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        frames.append(cv2.resize(rgb, (image_size, image_size), interpolation=cv2.INTER_AREA))
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames decoded from {path}")
    return np.stack(frames)


def read_joint_csv(path):
    table = np.genfromtxt(path, delimiter=",", names=True, dtype=np.float64)
    times = np.asarray(table["timestamp"])
    values = np.stack([np.asarray(table[column]) for column in LEFT_COLUMNS], axis=-1)
    if (times.ndim != 1 or len(times) < 2 or not np.isfinite(times).all()
            or not np.all(np.diff(times) > 0) or not np.isfinite(values).all()):
        raise ValueError(f"Invalid joint data in {path}")
    return times, values


def interpolate(times, values, query):
    return np.stack([np.interp(query, times, values[:, i]) for i in range(values.shape[1])], -1)


def prepare_episode(take, output, image_size, fps, chunk_size):
    chest_path = next(take.glob("chest_rgb_*.avi"))
    wrist_path = next(take.glob("left_wrist_rgb_*.avi"))
    chest = decode_video(chest_path, image_size)
    wrist = decode_video(wrist_path, image_size)
    with open(take / "take_meta.json") as f:
        metadata = json.load(f)
    if float(metadata.get("fps", fps)) != fps:
        raise ValueError(f"Unexpected video FPS in {take}")
    ended = float(metadata["ended_at_epoch_s"])
    joint_times, joints = read_joint_csv(take / "leader_joint.csv")

    chest_times = ended - (len(chest) - 1 - np.arange(len(chest))) / fps
    wrist_times = ended - (len(wrist) - 1 - np.arange(len(wrist))) / fps
    wrist_start = wrist_times[0]
    target_offsets = np.arange(1, chunk_size + 1, dtype=np.float64) / fps

    chest_indices = []
    wrist_indices = []
    obs_times = []
    for chest_index, obs_time in enumerate(chest_times):
        wrist_index = int(round((obs_time - wrist_start) * fps))
        if wrist_index < 0 or wrist_index >= len(wrist):
            continue
        if obs_time < joint_times[0] or obs_time + target_offsets[-1] > joint_times[-1]:
            continue
        chest_indices.append(chest_index)
        wrist_indices.append(wrist_index)
        obs_times.append(obs_time)

    obs_times = np.asarray(obs_times)
    if len(obs_times) == 0:
        raise RuntimeError(f"No synchronized samples in {take}")
    proprio = interpolate(joint_times, joints, obs_times).astype(np.float32)
    action_times = obs_times[:, None] + target_offsets[None]
    actions = interpolate(joint_times, joints, action_times.reshape(-1))
    actions = actions.reshape(len(obs_times), chunk_size, len(LEFT_COLUMNS)).astype(np.float32)

    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "chest_rgb.npy", chest)
    np.save(output / "wrist_rgb.npy", wrist)
    np.save(output / "chest_indices.npy", np.asarray(chest_indices, dtype=np.int32))
    np.save(output / "wrist_indices.npy", np.asarray(wrist_indices, dtype=np.int32))
    np.save(output / "proprio.npy", proprio)
    np.save(output / "actions.npy", actions)
    return len(obs_times), actions, proprio, len(chest), len(wrist)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="real_robot/config.yaml")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    data_root = Path(cfg["data_root"])
    data_roots = [Path(p) for p in cfg.get("data_roots", [str(data_root)])]
    cache_dir = Path(cfg["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_dir / "manifest.json"
    if manifest_path.exists() and not args.overwrite:
        with open(manifest_path) as f:
            existing = json.load(f)
        for key in ("data_root", "image_size", "fps", "chunk_size"):
            if existing[key] != cfg[key]:
                raise ValueError(f"Cache {key} differs from config; choose another cache directory")
        if existing.get("data_roots", [existing["data_root"]]) != [str(p) for p in data_roots]:
            raise ValueError("Cache source roots differ from config")
        print(f"Matching cache already exists: {manifest_path}")
        return

    takes = []
    excluded = list(cfg.get("preparation_exclusions", []))
    all_paths = sorted(p for root in data_roots for p in root.iterdir() if p.is_dir())
    if "include_takes" in cfg:
        included = set(cfg["include_takes"])
        all_paths = [p for p in all_paths if p.name in included]
        missing_included = included - {p.name for p in all_paths}
        if missing_included:
            raise ValueError(f"Requested takes not found: {sorted(missing_included)}")
    names = [p.name for p in all_paths]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate episode names across roots; refusing to overwrite cached episodes")
    for path in all_paths:
        required = ("leader_joint.csv", "take_meta.json", "chest_rgb_*.avi", "left_wrist_rgb_*.avi")
        missing = [pattern for pattern in required if not list(path.glob(pattern))]
        if missing:
            reason = "Missing required files: " + ", ".join(missing)
            excluded.append({"name": path.name, "reason": reason})
            print(f"Excluded {path.name}: {reason}")
        else:
            try:
                read_joint_csv(path / "leader_joint.csv")
            except ValueError as exc:
                reason = str(exc)
                excluded.append({"name": path.name, "reason": reason})
                print(f"Excluded {path.name}: {reason}")
            else:
                takes.append(path)
    if "validation_takes" in cfg:
        validation = set(cfg["validation_takes"])
    elif "validation_fraction" in cfg:
        fraction = float(cfg["validation_fraction"])
        if not 0 < fraction < 1 or len(takes) < 2:
            raise ValueError("Need >=2 takes and a validation fraction in (0, 1)")
        count = min(len(takes) - 1, max(1, round(len(takes) * fraction)))
        indices = np.random.default_rng(int(cfg["seed"])).choice(len(takes), count, replace=False)
        validation = {takes[i].name for i in indices}
    else:
        validation = VALIDATION_TAKES
    actual_validation = {take.name for take in takes} & validation
    if not actual_validation or len(actual_validation) == len(takes):
        raise ValueError("Split must contain both training and validation takes")
    print(f"Episode split: {len(takes) - len(actual_validation)} train / "
          f"{len(actual_validation)} validation")

    episodes = []
    train_actions = []
    train_proprio = []
    for take in takes:
        split = "val" if take.name in validation else "train"
        samples, actions, proprio, chest_frames, wrist_frames = prepare_episode(
            take, cache_dir / take.name, int(cfg["image_size"]), float(cfg["fps"]),
            int(cfg["chunk_size"]),
        )
        episodes.append({
            "name": take.name,
            "source_root": str(take.parent),
            "split": split,
            "samples": samples,
            "chest_frames": chest_frames,
            "wrist_frames": wrist_frames,
        })
        if split == "train":
            train_actions.append(actions)
            train_proprio.append(proprio)
        print(f"{take.name}: {samples} {split} samples")

    actions = np.concatenate(train_actions)
    proprio = np.concatenate(train_proprio)
    stats = {
        "action_min": actions.min(axis=(0, 1)).tolist(),
        "action_max": actions.max(axis=(0, 1)).tolist(),
        "proprio_min": proprio.min(axis=0).tolist(),
        "proprio_max": proprio.max(axis=0).tolist(),
    }
    with open(cache_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    with open(manifest_path, "w") as f:
        json.dump({
            "data_root": str(data_root),
            "data_roots": [str(p) for p in data_roots],
            "image_size": int(cfg["image_size"]),
            "fps": float(cfg["fps"]),
            "chunk_size": int(cfg["chunk_size"]),
            "action_columns": LEFT_COLUMNS,
            "alignment": "each video end anchored to ended_at_epoch_s",
            "excluded_episodes": excluded,
            "episodes": episodes,
        }, f, indent=2)
    print(f"Prepared {sum(x['samples'] for x in episodes)} samples in {cache_dir}")


if __name__ == "__main__":
    main()
