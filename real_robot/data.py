import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


LEFT_COLUMNS = [
    "openarm_left_joint1",
    "openarm_left_joint2",
    "openarm_left_joint3",
    "openarm_left_joint4",
    "openarm_left_joint5",
    "openarm_left_joint6",
    "openarm_left_joint7",
    "openarm_left_finger_joint1",
]


class RealRobotDataset(Dataset):
    """Memory-mapped, episode-safe samples produced by prepare_data.py."""

    def __init__(self, cache_dir, split, augment=False):
        self.cache_dir = Path(cache_dir)
        with open(self.cache_dir / "manifest.json") as f:
            manifest = json.load(f)
        self.image_size = int(manifest["image_size"])
        self.augment = augment
        self.episodes = {}
        self.samples = []

        for episode in manifest["episodes"]:
            if episode["split"] != split:
                continue
            name = episode["name"]
            root = self.cache_dir / name
            self.episodes[name] = {
                "chest": np.load(root / "chest_rgb.npy", mmap_mode="r"),
                "wrist": np.load(root / "wrist_rgb.npy", mmap_mode="r"),
                "chest_indices": np.load(root / "chest_indices.npy"),
                "wrist_indices": np.load(root / "wrist_indices.npy"),
                "proprio": np.load(root / "proprio.npy", mmap_mode="r"),
                "actions": np.load(root / "actions.npy", mmap_mode="r"),
            }
            self.samples.extend((name, i) for i in range(episode["samples"]))

    def __len__(self):
        return len(self.samples)

    def _augment_pair(self, chest, wrist):
        if not self.augment:
            return chest, wrist
        pad = 4
        dy = np.random.randint(0, 2 * pad + 1)
        dx = np.random.randint(0, 2 * pad + 1)
        chest = np.pad(chest, ((pad, pad), (pad, pad), (0, 0)), mode="edge")
        wrist = np.pad(wrist, ((pad, pad), (pad, pad), (0, 0)), mode="edge")
        size = self.image_size
        chest = chest[dy:dy + size, dx:dx + size]
        wrist = wrist[dy:dy + size, dx:dx + size]
        gain = np.random.uniform(0.9, 1.1)
        chest = np.clip(chest.astype(np.float32) * gain, 0, 255).astype(np.uint8)
        wrist = np.clip(wrist.astype(np.float32) * gain, 0, 255).astype(np.uint8)
        return chest, wrist

    def __getitem__(self, index):
        name, i = self.samples[index]
        ep = self.episodes[name]
        chest = np.asarray(ep["chest"][ep["chest_indices"][i]])
        wrist = np.asarray(ep["wrist"][ep["wrist_indices"][i]])
        chest, wrist = self._augment_pair(chest, wrist)
        return {
            "chest_rgb": torch.from_numpy(chest.copy()).permute(2, 0, 1),
            "wrist_rgb": torch.from_numpy(wrist.copy()).permute(2, 0, 1),
            "proprio": torch.from_numpy(np.asarray(ep["proprio"][i]).copy()),
            "actions": torch.from_numpy(np.asarray(ep["actions"][i]).copy()),
        }


def load_stats(cache_dir):
    with open(Path(cache_dir) / "stats.json") as f:
        stats = json.load(f)
    return {key: torch.tensor(value, dtype=torch.float32) for key, value in stats.items()}
