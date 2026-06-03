import os
import random
from typing import Optional, Union, Dict, Any, List

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class SequenceVLDataset(Dataset):
    """Dataset wrapper that adds vision-language task information to sequence data.

    Optionally injects a pre-encoded Wan-VAE latent for the 33-frame stack
    ending at sample t. When `wan_cache_path` is set, every sample gains
    ``obs[f"_wan_lat_{cam}"] = (48, 9, 8, 8) fp16`` so the encoder can skip
    its inline Wan-VAE forward. The cache file layout is the one produced by
    ``scripts/cache_libero_bc_wan.py``.

    Args:
        sequence_dataset: Base sequence dataset
        task_emb: Optional task embedding tensor or list of tensors
        lang_inst: Optional language instruction string or list of strings
        task_id: Optional task ID (int/tensor) or list of task IDs
        wan_cache_path: Optional path to per-task Wan-VAE latent cache (.h5)
        wan_cache_cameras: Tuple of camera keys to inject latents for. The
            cache file currently only stores one camera (typically
            "agentview_image"). If multiple are listed and the cache covers
            only the one in its `camera_key` attr, the others are skipped.
    """
    def __init__(
        self,
        sequence_dataset: Dataset,
        task_emb: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        lang_inst: Optional[Union[str, List[str]]] = None,
        task_id: Optional[Union[int, torch.Tensor, List[Union[int, torch.Tensor]]]] = None,
        wan_cache_path: Optional[str] = None,
        wan_cache_cameras: tuple = ("agentview_image",),
    ):
        self.sequence_dataset = sequence_dataset
        # Convert all inputs to lists
        self.task_emb = [task_emb] if task_emb is not None and not isinstance(task_emb, list) else task_emb
        self.lang_inst = [lang_inst] if lang_inst is not None and not isinstance(lang_inst, list) else lang_inst
        self.task_id = [task_id] if task_id is not None and not isinstance(task_id, list) else task_id
        self.n_demos = self.sequence_dataset.n_demos
        self.total_num_sequences = self.sequence_dataset.total_num_sequences

        # ---- Wan-VAE latent cache (optional) -------------------------------
        self.wan_cache_path = wan_cache_path
        self.wan_cache_cameras = tuple(wan_cache_cameras)
        self._wan_cache_file = None    # lazy-opened per worker
        self._wan_cache_camera_key = None  # camera the cache actually stores

        # Map demo_key (e.g. 'demo_17') -> demo_idx into the cached `latents`
        # tensor. SequenceDataset already sorts self.demos by integer suffix
        # and the cache was built with the same sort, so positional indexing
        # is identical; we still build the dict to avoid silently breaking
        # if that contract changes.
        self._demo_key_to_idx = {k: i for i, k in enumerate(self.sequence_dataset.demos)}

        if self.wan_cache_path is not None:
            assert os.path.exists(self.wan_cache_path), (
                f"wan_cache_path does not exist: {self.wan_cache_path}"
            )

    # ------------------------------------------------------------------
    def _ensure_cache_open(self):
        """Lazy-open the cache .h5 — called inside __getitem__ so each
        DataLoader worker process gets its own file handle after fork."""
        if self._wan_cache_file is not None or self.wan_cache_path is None:
            return
        f = h5py.File(self.wan_cache_path, "r", libver="latest", swmr=False)
        # Sanity: the cache must list the same demos in the same order.
        cache_keys = [s.decode() for s in f["demo_keys"][:]]
        for k, ck in zip(self.sequence_dataset.demos, cache_keys):
            assert k == ck, (
                f"demo order mismatch between SequenceDataset and Wan cache: "
                f"{k} vs {ck} (cache={self.wan_cache_path})"
            )
        self._wan_cache_camera_key = f.attrs.get("camera_key", "agentview_image")
        if isinstance(self._wan_cache_camera_key, bytes):
            self._wan_cache_camera_key = self._wan_cache_camera_key.decode()
        self._wan_cache_file = f

    def __len__(self):
        return len(self.sequence_dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return_dict = self.sequence_dataset.__getitem__(idx)

        # Add any provided task information to the return dict, sampling from lists
        if self.task_emb is not None:
            return_dict["task_emb"] = random.choice(self.task_emb)
        if self.lang_inst is not None:
            return_dict["lang_inst"] = random.choice(self.lang_inst)
        if self.task_id is not None:
            return_dict["task_id"] = random.choice(self.task_id)

        # ---- Wan latent injection ---------------------------------------
        if self.wan_cache_path is not None:
            self._ensure_cache_open()
            demo_key = self.sequence_dataset._index_to_demo_id[idx]
            demo_idx = self._demo_key_to_idx[demo_key]
            t_in_demo = idx - self.sequence_dataset._demo_id_to_start_indices[demo_key]
            lat = self._wan_cache_file["latents"][demo_idx, t_in_demo]   # (48, 9, 8, 8) fp16
            lat_t = torch.from_numpy(np.ascontiguousarray(lat))
            if "obs" not in return_dict:
                return_dict["obs"] = {}
            # Only the camera the cache stores gets a key. Other cameras
            # listed in wan_cache_cameras are silently skipped (encoder
            # falls back to live encode for those).
            cam = self._wan_cache_camera_key
            if cam in self.wan_cache_cameras:
                return_dict["obs"][f"_wan_lat_{cam}"] = lat_t

        return return_dict
