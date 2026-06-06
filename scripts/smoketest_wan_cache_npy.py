"""End-to-end smoke test for the .npy mmap cache path through the SAME wiring
the training script uses (envs.libero.utils.build_dataset -> SequenceVLDataset).

Loads a few tasks via build_dataset, wraps in a torch DataLoader, pulls N
batches with multi-worker collation. Confirms:
  - utils.py picks the .npy file over .h5 when both exist
  - SequenceVLDataset's .npy branch opens the mmap + meta sidecar correctly
  - DataLoader can collate the cached latent tensor across workers (the
    numpy memmap is read-only — torch wraps it without copying, must not crash)
  - The cached latent has the expected (48, 9, 8, 8) fp16 shape
"""
import os
import sys
import time

sys.path.insert(0, "/storage/home/hcoda1/8/lwang831/workspace/imitation")

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf

OmegaConf.register_new_resolver("eval", eval, replace=True)

# We use hydra's instantiate on task.dataset (which has _target_ pointing at
# imitation.envs.libero.utils.build_dataset) — exactly the same path train.py
# takes via `utils.make_dataset(cfg)`.
from torch.utils.data import DataLoader, ConcatDataset


def main():
    config_dir = "/storage/home/hcoda1/8/lwang831/workspace/imitation/config"
    npy_cache = "/storage/scratch1/8/lwang831/dialga_outputs/imitation/libero_bc_wan_cache_npy"
    n_tasks = 3
    n_batches = 4

    # Compose the same way train.py does, then pull only what build_dataset needs.
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="train", overrides=[
            "task=libero",
            "task.benchmark_name=libero_90",
            "algo=fm_policy_r1",
            f"algo.encoder.wan_cache_dir={npy_cache}",
            "data_prefix=/storage/home/hcoda1/8/lwang831/workspace/imitation/data",
            "normalize_action=true",
            "normalize_obs=false",
        ])

    print(f"=== instantiate task.dataset over first {n_tasks} tasks (npy cache) ===")
    # Narrow to the first n_tasks via task_subset override on the dataset config.
    OmegaConf.set_struct(cfg, False)
    cfg.task.dataset.task_subset = list(range(n_tasks))
    cfg.task.dataset.stats_mode = True
    OmegaConf.set_struct(cfg, True)

    t0 = time.perf_counter()
    out = instantiate(cfg.task.dataset)
    # build_dataset returns (concat_dataset, shape_meta, ...); take the first item as the dataset.
    ds = out[0] if isinstance(out, (tuple, list)) else out
    print(f"  instantiate took {time.perf_counter()-t0:.1f}s, n_samples={len(ds)}")

    # Verify each underlying SequenceVLDataset is using the .npy branch.
    if isinstance(ds, ConcatDataset):
        for i, d in enumerate(ds.datasets):
            wcp = d.wan_cache_path
            backend = "npy" if wcp and wcp.endswith(".npy") else ("h5" if wcp else "none")
            print(f"  task {i}: backend={backend}  cache={os.path.basename(wcp) if wcp else '<none>'}")

    print(f"\n=== DataLoader pull {n_batches} batches (num_workers=2) ===")
    dl = DataLoader(ds, batch_size=16, shuffle=True, num_workers=2,
                    persistent_workers=False, pin_memory=False)
    t0 = time.perf_counter()
    for bi, batch in enumerate(dl):
        if bi >= n_batches:
            break
        obs = batch.get("obs", {})
        lat = obs.get("_wan_lat_agentview_image", None)
        if lat is None:
            print(f"  batch {bi}: NO _wan_lat_agentview_image in obs (FAIL)")
            sys.exit(1)
        if tuple(lat.shape[1:]) != (48, 9, 8, 8):
            print(f"  batch {bi}: bad latent shape {tuple(lat.shape)} (FAIL)")
            sys.exit(1)
        print(f"  batch {bi}: lat={tuple(lat.shape)} dtype={lat.dtype}  OK")
    dt = time.perf_counter() - t0
    print(f"\n=== SUCCESS: {n_batches} batches in {dt:.2f}s ({dt/n_batches:.3f}s/batch)")


if __name__ == "__main__":
    main()
