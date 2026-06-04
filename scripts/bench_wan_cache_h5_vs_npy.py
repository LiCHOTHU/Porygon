"""Microbenchmark: random-access reads of the Wan-VAE latent cache via
h5py (current path) vs np.memmap (proposed path).

For each backend we read N samples at random (demo_idx, t_idx) — the exact
same access pattern as SequenceVLDataset. Report wall-clock per-read and
overall ratio.
"""
import argparse
import os
import time

import h5py
import numpy as np


def bench_h5(h5_path: str, n_reads: int, seed: int = 0):
    with h5py.File(h5_path, "r", libver="latest", swmr=False) as f:
        lat = f["latents"]
        demo_lengths = f["demo_lengths"][:]
        D = lat.shape[0]
        rng = np.random.default_rng(seed)
        demo_idxs = rng.integers(0, D, size=n_reads)
        t_idxs = np.array([rng.integers(0, demo_lengths[d]) for d in demo_idxs])
        # Warm one read (open chunk cache, etc.).
        _ = lat[demo_idxs[0], t_idxs[0]]
        t0 = time.perf_counter()
        for d, t in zip(demo_idxs, t_idxs):
            x = lat[d, t]   # (48, 9, 8, 8) fp16
        dt = time.perf_counter() - t0
    return dt, demo_idxs, t_idxs


def convert_to_npy(h5_path: str, npy_path: str):
    if os.path.exists(npy_path):
        print(f"  [convert] {npy_path} exists, skipping")
        return
    print(f"  [convert] {h5_path} -> {npy_path}")
    with h5py.File(h5_path, "r") as f:
        lat = f["latents"]
        arr = np.array(lat, dtype=np.float16)
    out = np.lib.format.open_memmap(
        npy_path, mode="w+", dtype=np.float16, shape=arr.shape
    )
    out[...] = arr
    out.flush()
    del out
    print(f"  [convert] wrote {arr.nbytes / 1e6:.1f} MB shape={arr.shape}")


def bench_npy(npy_path: str, demo_idxs, t_idxs):
    arr = np.load(npy_path, mmap_mode="r")
    # Warm one read.
    _ = arr[demo_idxs[0], t_idxs[0]]
    t0 = time.perf_counter()
    for d, t in zip(demo_idxs, t_idxs):
        x = arr[d, t]
    dt = time.perf_counter() - t0
    return dt


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--h5", default="/storage/scratch1/8/lwang831/dialga_outputs/imitation/libero_bc_wan_cache/KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet_demo.h5"
    )
    p.add_argument("--npy_dir", default="/storage/scratch1/8/lwang831/dialga_outputs/imitation/libero_bc_wan_cache_npy")
    p.add_argument("--n_reads", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--clear_page_cache", action="store_true",
                   help="warn: requires sudo on most systems, skip on PACE")
    args = p.parse_args()

    os.makedirs(args.npy_dir, exist_ok=True)
    npy_path = os.path.join(args.npy_dir, os.path.basename(args.h5).replace(".h5", ".npy"))

    print(f"=== h5  bench: {args.h5}")
    h5_dt, demo_idxs, t_idxs = bench_h5(args.h5, args.n_reads, args.seed)
    print(f"  {args.n_reads} reads in {h5_dt*1000:.1f} ms "
          f"({h5_dt/args.n_reads*1000:.3f} ms/read)")

    print(f"=== converting to npy")
    convert_to_npy(args.h5, npy_path)

    print(f"=== npy bench: {npy_path}")
    npy_dt = bench_npy(npy_path, demo_idxs, t_idxs)
    print(f"  {args.n_reads} reads in {npy_dt*1000:.1f} ms "
          f"({npy_dt/args.n_reads*1000:.3f} ms/read)")

    print(f"\n=== SUMMARY ===")
    print(f"h5  : {h5_dt*1000:.1f} ms ({h5_dt/args.n_reads*1000:.3f} ms/read)")
    print(f"npy : {npy_dt*1000:.1f} ms ({npy_dt/args.n_reads*1000:.3f} ms/read)")
    print(f"ratio (h5 / npy) = {h5_dt / npy_dt:.2f}x")

    # Project to per-epoch numbers.
    samples_per_epoch = 669_000
    print(f"\n=== per-epoch projection (669k samples) ===")
    print(f"h5  : {h5_dt / args.n_reads * samples_per_epoch:.1f} s "
          f"= {h5_dt / args.n_reads * samples_per_epoch / 60:.1f} min")
    print(f"npy : {npy_dt / args.n_reads * samples_per_epoch:.1f} s "
          f"= {npy_dt / args.n_reads * samples_per_epoch / 60:.1f} min")


if __name__ == "__main__":
    main()
