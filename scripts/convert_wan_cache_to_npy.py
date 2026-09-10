"""Convert every Wan-VAE latent cache .h5 in --src_dir to a .npy memmap in
--dst_dir, parallelized across workers. Output shape/dtype identical to the
.h5 "latents" dataset.
"""
import argparse
import json
import multiprocessing as mp
import os
import time

import h5py
import numpy as np


def convert_one(args):
    h5_path, dst_dir = args
    base = os.path.basename(h5_path).replace(".h5", "")
    npy_path = os.path.join(dst_dir, f"{base}.npy")
    meta_path = os.path.join(dst_dir, f"{base}.meta.json")
    if os.path.exists(npy_path) and os.path.exists(meta_path):
        return (base, "skipped", 0.0, 0)
    t0 = time.perf_counter()
    with h5py.File(h5_path, "r") as f:
        lat = f["latents"]
        arr = np.array(lat, dtype=np.float16)
        demo_keys = [s.decode() if isinstance(s, bytes) else s for s in f["demo_keys"][:]]
        demo_lengths = np.array(f["demo_lengths"], dtype=np.int32).tolist()
        cam = f.attrs.get("camera_key", "agentview_image")
        if isinstance(cam, bytes):
            cam = cam.decode()
        chunk_T = int(f.attrs.get("chunk_T", 33))
    out = np.lib.format.open_memmap(
        npy_path, mode="w+", dtype=np.float16, shape=arr.shape
    )
    out[...] = arr
    out.flush()
    del out
    meta = {
        "demo_keys": demo_keys,
        "demo_lengths": demo_lengths,
        "camera_key": cam,
        "chunk_T": chunk_T,
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "source_h5": h5_path,
    }
    with open(meta_path, "w") as fh:
        json.dump(meta, fh, indent=2)
    return (base, "ok", time.perf_counter() - t0, arr.nbytes)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src_dir", default="/storage/scratch1/8/lwang831/dialga_outputs/imitation/libero_bc_wan_cache")
    p.add_argument("--dst_dir", default="/storage/scratch1/8/lwang831/dialga_outputs/imitation/libero_bc_wan_cache_npy")
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()

    os.makedirs(args.dst_dir, exist_ok=True)
    h5_files = sorted(
        os.path.join(args.src_dir, f) for f in os.listdir(args.src_dir) if f.endswith(".h5")
    )
    print(f"Found {len(h5_files)} .h5 files in {args.src_dir}")
    print(f"Writing to {args.dst_dir} with {args.workers} workers")

    work = [(p_, args.dst_dir) for p_ in h5_files]
    total_bytes = 0
    t0 = time.perf_counter()
    with mp.Pool(args.workers) as pool:
        for i, (name, status, dt, nbytes) in enumerate(pool.imap_unordered(convert_one, work), 1):
            total_bytes += nbytes
            print(f"[{i:3d}/{len(h5_files)}] {status:8s} {dt:5.1f}s  {nbytes/1e6:6.1f} MB  {name}")
    total_dt = time.perf_counter() - t0
    print(f"\nDone in {total_dt:.1f}s, total {total_bytes/1e9:.1f} GB")


if __name__ == "__main__":
    main()
