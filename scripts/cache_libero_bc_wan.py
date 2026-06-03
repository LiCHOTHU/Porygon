"""Pre-encode LIBERO-90 BC observations as Wan-VAE latents to disk.

Eliminates the per-batch Wan-VAE forward in `WanDialgaEncoder` so that
r1/r3/r4/r5 train at roughly the vanilla baseline (r2) speed. Without
this cache, the inline frozen Wan-VAE forward dominates the step time
and an epoch takes ~5h on L40S vs ~1h for r2.

For every BC sample (task, demo, t), the trainer's SequenceDataset
returns a 33-frame stack `obs[t-32:t+1]` with the first frame replicated
to handle t < 32. We cache *exactly* the Wan-VAE forward over that
33-frame window so latents drop in 1-to-1 at training time.

Cache layout:
    <cache_dir>/
        manifest.json                   — args + per-task summary
        <task_filename>.h5              — one file per LIBERO task
            latents       fp16  (n_demos, max_T, 48, 9, 8, 8)
            demo_lengths  int32 (n_demos,)
            demo_keys     bytes (n_demos,)             — 'demo_0', 'demo_1', ...

Padding rows past demo_lengths[i] are zeros and never read by the trainer.

Usage:
    python scripts/cache_libero_bc_wan.py \
        --data_dir /storage/home/hcoda1/8/lwang831/workspace/imitation/data/libero/libero_90 \
        --cache_dir /storage/scratch1/8/lwang831/dialga_outputs/libero_bc_wan_cache \
        --camera_key agentview_image \
        --batch_size 16

Resume-safe: skips any task whose .h5 already has a `done` attr set.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch


CHUNK_T = 33                                  # DIALGA pretraining stack length
WAN_LAT_T = (CHUNK_T - 1) // 4 + 1            # = 9 latent frames per stack
WAN_LAT_C, WAN_LAT_H, WAN_LAT_W = 48, 8, 8    # Wan-2.2 TI2V-5B encoder output shape


def load_wan_vae(model_id: str, dtype: torch.dtype, device: torch.device):
    """Frozen Wan-2.2 TI2V-5B VAE. Encoder half only."""
    from diffusers import AutoencoderKLWan
    vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=dtype)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    return vae.to(device)


@torch.no_grad()
def encode_batch(vae, frames_btchw: torch.Tensor, device, dtype) -> torch.Tensor:
    """frames_btchw: (B, T=33, 3, H, W) uint8 in [0, 255]
                 -> latents (B, 48, 9, 8, 8) fp16 on CPU.

    Matches `WanDialgaEncoder._encode_wan`:
        x = float(rgb) / 127.5 - 1  (after the dataset's [0, 1] normalisation
        the encoder feeds 2*x - 1, equivalent end-to-end)
        permute to (B, 3, T, H, W)
    """
    x = frames_btchw.to(device).to(dtype)
    x = (x / 127.5) - 1.0
    x = x.permute(0, 4, 1, 2, 3).contiguous()        # (B, 3, T, H, W)
    out = vae.encode(x)
    z = out.latent_dist.mean if hasattr(out, "latent_dist") else out.latents
    return z.to(torch.float16).cpu()                 # (B, 48, 9, 8, 8)


def build_window(demo_rgb: np.ndarray, t: int) -> np.ndarray:
    """Return the 33-frame stack ending at t with first-frame padding.

    demo_rgb: (T, H, W, 3) uint8 — full demo for this episode.
    t:        int in [0, T) — last frame of the window.

    Mirrors imitation/dataset/sequence_dataset.py::get_sequence_from_demo
    with num_frames_to_stack=32, seq_length=1, pad_same=True.
    """
    T = demo_rgb.shape[0]
    start = max(0, t - (CHUNK_T - 1))
    end = t + 1
    pad_front = max(0, (CHUNK_T - 1) - t)
    real = demo_rgb[start:end]                       # (>=1, H, W, 3)
    if pad_front > 0:
        pad = np.broadcast_to(real[0:1], (pad_front, *real.shape[1:]))
        return np.concatenate([pad, real], axis=0)
    return real


def process_task(
    hdf5_path: Path,
    cache_path: Path,
    camera_key: str,
    vae,
    device,
    dtype,
    batch_size: int,
    n_demos: int | None,
) -> dict:
    """Encode one LIBERO task's hdf5 -> one cache .h5."""
    with h5py.File(hdf5_path, "r") as f:
        # SequenceDataset sorts demos by integer suffix; mirror that.
        demos = list(f["data"].keys())
        demos.sort(key=lambda k: int(k.split("_")[-1]))
        if n_demos is not None:
            demos = demos[:n_demos]

        demo_lengths = np.array(
            [int(f[f"data/{d}"].attrs["num_samples"]) for d in demos], dtype=np.int32
        )
        max_T = int(demo_lengths.max())

        out_shape = (len(demos), max_T, WAN_LAT_C, WAN_LAT_T, WAN_LAT_H, WAN_LAT_W)

        # Resume-aware: if cache exists with done=True, skip.
        if cache_path.exists():
            with h5py.File(cache_path, "r") as cf:
                if cf.attrs.get("done", False) and cf["latents"].shape == out_shape:
                    return {
                        "task": hdf5_path.stem, "status": "skipped",
                        "n_demos": len(demos), "max_T": max_T,
                    }
            cache_path.unlink()

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cf = h5py.File(cache_path, "w")
        cf.attrs["camera_key"] = camera_key
        cf.attrs["chunk_T"] = CHUNK_T
        cf.attrs["source"] = str(hdf5_path)
        cf.create_dataset(
            "latents", shape=out_shape, dtype=np.float16,
            chunks=(1, 1, WAN_LAT_C, WAN_LAT_T, WAN_LAT_H, WAN_LAT_W),
            compression=None,
        )
        cf.create_dataset("demo_lengths", data=demo_lengths)
        cf.create_dataset(
            "demo_keys",
            data=np.array(demos, dtype=h5py.string_dtype(encoding="utf-8")),
        )

        try:
            # Iterate demos; encode in batches of `batch_size` windows.
            buf_frames = []     # list of (33, H, W, 3) uint8 np arrays
            buf_meta = []       # list of (demo_idx, t)

            def flush():
                if not buf_frames:
                    return
                stacks = np.stack(buf_frames, axis=0)              # (B, 33, H, W, 3)
                stacks_t = torch.from_numpy(stacks)
                lats = encode_batch(vae, stacks_t, device, dtype)  # (B, 48, 9, 8, 8) fp16 cpu
                lats_np = lats.numpy()
                for k, (d_idx, t) in enumerate(buf_meta):
                    cf["latents"][d_idx, t] = lats_np[k]
                buf_frames.clear()
                buf_meta.clear()

            for d_idx, demo_key in enumerate(demos):
                rgb_path = f"data/{demo_key}/obs/{camera_key}"
                if rgb_path not in f:
                    raise KeyError(f"missing {rgb_path} in {hdf5_path}")
                demo_rgb = f[rgb_path][...]            # (T, H, W, 3) uint8
                assert demo_rgb.dtype == np.uint8, f"expected uint8, got {demo_rgb.dtype}"
                T = demo_rgb.shape[0]
                assert T == demo_lengths[d_idx], (
                    f"demo length mismatch on {demo_key}: rgb={T} attr={demo_lengths[d_idx]}"
                )
                for t in range(T):
                    buf_frames.append(build_window(demo_rgb, t))
                    buf_meta.append((d_idx, t))
                    if len(buf_frames) >= batch_size:
                        flush()
                # flush at end of each demo to free memory
                flush()

            cf.attrs["done"] = True
        finally:
            cf.close()

    return {
        "task": hdf5_path.stem, "status": "encoded",
        "n_demos": len(demos), "max_T": max_T,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True,
                    help="Directory of LIBERO-90 hdf5 files (one per task).")
    ap.add_argument("--cache_dir", type=str, required=True,
                    help="Destination for per-task .h5 caches + manifest.")
    ap.add_argument("--camera_key", type=str, default="agentview_image",
                    help="Which camera to encode. Must match dialga_cameras "
                         "in the encoder config.")
    ap.add_argument("--model_id", type=str, default="Wan-AI/Wan2.2-TI2V-5B-Diffusers")
    ap.add_argument("--dtype", type=str, default="float16",
                    choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--batch_size", type=int, default=16,
                    help="Wan-VAE forward batch size. 16 fits on L40S-48GB.")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--n_demos", type=int, default=50,
                    help="Demos per task. Matches task.demos_per_env.")
    ap.add_argument("--task_subset", type=str, default=None,
                    help="Optional path to a .txt file with task filenames (one per line) "
                         "to restrict the encode to. Used for sharding across array jobs.")
    ap.add_argument("--start", type=int, default=0,
                    help="Skip the first N tasks. For sharding.")
    ap.add_argument("--end", type=int, default=0,
                    help="Stop before this task index. 0 = end.")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = {"float16": torch.float16,
             "bfloat16": torch.bfloat16,
             "float32": torch.float32}[args.dtype]

    data_dir = Path(args.data_dir)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    hdf5s = sorted(data_dir.glob("*.hdf5"))
    if args.task_subset is not None:
        wanted = set(l.strip() for l in Path(args.task_subset).read_text().splitlines() if l.strip())
        hdf5s = [p for p in hdf5s if p.name in wanted]
    if args.end == 0:
        args.end = len(hdf5s)
    hdf5s = hdf5s[args.start:args.end]
    print(f"[plan] {len(hdf5s)} tasks to encode (slice [{args.start},{args.end}))")

    print(f"[vae ] loading {args.model_id}")
    vae = load_wan_vae(args.model_id, dtype, device)
    n_params = sum(p.numel() for p in vae.parameters())
    print(f"[vae ] {n_params/1e6:.1f}M params, device={device}, dtype={dtype}")

    t0 = time.time()
    manifest = []
    for i, hp in enumerate(hdf5s):
        cache_path = cache_dir / f"{hp.stem}.h5"
        ti = time.time()
        info = process_task(
            hdf5_path=hp,
            cache_path=cache_path,
            camera_key=args.camera_key,
            vae=vae,
            device=device,
            dtype=dtype,
            batch_size=args.batch_size,
            n_demos=args.n_demos,
        )
        info["seconds"] = round(time.time() - ti, 1)
        manifest.append(info)
        elapsed = time.time() - t0
        eta_min = elapsed / max(i + 1, 1) * max(len(hdf5s) - i - 1, 0) / 60
        print(
            f"  [{i+1:>3}/{len(hdf5s)}] {info['status']:>7} "
            f"{info['task'][:60]:60s} "
            f"n_demos={info['n_demos']} max_T={info['max_T']:>3} "
            f"({info['seconds']:.1f}s, ETA {eta_min:.1f} min)"
        )
        # Flush manifest after every task for resumability.
        with (cache_dir / "manifest.json").open("w") as f:
            json.dump({
                "args": vars(args),
                "tasks": manifest,
            }, f, indent=2)

    print(f"\n[done] {len(manifest)} tasks processed in {(time.time()-t0)/60:.1f} min")
    print(f"       manifest: {cache_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
