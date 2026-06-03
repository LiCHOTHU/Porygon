"""Evaluate the previous (std-norm) PolicyDrifting checkpoint by computing
sample_actions vs ground-truth demos MSE on task 0 training data.

This tells us whether the model -- trained with the prior wrong-scale drift
loss -- was actually learning to map state -> action, even though the
drift loss value was nearly flat.
"""
import os, sys, time
sys.path.insert(0, "/storage/home/hcoda1/8/lwang831/workspace/imitation")

import torch
import numpy as np
from omegaconf import OmegaConf
from hydra import initialize_config_dir, compose
from hydra.utils import instantiate

# Register the "eval:" resolver that train.py expects.
if not OmegaConf.has_resolver("eval"):
    OmegaConf.register_new_resolver("eval", eval)

EXP_DIR = "/storage/scratch1/8/lwang831/imitation/local/libero/libero_90/drift_single_t0_local"
DEVICE = "cuda:0"
EPOCHS_TO_CHECK = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

# 1) Build the policy & dataloader using the SAVED training config so encoder/dims match.
with initialize_config_dir(config_dir=EXP_DIR, version_base="1.2"):
    cfg = compose(config_name="config")
OmegaConf.set_struct(cfg, False)

# The saved config has feature_normalize="std" but the new PolicyDrifting __init__ doesn't
# accept that kwarg. Strip it before instantiating.
if "feature_normalize" in cfg.algo.policy:
    del cfg.algo.policy.feature_normalize
# Same for the prior drift_R_list — use whatever was saved, since arch is identical.

print("Building policy from saved config (Hydra) ...")
policy = instantiate(cfg.algo.policy)
policy.eval().to(DEVICE)
print(f"  param count = {sum(p.numel() for p in policy.parameters())}")

# 2) Build a dataset with just task 0.
print("Building task-0 dataset (50 demos) ...")
import imitation.envs.libero.utils as libutils
dataset = instantiate(cfg.task.dataset)
loader = torch.utils.data.DataLoader(
    dataset, batch_size=64, shuffle=False, num_workers=0, pin_memory=True
)

# 3) Eval loop: for each checkpoint, load weights, run sample_actions, compute MSE.
results = []
for ep in EPOCHS_TO_CHECK:
    ckpt = os.path.join(EXP_DIR, f"multitask_model_epoch_{ep:04d}.pth")
    if not os.path.exists(ckpt):
        # Try the "latest" alias for the final epoch
        if ep == EPOCHS_TO_CHECK[-1]:
            ckpt = os.path.join(EXP_DIR, "multitask_model_latest.pth")
        if not os.path.exists(ckpt):
            print(f"  [skip] epoch {ep}: no ckpt")
            continue
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    policy.load_state_dict(state["model"])
    if "norm_stats" in state and state["norm_stats"] is not None:
        policy.normalizer.fit(state["norm_stats"])
    policy.eval().to(DEVICE)

    # Run sample_actions on a few batches and compute per-batch MSE
    mses = []
    n_batches = 0
    for batch in loader:
        # Move to device
        def _move(x):
            if isinstance(x, torch.Tensor):
                return x.to(DEVICE)
            if isinstance(x, dict):
                return {k: _move(v) for k, v in x.items()}
            return x
        batch = _move(batch)
        # Ground truth normalized actions: shape (B, H, A). action key:
        key = "abs_actions" if cfg.algo.abs_action else "actions"
        # NOTE: dataset returns RAW actions; normalize via the policy's normalizer to match
        # sample_actions output space.
        with torch.no_grad():
            # sample_actions returns NORMALIZED actions in [-1, 1] (the model's
            # native output space). The base policy class unnormalizes them
            # externally before sending to the env. So for an apples-to-apples
            # MSE we must normalize the GT actions too.
            out = policy.sample_actions(batch)            # numpy (B, H, A), normalized [-1, 1]
        out_t = torch.from_numpy(out).to(DEVICE)
        gt_raw = batch[key].float()
        gt_norm = policy.normalizer.normalize({key: gt_raw})[key]      # match output space
        # gt may have shape (B, chunk, A); take first chunk_size steps if longer.
        if gt_norm.shape == out_t.shape:
            mse_norm = ((out_t - gt_norm) ** 2).mean().item()
        else:
            mse_norm = ((out_t - gt_norm[:, :out_t.shape[1]]) ** 2).mean().item()
        mses.append(mse_norm)
        n_batches += 1
        if n_batches >= 6:    # cap eval batches per epoch for speed
            break
    mean_mse = float(np.mean(mses))
    results.append((ep, mean_mse))
    print(f"  epoch {ep:3d}: sample_actions MSE = {mean_mse:.6f}   (n={n_batches} batches)")

print("\n=== summary ===")
print("epoch | action MSE")
for ep, m in results:
    print(f"  {ep:3d} | {m:.6f}")
