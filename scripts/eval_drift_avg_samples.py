"""Test the hypothesis that drifting's single-sample inference is noisy.
Compare:
  (a) single sample_actions call (current)
  (b) average over G=4 sample_actions calls (matches training topology)
on the epoch-50 checkpoint.
"""
import os, sys
sys.path.insert(0, "/storage/home/hcoda1/8/lwang831/workspace/imitation")

import torch
import numpy as np
from omegaconf import OmegaConf
from hydra import initialize_config_dir, compose
from hydra.utils import instantiate

if not OmegaConf.has_resolver("eval"):
    OmegaConf.register_new_resolver("eval", eval)

EXP_DIR = "/storage/scratch1/8/lwang831/imitation/local/libero/libero_90/drift_single_t0_local"
CKPT = os.path.join(EXP_DIR, "multitask_model_epoch_0050.pth")
DEVICE = "cuda:0"

with initialize_config_dir(config_dir=EXP_DIR, version_base="1.2"):
    cfg = compose(config_name="config")
OmegaConf.set_struct(cfg, False)
if "feature_normalize" in cfg.algo.policy:
    del cfg.algo.policy.feature_normalize

policy = instantiate(cfg.algo.policy)
state = torch.load(CKPT, map_location="cpu", weights_only=False)
policy.load_state_dict(state["model"])
policy.normalizer.fit(state["norm_stats"])
policy.eval().to(DEVICE)

dataset = instantiate(cfg.task.dataset)
loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)
key = "abs_actions" if cfg.algo.abs_action else "actions"


def _to_dev(x):
    if isinstance(x, torch.Tensor): return x.to(DEVICE)
    if isinstance(x, dict): return {k: _to_dev(v) for k, v in x.items()}
    return x


def eval_mse(num_samples):
    mses = []
    n = 0
    for batch in loader:
        batch = _to_dev(batch)
        gt_raw = batch[key].float()
        gt_norm = policy.normalizer.normalize({key: gt_raw})[key]
        with torch.no_grad():
            outs = []
            for _ in range(num_samples):
                outs.append(torch.from_numpy(policy.sample_actions(batch)).to(DEVICE))
            out = torch.stack(outs, dim=0).mean(dim=0)            # average across noise draws
        if out.shape == gt_norm.shape:
            mse = ((out - gt_norm) ** 2).mean().item()
        else:
            mse = ((out - gt_norm[:, :out.shape[1]]) ** 2).mean().item()
        mses.append(mse)
        n += 1
        if n >= 6:
            break
    return float(np.mean(mses))


for k in (1, 2, 4, 8, 16, 32):
    mse = eval_mse(k)
    print(f"avg over {k:>2} noise draws: action MSE = {mse:.5f}")
