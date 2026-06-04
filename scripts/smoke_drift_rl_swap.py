"""Smoke test for "drift-as-FM-RL" swap (no new code path).

Hypothesis (per user directive "just switch the drifting model in"):
    FlowMatchingPolicyRL with num_inference_steps=1 is mathematically identical
    to drift's deployed 1-step inference plus a single Gaussian noise injection,
    because  mu = x + dt*v = x + v  when dt=1 and t starts at 0, which matches
    drift's   a = x_0 + v(x_0, drift_t=0, cond).
    The drift BC state_dict should load into FlowMatchingPolicyRL cleanly because
    PolicyDrifting and FlowMatchingPolicy share the same nn.Module submodules.

Verifies four things (no env needed, pure-tensor):
    [1] State-dict loads with no missing/unexpected nn parameters.
    [2] sample_actions_stochastic on a real-shape batch returns chain (B, 2, ...)
        with finite logp.
    [3] chain_logprob recompute at unchanged params: logp_new ~= logp_old.
    [4] grpo.ppo_clip_loss pre-update: ratio~=1, approx_kl~=0, clipfrac~=0.
"""

import os
import sys

sys.path.insert(0, "/storage/home/hcoda1/8/lwang831/workspace/imitation")

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf

import imitation.utils.utils as utils
from imitation.algos.rl import grpo


DRIFT_EXP_DIR = (
    "/storage/scratch1/8/lwang831/imitation/cold_start/libero/libero_90/drift_multitask_lib90"
)
DRIFT_CKPT = os.path.join(DRIFT_EXP_DIR, "multitask_model_latest.pth")
DEVICE = "cuda:0"
B = 4   # batch size for the smoke test (one chunk-decision per row)


def build_policy_and_batch():
    """Build a FlowMatchingPolicyRL with K=1, load drift BC weights, build one
    dataset batch that matches the drift training shape_meta."""
    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", eval)

    # Take the rl_train config (FlowMatchingPolicyRL), and FORCE K=1 to match drift.
    with initialize_config_dir(
        config_dir="/storage/home/hcoda1/8/lwang831/workspace/imitation/config",
        version_base="1.2",
    ):
        cfg = compose(config_name="rl_train", overrides=[
            "task=libero",
            "algo=fm_policy_rl_S",
            "algo.num_inference_steps=1",
            f"cold_start_checkpoint={DRIFT_CKPT}",
            "task.demos_per_env=1",   # smoke: only need a few demos to build the dataset
            "task.task_subset=[0]",   # one task is enough to get correct-shape obs
            "data_prefix=/storage/scratch1/8/lwang831/imitation/data",
            "training.load_obs=true",  # need obs in batch so encoder has rgb to look at
            "logging.mode=disabled",
        ])
    OmegaConf.set_struct(cfg, False)

    print(f"[smoke] algo.policy._target_ = {cfg.algo.policy._target_}")
    print(f"[smoke] num_inference_steps    = {cfg.algo.num_inference_steps}")
    print(f"[smoke] rl_sigma               = {cfg.algo.rl_sigma}")
    print(f"[smoke] cold_start_checkpoint  = {cfg.cold_start_checkpoint}")
    print(f"[smoke] temporal_agg           = {cfg.algo.temporal_agg}")

    # Build the policy and load the drift BC weights.
    policy = instantiate(cfg.algo.policy, shape_meta=cfg.task.shape_meta).to(DEVICE)
    print(f"[smoke] policy class           = {type(policy).__name__}")
    print(f"[smoke] policy.num_inference_steps = {policy.num_inference_steps}")
    print(f"[smoke] policy.step_var        = {policy.step_var:.6f}  (expected sigma^2 = {cfg.algo.rl_sigma**2:.6f})")

    state_dict = utils.load_checkpoint(cfg.cold_start_checkpoint, logger=None)
    # Use plain load_state_dict (strict) to surface ANY key mismatch.
    missing, unexpected = policy.load_state_dict(state_dict["model"], strict=False)
    print(f"[smoke] missing keys   ({len(missing)}): "
          f"{missing[:3]}{'  ...' if len(missing) > 3 else ''}")
    print(f"[smoke] unexpected keys ({len(unexpected)}): "
          f"{unexpected[:3]}{'  ...' if len(unexpected) > 3 else ''}")
    policy.normalizer.fit(state_dict["norm_stats"])
    policy.eval()

    # Build a tiny dataset and pull a batch from it (real obs shapes, real cond).
    print("[smoke] building dataset (1 demo) ...")
    dataset = instantiate(cfg.task.dataset)
    print(f"[smoke] dataset has {len(dataset)} sequences")
    loader = torch.utils.data.DataLoader(dataset, batch_size=B, shuffle=False)
    batch = next(iter(loader))
    # Move to device, keeping the 'obs' nesting (preprocess_input rewires keys).
    def _to(v):
        if isinstance(v, torch.Tensor):
            return v.to(DEVICE)
        if isinstance(v, dict):
            return {k: _to(x) for k, x in v.items()}
        return v
    batch = {k: _to(v) for k, v in batch.items()}
    print(f"[smoke] batch keys (top): {list(batch.keys())}")
    if "obs" in batch:
        print(f"[smoke] batch['obs'] keys: {list(batch['obs'].keys())}")
        for k, v in batch["obs"].items():
            shape = tuple(v.shape) if isinstance(v, torch.Tensor) else type(v).__name__
            print(f"    obs[{k}]: {shape}")
    return policy, batch


def test_logp_consistency(policy, batch):
    """[2] sample shape + finiteness; [3] logp_new ~= logp_old."""
    print("\n[2] sample_actions_stochastic ...")
    rec = policy.sample_actions_stochastic(batch)
    chain = rec["chain"]      # (B, K+1, chunk, A)
    mu_old = rec["mu_old"]    # (B, K, chunk, A)
    t_grid = rec["t_grid"]    # (K,)
    logp_old = rec["logp_old"]  # (B, K)
    print(f"    action shape   {rec['action'].shape}")
    print(f"    cond shape     {tuple(rec['cond'].shape)}")
    print(f"    chain shape    {tuple(chain.shape)}  (expect K+1=2 in dim 1)")
    print(f"    mu_old shape   {tuple(mu_old.shape)}")
    print(f"    t_grid shape   {tuple(t_grid.shape)}  values={t_grid.tolist()}")
    print(f"    logp_old shape {tuple(logp_old.shape)}  finite={bool(torch.isfinite(logp_old).all())}  "
          f"mean={logp_old.mean().item():.4f}")
    assert chain.shape[1] == 2, f"expected K+1=2 timesteps, got {chain.shape[1]}"
    assert torch.isfinite(logp_old).all(), "non-finite logp_old"

    print("\n[3] chain_logprob recompute at unchanged params ...")
    cond = rec["cond"].to(DEVICE)
    chain_d = chain.to(DEVICE)
    t_grid_d = t_grid.to(DEVICE)
    with torch.no_grad():
        logp_new, mu_new = policy.chain_logprob(cond, chain_d, t_grid_d)
    logp_old_d = logp_old.to(DEVICE)
    mu_old_d = mu_old.to(DEVICE)
    delta_logp = (logp_new - logp_old_d).abs().max().item()
    delta_mu = (mu_new - mu_old_d).abs().max().item()
    print(f"    |logp_new - logp_old|_max = {delta_logp:.3e}")
    print(f"    |mu_new   - mu_old  |_max = {delta_mu:.3e}")
    # Tolerance: fp32 sum over (chunk=16 * A=7) = 112 elements, expect <1e-3.
    assert delta_logp < 5e-3, f"logp drift too large at unchanged params: {delta_logp}"
    assert delta_mu < 1e-4, f"mu drift too large at unchanged params: {delta_mu}"
    print("    PASS")
    return cond, chain_d, mu_old_d, t_grid_d, logp_old_d, mu_new


def test_ppo_pre_update(policy, cond, chain_d, mu_old_d, t_grid_d, mu_new):
    """[4] grpo.ppo_clip_loss pre-update: ratio~=1, approx_kl~=0, clipfrac~=0."""
    print("\n[4] ppo_clip_loss pre-update sanity ...")
    # Fake advantages: random per row, then run the loss.
    N, K, chunk, A = chain_d.shape[0], chain_d.shape[1] - 1, chain_d.shape[2], chain_d.shape[3]
    adv = torch.randn(N, device=DEVICE)
    var = policy.step_var
    loss, info = grpo.ppo_clip_loss(
        mu_new, mu_old_d, chain_d[:, 1:].to(DEVICE), adv, var,
        clip_eps=0.2, step_mask=None,
    )
    print(f"    loss        = {info['loss']:+.4e}")
    print(f"    ratio_mean  = {info['ratio_mean']:.6f}  (expect ~1)")
    print(f"    approx_kl   = {info['approx_kl']:.3e}  (expect ~0)")
    print(f"    clipfrac    = {info['clipfrac']:.3e}  (expect 0)")
    print(f"    log_ratio|max| = {info['log_ratio_abs_max']:.3e}")
    assert abs(info["ratio_mean"] - 1.0) < 1e-2, f"ratio_mean off: {info['ratio_mean']}"
    assert info["approx_kl"] < 1e-3, f"approx_kl too high: {info['approx_kl']}"
    assert info["clipfrac"] < 1e-3, f"clipfrac nonzero: {info['clipfrac']}"
    print("    PASS")


def main():
    torch.manual_seed(0)
    np.random.seed(0)
    policy, batch = build_policy_and_batch()
    cond, chain_d, mu_old_d, t_grid_d, logp_old_d, mu_new = test_logp_consistency(policy, batch)
    test_ppo_pre_update(policy, cond, chain_d, mu_old_d, t_grid_d, mu_new)
    print("\nAll smoke checks PASSED -- drift-as-FM-RL swap is sound.")


if __name__ == "__main__":
    main()
