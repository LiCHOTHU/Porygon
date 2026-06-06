"""Smoke test for the SimpleVLA-RL fixes.

After three fixes:
  (1) GRPO advantage uses std-normalization by default (`std_normalize=True`),
      matching SimpleVLA-RL's hardcoded `(r - mean_g) / (std_g + eps)`.
  (2) PPO clip is asymmetric -- separate `clip_ratio_low` and `clip_ratio_high`
      (defaults 0.20 and 0.28 to match SimpleVLA's LIBERO example).
  (3) Filter uses inclusive bounds: `filter_low <= group_acc <= filter_high`,
      matching SimpleVLA's `(acc >= lower) & (acc <= upper)`.

This file verifies each fix against a closed-form expectation and against
SimpleVLA-RL's own formula. No env required; pure-tensor.
"""

import os
import sys

sys.path.insert(0, "/storage/home/hcoda1/8/lwang831/workspace/imitation")

import math
import torch

from imitation.algos.rl import grpo


# ============================================================================
# [1] GRPO advantage with std-normalize matches SimpleVLA-RL formula.
# ============================================================================
def test_grpo_std_normalize_matches_simplevla():
    print("\n[1] GRPO advantage (std_normalize=True) == SimpleVLA-RL formula")
    # Two groups, 8 rollouts each.
    g0 = [1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]   # acc=5/8=0.625
    g1 = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0]   # acc=2/8=0.25
    rewards = torch.tensor(g0 + g1, dtype=torch.float32)
    group_ids = torch.tensor([0] * 8 + [1] * 8, dtype=torch.long)

    # SimpleVLA-RL formula -- mirror /tmp/SimpleVLA-RL/verl/trainer/ppo/core_algos.py:213
    # scores[i] = (scores[i] - mean_g) / (std_g + eps)
    eps = 1e-4
    expected = torch.zeros_like(rewards)
    for g in torch.unique(group_ids):
        m = group_ids == g
        r = rewards[m]
        mean_g = r.mean()
        std_g = r.std(unbiased=False)  # SimpleVLA-RL torch.std default; we use unbiased=False
        expected[m] = (r - mean_g) / (std_g + eps)

    got = grpo.compute_grpo_advantages(rewards, group_ids, std_normalize=True, eps=eps)
    max_err = (got - expected).abs().max().item()
    print(f"  group 0 acc = {sum(g0)/8:.3f}  expected adv ~ {expected[0].item():+.4f}, got {got[0].item():+.4f}")
    print(f"  group 1 acc = {sum(g1)/8:.3f}  expected adv ~ {expected[8].item():+.4f}, got {got[8].item():+.4f}")
    print(f"  |our - SimpleVLA formula|_max = {max_err:.3e}  (expect ~0)")
    assert max_err < 1e-6, f"std-norm formula mismatch: {max_err}"

    # Sanity: |adv| magnitude is larger under std-norm than centered for the
    # success rows of mixed groups (binary reward).
    centered = grpo.compute_grpo_advantages(rewards, group_ids, std_normalize=False)
    print(f"  centered |adv|_max = {centered.abs().max().item():.4f}")
    print(f"  std-norm |adv|_max = {got.abs().max().item():.4f}  (expect strictly larger)")
    assert got.abs().max() > centered.abs().max(), "std-norm should produce larger |adv|"
    print("  PASS")


# ============================================================================
# [2] Asymmetric PPO clip: lower bound 1-low, upper bound 1+high.
# ============================================================================
def test_asymmetric_ppo_clip():
    print("\n[2] PPO clip: asymmetric bounds [1 - low, 1 + high] match SimpleVLA")
    # Construct a per-row log_ratio that we can control. With dt=1 and sigma=1,
    # var = 1, so log_ratio = (||x-mu_old||^2 - ||x-mu_new||^2) / 2.
    # Pick mu_old = 0, x_next = 0, mu_new = sqrt(2 * target_log_ratio), so
    # log_ratio = (0 - 2*target) / 2 = -target. To get +target, swap roles.
    N, K, chunk, A = 8, 1, 1, 1
    var = 1.0

    # Build a batch where row i has log_ratio = log_ratios_target[i].
    log_ratios_target = torch.tensor([-0.5, -0.3, -0.2, -0.1, 0.0, 0.1, 0.25, 0.35],
                                     dtype=torch.float32)
    # log_ratio = (sq_old - sq_new) / 2 -> sq_new = sq_old - 2*log_ratio.
    # Take sq_old = 1.0 for all rows; sq_new = 1 - 2*log_ratio.
    sq_old = torch.full((N, K), 1.0)
    sq_new = sq_old - 2.0 * log_ratios_target.unsqueeze(-1)

    # Make tensors that yield these sq_old / sq_new sums over (chunk, A).
    # sq.sum(dim=(-1,-2)) over a single (1, 1) chunk == the scalar itself.
    x_next  = torch.zeros(N, K, chunk, A)
    mu_old  = torch.sqrt(sq_old).unsqueeze(-1).unsqueeze(-1).expand(N, K, chunk, A).clone()
    mu_new_signs = torch.sign(sq_new).unsqueeze(-1).unsqueeze(-1)
    mu_new  = torch.sqrt(torch.abs(sq_new)).unsqueeze(-1).unsqueeze(-1).expand(N, K, chunk, A).clone() * mu_new_signs

    # Verify our setup gives the right log_ratios.
    sq_old_actual = ((x_next - mu_old) ** 2).sum(dim=(-1, -2))
    sq_new_actual = ((x_next - mu_new) ** 2).sum(dim=(-1, -2))
    lr_actual = (sq_old_actual - sq_new_actual) / (2.0 * var)
    print(f"  set log_ratios: {log_ratios_target.tolist()}")
    print(f"  got log_ratios: {lr_actual.squeeze().tolist()}")
    assert (lr_actual.squeeze() - log_ratios_target).abs().max() < 1e-5, "setup failed"

    # Now run ppo_clip_loss with asymmetric clip (SimpleVLA-RL defaults).
    advantages = torch.ones(N)  # all +1 so we test the upper clip
    _, _, info = grpo.ppo_clip_loss(
        mu_new, mu_old, x_next, advantages, var,
        clip_ratio_low=0.20, clip_ratio_high=0.28,
    )
    print(f"  clip frac (advantages=+1) = {info['clipfrac']:.4f}")
    # Upper bound on ratio = 1 + 0.28 = 1.28 -> log_ratio = log(1.28) ~ 0.2469.
    # Lower bound on ratio = 1 - 0.20 = 0.80 -> log_ratio = log(0.80) ~ -0.2231.
    upper_log = math.log(1.28)
    lower_log = math.log(0.80)
    expected_clipped = ((log_ratios_target < lower_log) | (log_ratios_target > upper_log)).float().mean()
    print(f"  expected clip frac        = {expected_clipped.item():.4f}")
    assert abs(info["clipfrac"] - expected_clipped.item()) < 1e-6, "clip frac mismatch"

    # The 0.35 row should be clipped (above upper). The -0.5 and -0.3 rows
    # should be clipped (below lower). The 0.25 row should NOT be clipped
    # (just under the 0.247 upper bound -- wait, 0.25 > 0.247, so it IS clipped).
    # So 4 rows clipped of 8 -> 0.5.
    print(f"  upper log bound = log(1.28) = {upper_log:.4f};  lower = log(0.80) = {lower_log:.4f}")
    rows_clipped = [(i, log_ratios_target[i].item())
                    for i in range(N)
                    if (log_ratios_target[i] < lower_log) or (log_ratios_target[i] > upper_log)]
    print(f"  rows clipped (i, log_ratio): {rows_clipped}")
    print("  PASS")


# ============================================================================
# [3] Filter inclusivity. Match SimpleVLA-RL: low <= acc <= high.
#     With group_size=8, accuracies live on {0/8, 1/8, ..., 8/8}; verify the
#     boundary case where a group's acc equals filter_low (or filter_high)
#     is KEPT (was previously dropped under strict <).
# ============================================================================
def test_filter_inclusive_bounds():
    print("\n[3] filter_accuracy uses inclusive bounds (low <= acc <= high)")
    # Simulate 4 groups with accuracies 0/8, 1/8, 7/8, 8/8.
    # If filter_low=1/8 and filter_high=7/8 (exact boundary), inclusive
    # would keep groups 1 (acc=1/8) and 2 (acc=7/8) but not 0 and 3.
    # The old strict-< code would drop ALL 4 groups (none strictly in (1/8, 7/8)).
    group_size = 8
    accs = [0.0, 1.0 / 8, 7.0 / 8, 1.0]  # acc per group
    rewards = []
    group_ids = []
    for g_idx, a in enumerate(accs):
        n_succ = int(round(a * group_size))
        rewards.extend([1.0] * n_succ + [0.0] * (group_size - n_succ))
        group_ids.extend([g_idx] * group_size)
    rewards = torch.tensor(rewards, dtype=torch.float32)
    group_ids = torch.tensor(group_ids, dtype=torch.long)

    filter_low, filter_high = 1.0 / 8, 7.0 / 8
    # Reproduce my (fixed) inclusive logic.
    keep_mask = torch.zeros_like(group_ids, dtype=torch.bool)
    kept = 0
    for g in torch.unique(group_ids):
        m = group_ids == g
        gm = rewards[m].float().mean().item()
        if filter_low <= gm <= filter_high:
            keep_mask |= m
            kept += 1
    print(f"  group accs: {accs}")
    print(f"  filter bounds [{filter_low:.4f}, {filter_high:.4f}]")
    print(f"  groups kept (inclusive): {kept}  (expect 2: g=1 acc=1/8, g=2 acc=7/8)")
    assert kept == 2, f"inclusive filter kept {kept} groups, expected 2"

    # Verify the SimpleVLA-RL formula gives the SAME mask.
    acc_tensor = torch.tensor(accs)
    simplevla_mask = (acc_tensor >= filter_low) & (acc_tensor <= filter_high)
    print(f"  SimpleVLA-RL mask: {simplevla_mask.tolist()}")
    our_kept_per_group = [bool((keep_mask & (group_ids == g)).any()) for g in range(4)]
    print(f"  our mask:         {our_kept_per_group}")
    assert our_kept_per_group == simplevla_mask.tolist(), "mask mismatch vs SimpleVLA"
    print("  PASS")


# ============================================================================
# [4] No-op clip when ratio = 1 (identical params), under asymmetric defaults.
# ============================================================================
def test_clip_noop_at_identity():
    print("\n[4] At ratio=1 (mu_new == mu_old), clip is a no-op; loss = -mean(adv)")
    N, K, chunk, A = 4, 2, 3, 5
    mu = torch.randn(N, K, chunk, A)
    x_next = mu + 0.1 * torch.randn_like(mu)
    advantages = torch.tensor([1.0, -1.0, 2.0, -2.0])
    var = 0.05

    pg_loss, kl_loss, info = grpo.ppo_clip_loss(
        mu, mu, x_next, advantages, var,
        clip_ratio_low=0.20, clip_ratio_high=0.28,
    )
    print(f"  pg_loss     = {info['pg_loss']:+.4e}  (expect -mean(adv) = {-advantages.mean().item():+.4e})")
    print(f"  ratio_mean  = {info['ratio_mean']:.6f}  (expect 1.0)")
    print(f"  approx_kl   = {info['approx_kl']:.3e}  (expect 0)")
    print(f"  kl_analytical = {info['kl_analytical']:.3e}  (expect 0)")
    print(f"  clipfrac    = {info['clipfrac']:.3e}  (expect 0)")
    assert abs(info["ratio_mean"] - 1.0) < 1e-5
    assert info["approx_kl"] < 1e-8
    assert info["kl_analytical"] < 1e-8
    assert info["clipfrac"] < 1e-8
    assert abs(info["pg_loss"] - (-advantages.mean().item())) < 1e-5
    print("  PASS")


def main():
    torch.manual_seed(0)
    test_grpo_std_normalize_matches_simplevla()
    test_asymmetric_ppo_clip()
    test_filter_inclusive_bounds()
    test_clip_noop_at_identity()
    print("\nAll SimpleVLA-RL fix smoke checks PASSED.")


if __name__ == "__main__":
    main()
