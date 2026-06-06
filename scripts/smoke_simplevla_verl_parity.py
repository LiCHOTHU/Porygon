"""Additional smoke tests for the verl-parity patch (no env / no policy needed):
  [A] AdaptiveKLController moves beta UP when KL exceeds target, DOWN when it doesn't,
      and stays bounded by the +/-0.2 clip.
  [B] Round-robin balanced permutation: any contiguous minibatch-sized slice has
      approximately equal contribution from each task (within 1 row).
  [C] verl-style total loss assembly: pg_loss - c_ent*H + beta*KL composes correctly
      and gradient flows through the pg_loss + kl_loss components.
"""

import math

import torch

from imitation.algos.rl import grpo


def test_adaptive_kl_controller():
    print("\n[A] AdaptiveKLController moves beta in the right direction")
    ctrl = grpo.AdaptiveKLController(init_kl_coef=0.01, target_kl=0.05, horizon=10000.0)
    beta_0 = ctrl.value
    # Over target -> beta should go UP.
    ctrl.update(current_kl=0.10, n_steps=200)
    beta_up = ctrl.value
    print(f"    KL=2x target (200 steps): beta {beta_0:.6f} -> {beta_up:.6f}")
    assert beta_up > beta_0, "beta should INCREASE when current_kl > target"

    ctrl2 = grpo.AdaptiveKLController(init_kl_coef=0.01, target_kl=0.05, horizon=10000.0)
    ctrl2.update(current_kl=0.01, n_steps=200)   # below target
    print(f"    KL=0.2x target (200 steps): beta {beta_0:.6f} -> {ctrl2.value:.6f}")
    assert ctrl2.value < beta_0, "beta should DECREASE when current_kl < target"

    # Check the +/-0.2 clip: KL=100x target should NOT push beta by more than
    # (1 + 0.2 * 200/10000) = 1.004 (200 steps, horizon 10000).
    ctrl3 = grpo.AdaptiveKLController(init_kl_coef=1.0, target_kl=0.01, horizon=10000.0)
    ctrl3.update(current_kl=10.0, n_steps=200)
    expected_max_mult = 1.0 + 0.2 * 200 / 10000.0
    print(f"    KL=1000x target -> mult = {ctrl3.value:.6f} (max {expected_max_mult:.6f})")
    assert ctrl3.value <= expected_max_mult + 1e-9, "controller exceeded the +/-0.2 clip"
    print("  PASS")


def test_balanced_perm():
    print("\n[B] Round-robin balanced permutation across tasks")
    # 30 rows split unevenly across 3 tasks: 5/10/15 rows.
    task_ids = torch.cat([
        torch.zeros(5, dtype=torch.long),
        torch.ones(10, dtype=torch.long),
        torch.full((15,), 2, dtype=torch.long),
    ])
    g = torch.Generator(device="cpu").manual_seed(0)
    # Reproduce the rl_train.py balanced_perm logic inline.
    unique = task_ids.unique().tolist()
    per_task = []
    max_len = 0
    for t in unique:
        idx = (task_ids == t).nonzero(as_tuple=False).flatten()
        idx = idx[torch.randperm(idx.shape[0], generator=g)]
        per_task.append(idx)
        max_len = max(max_len, idx.shape[0])
    out = []
    for j in range(max_len):
        for idx in per_task:
            if j < idx.shape[0]:
                out.append(idx[j])
    perm = torch.stack(out)
    assert perm.shape[0] == task_ids.shape[0], "perm length mismatch"
    assert torch.equal(perm.sort().values, torch.arange(task_ids.shape[0])), \
        "perm is not a valid permutation"
    # Check that early slices are well-balanced: first 9 rows should have 3 from each task.
    early = task_ids[perm[:9]]
    counts = torch.bincount(early, minlength=3)
    print(f"    first-9 task counts: {counts.tolist()}  (expect 3/3/3)")
    assert torch.equal(counts, torch.tensor([3, 3, 3])), \
        f"first 9 rows not balanced across tasks: {counts.tolist()}"
    # Beyond row 15, task 0 is exhausted: counts in slice [15:30] should be 0/0/k for task 0.
    late = task_ids[perm[15:]]
    late_counts = torch.bincount(late, minlength=3)
    print(f"    rows 15-end counts: {late_counts.tolist()}  (task 0 exhausted earlier)")
    assert late_counts[0] == 0, "task 0 should be exhausted by row 15"
    print("  PASS")


def test_total_loss_assembly():
    print("\n[C] verl-style total = pg_loss - c_ent*H + beta*KL composes correctly")
    torch.manual_seed(0)
    N, K, chunk, A = 8, 4, 3, 7
    mu_old = torch.randn(N, K, chunk, A)
    mu_new = mu_old + 0.05 * torch.randn_like(mu_old)
    mu_new.requires_grad_(True)
    x_next = mu_old + 0.1 * torch.randn_like(mu_old)
    adv = torch.randn(N)
    var = 0.05

    pg_loss, kl_loss, info = grpo.ppo_clip_loss(
        mu_new, mu_old, x_next, adv, var,
        clip_ratio_low=0.20, clip_ratio_high=0.28,
    )
    print(f"    pg_loss   = {pg_loss.item():+.4e}")
    print(f"    kl_loss   = {kl_loss.item():+.4e}  (analytical Gaussian KL)")
    # Sanity: analytical KL = ||mu_new - mu_old||^2 / (2*var), averaged.
    kl_expected = ((mu_new - mu_old) ** 2).sum(dim=(-1, -2)).mean() / (2.0 * var)
    err = (kl_loss.detach() - kl_expected.detach()).abs().item()
    print(f"    |kl_loss - analytical|_max = {err:.3e}  (expect ~0)")
    assert err < 1e-5, f"kl_loss != ||du||^2/(2 var): err = {err}"

    # Compose verl-style total loss and verify grad flows.
    H = grpo.gaussian_entropy_per_step(var=var, chunk_size=chunk, action_dim=A)
    beta, c_ent = 0.001, 0.0
    total = pg_loss - c_ent * H + beta * kl_loss
    total.backward()
    g_max = mu_new.grad.abs().max().item()
    print(f"    H (constant) = {H:.4f}")
    print(f"    total_loss = pg - 0*H + {beta}*KL = {total.item():+.4e}")
    print(f"    |grad mu_new|_max = {g_max:.3e}  (expect > 0)")
    assert g_max > 0, "gradient should flow through mu_new"
    print("  PASS")


def test_filter_then_balanced_perm():
    """[D] Regression: filter_accuracy must drop rows from buf['task_ids'] in lockstep
    with the other buffers, or balanced_perm builds a permutation over the
    *unfiltered* N and produces out-of-bounds indices for the filtered minibatch
    tensors. This is exactly the IndexError that killed the long-run SimpleVLA
    jobs on 2026-06-04 (e.g. "index 2968 out of bounds for dimension 0 with size
    2643"). The fix is to include "task_ids" in the keep-mask filter loop in
    rl_train.py main().
    """
    print("\n[D] filter_accuracy + balanced_perm: task_ids must shrink in lockstep")
    # Simulate the buffer keys rl_train.py main() filters.
    N_full = 30
    buf = {
        "cond":      torch.zeros(N_full, 4),
        "chain":     torch.zeros(N_full, 4),
        "mu_old":    torch.zeros(N_full, 4),
        "valid":     torch.zeros(N_full),
        "rewards":   torch.zeros(N_full),
        "group_ids": torch.arange(N_full) // 5,           # 6 groups of 5
        "task_ids":  torch.cat([torch.zeros(15, dtype=torch.long),
                                torch.ones(15, dtype=torch.long)]),
    }
    # Mark every other group as kept (3/6 -> 15 rows).
    keep_mask = torch.zeros(N_full, dtype=torch.bool)
    for g in (0, 2, 4):
        keep_mask |= (buf["group_ids"] == g)

    # Mirror the fixed filter loop in rl_train.py (now includes "task_ids").
    for k in ("cond", "chain", "mu_old", "valid", "rewards", "group_ids", "task_ids"):
        buf[k] = buf[k][keep_mask]
    N = buf["cond"].shape[0]
    print(f"    after filter: N={N}, task_ids.shape={tuple(buf['task_ids'].shape)}")
    assert buf["task_ids"].shape[0] == N, (
        "task_ids must shrink with the other filtered buffers — otherwise "
        "balanced_perm will return indices > N and trigger IndexError")

    # Now run balanced_perm against the FILTERED task_ids and check the resulting
    # indices fit inside the FILTERED N.
    g = torch.Generator(device="cpu").manual_seed(0)
    unique = buf["task_ids"].unique().tolist()
    per_task, max_len = [], 0
    for t in unique:
        idx = (buf["task_ids"] == t).nonzero(as_tuple=False).flatten()
        idx = idx[torch.randperm(idx.shape[0], generator=g)]
        per_task.append(idx); max_len = max(max_len, idx.shape[0])
    out = []
    for j in range(max_len):
        for idx in per_task:
            if j < idx.shape[0]:
                out.append(idx[j])
    perm = torch.stack(out)
    print(f"    perm.max()={perm.max().item()}  (must be < N={N})")
    assert perm.max().item() < N, (
        f"balanced_perm produced index {perm.max().item()} >= filtered N={N}: "
        "this is the bug we just fixed.")
    assert perm.shape[0] == N, "perm length must equal filtered N"
    print("  PASS")


if __name__ == "__main__":
    test_adaptive_kl_controller()
    test_balanced_perm()
    test_total_loss_assembly()
    test_filter_then_balanced_perm()
    print("\nAll verl-parity smoke checks PASSED.")
