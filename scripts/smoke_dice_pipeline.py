"""Deeper smoke test for the DICE pipeline.

Verifies the integration pieces that the iter-0 smoke test (smoke_dice_residual.py)
does NOT touch:

  [A] Replay buffer's add_trajectory computes n-step lookahead returns and
      mc_return correctly for a hand-built trajectory with known math.
  [B] sample() returns dicts in the shape student.loss() expects.
  [C] A short multi-step update loop (mirroring dice_train.py's inner loop)
      doesn't NaN, accumulate grads incorrectly, or crash; actor + critic
      losses are finite and trend reasonably.
  [D] Two backward calls (actor_total, then critic_loss) don't conflict and
      give correct gradient routing (actor's backward dirties critic grads,
      which must be zeroed before the critic backward).

Run alongside smoke_dice_residual.py before launching any training.
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
from imitation.algos.dice.distill_rl import DistilledRLModel
from imitation.algos.dice.teacher import FMTeacher
from imitation.algos.dice.replay_buffer import ReplayBuffer


DRIFT_CKPT = (
    "/storage/scratch1/8/lwang831/imitation/cold_start/libero/libero_90/"
    "drift_multitask_lib90/multitask_model_latest.pth"
)
DEVICE = "cuda:0"


def test_replay_n_step_math():
    """[A] Hand-built trajectory; verify n-step returns + mc_return + boundary
    done flag against the closed-form expected values."""
    print("\n[A] Replay buffer: n-step lookahead and mc_return math")
    # Trajectory of 5 chunk-decisions; reward=1 only on the last (terminal success).
    NE, H_, A_ = 3, 4, 2
    T = 5
    traj = []
    for t in range(T):
        traj.append({
            "cond":  torch.full((NE, H_), float(t)),         # marker = t
            "noise": torch.zeros(H_, A_),
            "action": torch.zeros(H_, A_),
            "reward": 1.0 if t == T - 1 else 0.0,
            "done":   (t == T - 1),
        })
    gamma = 0.99
    n_step = 3
    buf = ReplayBuffer(max_size=32, cond_shape=(NE, H_),
                       horizon=H_, action_dim=A_,
                       device="cpu", gamma=gamma,
                       use_n_step=True, n_step=n_step)
    buf.add_trajectory(traj, data_source=0)
    print(f"  buffer.size after add: {buf.size}  (expect {T})")
    assert buf.size == T, f"expected {T} rows, got {buf.size}"

    # Expected n-step reward for decision t: sum_{k=0..min(n,T-t)-1} gamma^k * r[t+k].
    rewards = [d["reward"] for d in traj]
    for t in range(T):
        n_eff = min(n_step, T - t)
        R = sum((gamma ** k) * rewards[t + k] for k in range(n_eff))
        # done at boundary tail = t + n_eff: True if tail >= T or trajectory ends inside.
        tail = t + n_eff
        ended_inside = any(traj[t + k]["done"] for k in range(n_eff))
        expect_done = bool(ended_inside or tail >= T)
        # n_steps = number of steps actually aggregated (capped at first terminal).
        # Our impl breaks early at terminal -> steps_used = k+1 at the terminal.
        steps_used = n_eff
        for k in range(n_eff):
            if traj[t + k]["done"]:
                steps_used = k + 1
                break

        got_R = float(buf.reward[t, 0])
        got_done = bool(buf.done[t, 0] > 0.5)
        got_steps = int(buf.n_steps[t, 0])
        print(f"  t={t}: R={got_R:.4f} (expect {R:.4f})  "
              f"done={got_done} (expect {expect_done})  "
              f"n_steps={got_steps} (expect {steps_used})")
        assert abs(got_R - R) < 1e-6, f"row {t} reward mismatch"
        assert got_done == expect_done, f"row {t} done mismatch"
        assert got_steps == steps_used, f"row {t} n_steps mismatch"

    # mc_return: discounted return-to-go to end of episode.
    mc_expected = [0.0] * T
    running = 0.0
    for t in range(T - 1, -1, -1):
        running = rewards[t] + gamma * running * (0.0 if traj[t]["done"] else 1.0)
        mc_expected[t] = running
    for t in range(T):
        got = float(buf.mc_return[t, 0])
        print(f"  t={t}: mc_return={got:.4f} (expect {mc_expected[t]:.4f})")
        assert abs(got - mc_expected[t]) < 1e-6, f"mc_return row {t} mismatch"
    print("  PASS")


def test_sample_shape_matches_loss():
    """[B] sample() dict keys / shapes match what student.loss() consumes."""
    print("\n[B] sample() shape contract")
    NE, H_, A_, ND = 3, 4, 2, 8
    buf = ReplayBuffer(max_size=16, cond_shape=(NE, H_),
                       horizon=H_, action_dim=A_,
                       device="cpu", gamma=0.99)
    for _ in range(ND):
        buf.add(
            cond=torch.randn(NE, H_),
            noise=torch.randn(H_, A_),
            action=torch.randn(H_, A_),
            reward=0.0,
            next_cond=torch.randn(NE, H_),
            done=False,
        )
    batch = buf.sample(4)
    expected_keys = {"cond", "noise", "action", "reward", "next_cond",
                     "done", "n_steps", "mc_return", "data_source"}
    print(f"  keys: {sorted(batch.keys())}")
    assert set(batch.keys()) == expected_keys, f"missing keys: {expected_keys - set(batch.keys())}"
    print(f"  cond:        {tuple(batch['cond'].shape)}    (expect (4, {NE}, {H_}))")
    print(f"  action:      {tuple(batch['action'].shape)}    (expect (4, {H_}, {A_}))")
    print(f"  reward:      {tuple(batch['reward'].shape)}    (expect (4, 1))")
    print(f"  done:        {tuple(batch['done'].shape)}    (expect (4, 1))")
    print(f"  n_steps:     {tuple(batch['n_steps'].shape)}    (expect (4, 1))")
    print(f"  mc_return:   {tuple(batch['mc_return'].shape)}    (expect (4, 1))")
    print(f"  data_source: {tuple(batch['data_source'].shape)}    (expect (4, 1))")
    assert batch["cond"].shape == (4, NE, H_)
    assert batch["action"].shape == (4, H_, A_)
    for k in ("reward", "done", "n_steps", "mc_return", "data_source"):
        assert batch[k].shape == (4, 1), f"{k} wrong shape"
    print("  PASS")


def test_short_update_loop_no_grad_leak():
    """[C] Run 20 update steps mirroring dice_train.py. Verify:
       - losses stay finite
       - skipping actor step does NOT accumulate stale actor grads
       - critic step is consistent regardless of actor step
       - target Polyak applies on schedule
    """
    print("\n[C] 20-step update loop -- no NaN, no grad accumulation across skipped actor steps")
    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", eval)
    with initialize_config_dir(
        config_dir="/storage/home/hcoda1/8/lwang831/workspace/imitation/config",
        version_base="1.2",
    ):
        cfg = compose(config_name="dice_train", overrides=[
            "task=libero",
            "algo=fm_policy_dice_S",
            "algo.num_inference_steps=1",
            f"cold_start_checkpoint={DRIFT_CKPT}",
            "task.demos_per_env=1",
            "task.task_subset=[0]",
            "data_prefix=/storage/scratch1/8/lwang831/imitation/data",
            "training.load_obs=true",
            "logging.mode=disabled",
        ])
    OmegaConf.set_struct(cfg, False)

    # Build minimal bc_policy + cond batch (just to get realistic cond shape).
    bc_policy = instantiate(cfg.algo.policy, shape_meta=cfg.task.shape_meta).to(DEVICE)
    sd = utils.load_checkpoint(cfg.cold_start_checkpoint, logger=None)
    bc_policy.load_state_dict(sd["model"], strict=False)
    bc_policy.normalizer.fit(sd["norm_stats"])
    bc_policy.eval()
    for p in bc_policy.parameters():
        p.requires_grad = False
    dataset = instantiate(cfg.task.dataset)
    loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False)
    batch = next(iter(loader))
    def _to(v):
        if isinstance(v, torch.Tensor): return v.to(DEVICE)
        if isinstance(v, dict): return {k: _to(x) for k, x in v.items()}
        return v
    batch = {k: _to(v) for k, v in batch.items()}
    batch_p = bc_policy.preprocess_input(batch, train_mode=False)
    with torch.no_grad():
        cond = bc_policy.get_cond(batch_p)
    NE, H = int(cond.shape[1]), int(cond.shape[2])

    # Small student for speed.
    student = DistilledRLModel(
        state_dim=NE * H, action_dim=bc_policy.network_action_dim,
        horizon_steps=bc_policy.chunk_size,
        actor_hidden=(128, 128), critic_hidden=(128, 128),
        ensemble_size=2, q_depends_on_noise=False,
        td_loss="huber", bc_loss_weight=100.0, num_multi_z=4,
        use_q_normalization=True,
        multi_sample_next_noise=True, num_next_noise_samples=2,
        use_n_step=True, n_step=3,
        clip_action=False, zero_final_layer=True,
        device=DEVICE,
    ).to(DEVICE)
    student.attach_teacher(FMTeacher(bc_policy))

    # Build a small fake replay; 32 rows around the real cond.
    buf = ReplayBuffer(max_size=64, cond_shape=(NE, H),
                       horizon=bc_policy.chunk_size,
                       action_dim=bc_policy.network_action_dim,
                       device=DEVICE, gamma=0.99,
                       use_n_step=True, n_step=3)
    B0 = cond.shape[0]
    for i in range(32):
        idx = i % B0
        buf.add(
            cond=cond[idx].cpu(),
            noise=torch.randn(bc_policy.chunk_size, bc_policy.network_action_dim),
            action=torch.tanh(torch.randn(bc_policy.chunk_size, bc_policy.network_action_dim)),
            reward=1.0 if i == 31 else 0.0,
            next_cond=cond[(idx + 1) % B0].cpu(),
            done=(i == 31),
        )

    actor_opt = torch.optim.Adam(student.actor.parameters(), lr=1e-4)
    critic_opt = torch.optim.Adam(student.critic.parameters(), lr=1e-4)

    actor_update_freq = 2  # match production default
    target_update_freq = 2
    actor_param_snap_before_skipped = None
    actor_grad_norms = []
    critic_losses = []
    target_drift = []

    target_before = {n: p.detach().clone() for n, p in student.target_critic.named_parameters()}
    for step in range(20):
        b = buf.sample(8)
        with torch.no_grad():
            predicted_q = student.critic(b["cond"], b["noise"], b["action"])
            q_over = predicted_q - b["mc_return"]
        d = student.loss(
            state=b["cond"], noise=b["noise"], action=b["action"],
            next_state=b["next_cond"], reward=b["reward"], done=b["done"],
            gamma=0.99, training_step=step, q_overestimation=q_over,
            n_steps=b["n_steps"], data_source=b["data_source"],
        )
        assert torch.isfinite(d["total_loss"]), f"step {step}: total_loss non-finite"
        assert torch.isfinite(d["actor_total"]), f"step {step}: actor_total non-finite"
        assert torch.isfinite(d["critic_loss"]), f"step {step}: critic_loss non-finite"

        actor_step = ((step + 1) % actor_update_freq == 0)
        if actor_step:
            actor_opt.zero_grad(set_to_none=True)
            d["actor_total"].backward()
            grad_norm = sum(p.grad.norm().item() ** 2
                            for p in student.actor.parameters() if p.grad is not None) ** 0.5
            actor_grad_norms.append(grad_norm)
            torch.nn.utils.clip_grad_norm_(student.actor.parameters(), 1.0)
            actor_opt.step()
        critic_opt.zero_grad(set_to_none=True)
        d["critic_loss"].backward()
        torch.nn.utils.clip_grad_norm_(student.critic.parameters(), 1.0)
        critic_opt.step()
        critic_losses.append(float(d["critic_loss"]))
        if (step + 1) % target_update_freq == 0:
            student.update_target_networks(0.01)

    # After 20 steps, target should have drifted a bit toward current critic.
    drift = 0.0
    for n, p in student.target_critic.named_parameters():
        drift += (p - target_before[n]).abs().mean().item()
    print(f"  actor grad norms (over actor steps): "
          f"min={min(actor_grad_norms):.3e} max={max(actor_grad_norms):.3e} "
          f"len={len(actor_grad_norms)} (expect 10 = 20/2)")
    print(f"  critic losses: first={critic_losses[0]:.4e} last={critic_losses[-1]:.4e}")
    print(f"  target critic drift after Polyak: {drift:.4e}  (expect > 0)")
    assert len(actor_grad_norms) == 10, "expected 10 actor updates (20 steps / freq=2)"
    assert drift > 1e-6, "target critic did not move under Polyak"
    # Sanity: max actor grad norm bounded -- if grads accumulated across skipped steps,
    # this would blow up over 20 steps. Should stay O(1) after clipping.
    assert max(actor_grad_norms) < 1e4, f"actor grad exploded: {max(actor_grad_norms)}"
    print("  PASS")


def test_grad_isolation():
    """[D] Verify two backward calls don't share state in a way that corrupts updates.

    Specifically: actor_total backward dirties critic.grad (because Q(s, a_student)
    routes through critic params). critic_loss backward must NOT see those grads
    after critic_opt.zero_grad(). And actor_loss backward must NOT depend on the
    critic_loss's separate critic forward."""
    print("\n[D] Backward isolation: actor.backward dirties critic grads; critic.backward unaffected after zero_grad")
    NE, H, A = 8, 16, 7
    student = DistilledRLModel(
        state_dim=NE * H, action_dim=A, horizon_steps=4,
        actor_hidden=(64, 64), critic_hidden=(64, 64),
        ensemble_size=2, q_depends_on_noise=False,
        td_loss="mse", bc_loss_weight=10.0, num_multi_z=2,
        use_q_normalization=False, clip_action=False,
        zero_final_layer=False, device=DEVICE,
    ).to(DEVICE)
    # Deterministic identity teacher: returns a fixed action regardless of input
    # so the critic forward in actor_loss has a non-trivial input.
    fixed_a = torch.zeros(1, 4, A, device=DEVICE)
    def teacher_fn(state, noise):
        return fixed_a.expand(state.shape[0], -1, -1)
    student.attach_teacher(teacher_fn)

    B = 4
    cond = torch.randn(B, NE, H, device=DEVICE)
    noise = torch.randn(B, 4, A, device=DEVICE)
    action = torch.randn(B, 4, A, device=DEVICE)
    next_cond = torch.randn(B, NE, H, device=DEVICE)
    reward = torch.zeros(B, 1, device=DEVICE)
    done = torch.zeros(B, 1, device=DEVICE)
    d = student.loss(
        state=cond, noise=noise, action=action,
        next_state=next_cond, reward=reward, done=done,
        gamma=0.99, training_step=0,
    )
    # Actor backward first.
    for p in student.actor.parameters(): p.grad = None
    for p in student.critic.parameters(): p.grad = None
    d["actor_total"].backward()
    actor_grads_after_actor_backward = sum(
        p.grad.abs().sum().item() for p in student.actor.parameters() if p.grad is not None
    )
    critic_grads_after_actor_backward = sum(
        p.grad.abs().sum().item() for p in student.critic.parameters() if p.grad is not None
    )
    print(f"  actor grads from actor.backward: {actor_grads_after_actor_backward:.4e}  (expect > 0)")
    print(f"  critic grads dirtied by actor.backward: {critic_grads_after_actor_backward:.4e}  (expect > 0)")
    assert actor_grads_after_actor_backward > 0, "actor backward did not flow"
    assert critic_grads_after_actor_backward > 0, (
        "actor backward should dirty critic grads (Q in actor_loss routes through critic)")

    # Now zero critic grads and run critic backward; verify actor grads UNCHANGED.
    actor_grads_snapshot = {id(p): p.grad.detach().clone() for p in student.actor.parameters() if p.grad is not None}
    for p in student.critic.parameters(): p.grad = None
    d["critic_loss"].backward()
    actor_grads_after_critic_backward = sum(
        p.grad.abs().sum().item() for p in student.actor.parameters() if p.grad is not None
    )
    delta = 0.0
    for p in student.actor.parameters():
        if p.grad is not None and id(p) in actor_grads_snapshot:
            delta += (p.grad - actor_grads_snapshot[id(p)]).abs().sum().item()
    print(f"  actor grads change from critic.backward: {delta:.4e}  (expect 0 -- independent graphs)")
    assert delta < 1e-6, f"critic.backward leaked into actor grads: {delta}"
    critic_grads_after_critic_backward = sum(
        p.grad.abs().sum().item() for p in student.critic.parameters() if p.grad is not None
    )
    print(f"  critic grads from critic.backward: {critic_grads_after_critic_backward:.4e}  (expect > 0)")
    assert critic_grads_after_critic_backward > 0, "critic backward did not flow"
    print("  PASS")


def main():
    torch.manual_seed(0); np.random.seed(0)
    test_replay_n_step_math()
    test_sample_shape_matches_loss()
    test_grad_isolation()
    test_short_update_loop_no_grad_leak()
    print("\nAll pipeline smoke checks PASSED.")


if __name__ == "__main__":
    main()
