"""Smoke test for the fixed DICE-RL residual structure.

Three things to verify (no env needed, pure-tensor):
  [1] At iter 0, residual == 0 and a_student == a_teacher exactly (the genuine
      "from prior" start property).
  [2] One synthetic backprop on the actor moves the residual off zero -- gradient
      flows through the residual but not through the teacher (frozen).
  [3] When we swap an FM cold-start for a drift cold-start and set
      num_inference_steps=1, the teacher's action equals drift's deployed
      1-step rule (a = clamp(x_0 + v(x_0, t=drift_t=0, cond), -1, 1)). The
      residual math doesn't care which teacher it sits on top of.
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


SCR_ROOT = "/storage/scratch1/8/lwang831/imitation/cold_start/libero/libero_90"
FM_CKPT    = os.path.join(SCR_ROOT, "cold_multitask_lib90/multitask_model_latest.pth")
DRIFT_CKPT = os.path.join(SCR_ROOT, "drift_multitask_lib90/multitask_model_latest.pth")
DEVICE = "cuda:0"
B = 4


def build(ckpt: str, K: int):
    """Build bc_policy (FM-arch, K inference steps), student, and FM teacher; load ckpt."""
    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", eval)

    with initialize_config_dir(
        config_dir="/storage/home/hcoda1/8/lwang831/workspace/imitation/config",
        version_base="1.2",
    ):
        cfg = compose(config_name="dice_train", overrides=[
            "task=libero",
            "algo=fm_policy_dice_S",
            f"algo.num_inference_steps={K}",
            f"cold_start_checkpoint={ckpt}",
            "task.demos_per_env=1",
            "task.task_subset=[0]",
            "data_prefix=/storage/scratch1/8/lwang831/imitation/data",
            "training.load_obs=true",
            "logging.mode=disabled",
        ])
    OmegaConf.set_struct(cfg, False)

    bc_policy = instantiate(cfg.algo.policy, shape_meta=cfg.task.shape_meta).to(DEVICE)
    sd = utils.load_checkpoint(cfg.cold_start_checkpoint, logger=None)
    missing, unexpected = bc_policy.load_state_dict(sd["model"], strict=False)
    bc_policy.normalizer.fit(sd["norm_stats"])
    bc_policy.eval()
    for p in bc_policy.parameters():
        p.requires_grad = False
    print(f"  loaded ckpt {os.path.basename(ckpt)} -- "
          f"missing={len(missing)} unexpected={len(unexpected)} K={bc_policy.num_inference_steps}")

    # Build a dataset batch to get realistic cond shape and values.
    dataset = instantiate(cfg.task.dataset)
    loader = torch.utils.data.DataLoader(dataset, batch_size=B, shuffle=False)
    batch = next(iter(loader))
    def _to(v):
        if isinstance(v, torch.Tensor): return v.to(DEVICE)
        if isinstance(v, dict): return {k: _to(x) for k, x in v.items()}
        return v
    batch = {k: _to(v) for k, v in batch.items()}
    batch_p = bc_policy.preprocess_input(batch, train_mode=False)
    with torch.no_grad():
        cond = bc_policy.get_cond(batch_p)  # (B, num_enc, hidden)
    num_enc, hidden = int(cond.shape[1]), int(cond.shape[2])
    print(f"  cond shape: ({cond.shape[0]}, num_enc={num_enc}, hidden={hidden})")

    student = DistilledRLModel(
        state_dim=num_enc * hidden,
        action_dim=bc_policy.network_action_dim,
        horizon_steps=bc_policy.chunk_size,
        actor_hidden=(256, 256),  # small for the smoke
        critic_hidden=(256, 256),
        ensemble_size=2,
        q_depends_on_noise=False,  # official default
        conservative="min",
        td_loss="huber",
        bc_loss_weight=100.0,
        num_multi_z=2,
        use_q_normalization=True,
        multi_sample_next_noise=True,
        num_next_noise_samples=2,
        use_n_step=True,
        n_step=3,
        clip_action=False,  # matches official sim repo
        zero_final_layer=True,
        device=DEVICE,
    ).to(DEVICE)
    teacher = FMTeacher(bc_policy)
    student.attach_teacher(teacher)
    return bc_policy, student, teacher, cond


def test_residual_zero_at_init(student, teacher, cond):
    """[1] At iter 0, get_action == teacher exactly."""
    print("\n[1] iter-0 residual == 0 -> a_student == a_teacher exactly")
    noise = torch.randn(cond.shape[0], student.horizon_steps, student.action_dim, device=DEVICE)
    with torch.no_grad():
        a_total, a_teacher = student.get_action(cond, noise, return_pretrained_actions=True)
        a_teacher_direct = teacher(cond, noise)
        # Residual via the actor itself, before adding the teacher.
        residual = student.actor(cond, noise)
    delta_total = (a_total - a_teacher).abs().max().item()
    delta_teach = (a_teacher - a_teacher_direct).abs().max().item()
    res_abs_max = residual.abs().max().item()
    print(f"  |a_student - a_teacher|_max        = {delta_total:.3e}  (expect 0)")
    print(f"  |a_teacher_via_model - direct|_max = {delta_teach:.3e}  (expect 0)")
    print(f"  |residual|_max                     = {res_abs_max:.3e}  (expect 0)")
    assert delta_total == 0.0, f"residual is not zero at init: {delta_total}"
    assert delta_teach == 0.0, f"teacher path mismatch: {delta_teach}"
    assert res_abs_max == 0.0, f"residual nonzero at init: {res_abs_max}"
    print("  PASS")
    return noise


def test_gradient_flows(student, cond, noise):
    """[2] One backward step makes the residual nonzero."""
    print("\n[2] gradient flows through residual; teacher stays frozen")
    # Use the actor_loss (multi-z) to confirm the realistic update path.
    out = student.actor_loss(cond, training_step=0)
    print(f"  actor_total = {float(out['actor_total']):+.4e}")
    print(f"  actor_q_loss = {float(out['actor_q_loss']):+.4e}")
    print(f"  actor_bc_loss (= ||residual||^2) = {float(out['actor_bc_loss']):.4e}  (expect 0 at init)")
    assert float(out["actor_bc_loss"]) == 0.0, "BC loss at init must be 0 (zero residual)"

    out["actor_total"].backward()
    g = None
    for p in student.actor.parameters():
        if p.grad is not None and p.grad.abs().sum() > 0:
            g = p.grad
            break
    assert g is not None, "no actor gradient -- residual path is detached"
    print(f"  actor gradient nonzero: max(|grad|) = {g.abs().max().item():.3e}")
    # teacher (bc_policy) params must have no grad accumulated
    n_teacher_grad = sum(int(p.grad is not None) for p in student._teacher_fn.policy.parameters())
    print(f"  teacher params with grad accumulated: {n_teacher_grad}  (expect 0)")
    assert n_teacher_grad == 0, "teacher should be frozen with no grad"

    # Step the actor and verify residual moves off zero.
    opt = torch.optim.SGD(student.actor.parameters(), lr=1e-3)
    opt.step()
    with torch.no_grad():
        residual = student.actor(cond, noise)
    print(f"  |residual|_max after one SGD step = {residual.abs().max().item():.3e}  (expect > 0)")
    assert residual.abs().max().item() > 0, "residual still zero after SGD step"
    print("  PASS")


def test_full_loss_path(student, cond):
    """[4] End-to-end student.loss(...) with n-step + multi-z next-noise + Q-norm."""
    print("\n[4] full loss() path -- exercises Q-norm + multi-z target + n-step")
    B = cond.shape[0]
    H = student.horizon_steps
    A = student.action_dim
    noise = torch.randn(B, H, A, device=DEVICE)
    action = torch.tanh(torch.randn(B, H, A, device=DEVICE))  # in [-1, 1]-ish
    next_cond = cond + 0.01 * torch.randn_like(cond)
    reward = torch.zeros(B, 1, device=DEVICE)
    reward[0, 0] = 1.0  # one success in the batch
    done = torch.zeros(B, 1, device=DEVICE)
    done[0, 0] = 1.0
    n_steps = torch.full((B, 1), 3.0, device=DEVICE)
    data_source = torch.zeros(B, 1, device=DEVICE)
    # Fake q_overestimation for sQF (small underestimation, threshold gate is moot here).
    q_overestimation = torch.full((B, 1), -0.05, device=DEVICE)

    out = student.loss(
        state=cond, noise=noise, action=action,
        next_state=next_cond, reward=reward, done=done,
        gamma=0.99,
        training_step=0,  # in warmup
        q_overestimation=q_overestimation,
        n_steps=n_steps,
        data_source=data_source,
    )
    print(f"  total_loss     = {float(out['total_loss']):+.4e}")
    print(f"  actor_total    = {float(out['actor_total']):+.4e}")
    print(f"  critic_loss    = {float(out['critic_loss']):+.4e}")
    print(f"  actor_q_loss   = {float(out['actor_q_loss']):+.4e}  (after Q-norm)")
    print(f"  actor_bc_loss  = {float(out['actor_bc_loss']):.4e}  (warmup uniform)")
    print(f"  target_q_mean  = {float(out['target_q_mean']):+.4e}")
    assert torch.isfinite(out["total_loss"]), "total_loss non-finite"
    assert torch.isfinite(out["actor_total"]), "actor_total non-finite"
    assert torch.isfinite(out["critic_loss"]), "critic_loss non-finite"
    # Both backward calls should work without graph issues.
    out["actor_total"].backward(retain_graph=True)
    out["critic_loss"].backward()
    print("  actor + critic backward both completed without error")
    print("  PASS")


def test_drift_teacher_matches_drifts_own_rule(bc_policy, teacher, cond):
    """[3] With K=1 on FM-arch + drift weights, FMTeacher's Euler-1-step output
    equals what drift's own sample_actions would produce: a = clamp(x_0 + v).
    """
    print("\n[3] K=1 FMTeacher on drift ckpt == drift's deployed 1-step rule")
    noise = torch.randn(cond.shape[0], bc_policy.chunk_size,
                        bc_policy.network_action_dim, device=DEVICE)
    with torch.no_grad():
        a_via_teacher = teacher(cond, noise)
        # Drift's deployed rule (drift_t = 0, num_inference_steps = 1):
        x = noise
        t = torch.zeros(x.shape[0], device=DEVICE)
        enc_cache = bc_policy.velocity_net.forward_enc(cond)
        v = bc_policy.velocity_net.forward_dec(x, t, enc_cache)
        a_drift_rule = torch.clamp(x + 1.0 * v, -1, 1)  # dt=1/K=1 for K=1
    delta = (a_via_teacher - a_drift_rule).abs().max().item()
    print(f"  |FMTeacher(K=1) - drift_rule|_max = {delta:.3e}  (expect 0)")
    assert delta == 0.0, f"K=1 teacher path doesn't match drift's own rule: {delta}"
    print("  PASS")


def main():
    torch.manual_seed(0); np.random.seed(0)

    print("=" * 60)
    print("FM cold-start (K=10)")
    print("=" * 60)
    bc_fm, stu_fm, tea_fm, cond_fm = build(FM_CKPT, K=10)
    noise_fm = test_residual_zero_at_init(stu_fm, tea_fm, cond_fm)
    test_gradient_flows(stu_fm, cond_fm, noise_fm)

    print()
    print("=" * 60)
    print("Drift cold-start (K=1) -- the residual-RL swap")
    print("=" * 60)
    bc_dr, stu_dr, tea_dr, cond_dr = build(DRIFT_CKPT, K=1)
    noise_dr = test_residual_zero_at_init(stu_dr, tea_dr, cond_dr)
    test_gradient_flows(stu_dr, cond_dr, noise_dr)
    test_drift_teacher_matches_drifts_own_rule(bc_dr, tea_dr, cond_dr)
    # Rebuild a fresh student for the end-to-end loss test (the prior backward
    # populated grads we don't want to mix with).
    bc_dr2, stu_dr2, _, cond_dr2 = build(DRIFT_CKPT, K=1)
    test_full_loss_path(stu_dr2, cond_dr2)

    print("\nAll DICE-residual smoke checks PASSED.")


if __name__ == "__main__":
    main()
