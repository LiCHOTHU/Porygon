"""Local-GPU test of the UNIFIED adaptive-lambda field.

V = q_hat + lambda*bc_hat  (both unit-RMS), lambda = trust-region dual regulated to
hold ||residual|| at res_target. Demonstrates the PG/BC balance becomes AUTOMATIC:
for several targets, lambda should self-adjust so the residual converges to target
while Q climbs -- replacing the manual q_step/bc_step G-sweep with one dimensionless
knob (res_target = how far off the BC manifold you allow the policy to move).

Run:  MUJOCO_GL=egl python scripts/test_unified_field.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
import imitation.utils.utils as utils
import imitation.envs.libero.wrappers as lw
from powered_eval_matched import _build_dice_policy

CEDAR = "/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/imitation_scratch"
CK = f"{CEDAR}/imitation/experiments_dice/libero/libero_90/field_drift_t32_C_distrib_zeroth/dice_latest.pth"
TASK, B, K, STEPS = 32, 8, 16, 800


def main():
    dev = torch.device("cuda")
    dice_ck = utils.load_checkpoint(CK)
    er_cfg = OmegaConf.create(dice_ck["config"])["task"]["env_runner"]
    er_cfg["rollouts_per_env"] = 2
    env_runner = instantiate(er_cfg)
    bc_policy, student = _build_dice_policy(dice_ck, env_runner, [TASK], dev,
                                            best_of_k=1, det_base=True, return_student=True)
    for p in student.actor.parameters():
        p.requires_grad_(True)

    task_emb = {k: v.repeat(1, 1) for k, v in env_runner.benchmark.get_task_emb(TASK).items()}
    penv = env_runner.env_factory(task_id=TASK, benchmark=env_runner.benchmark)
    penv = lw.LiberoVectorWrapper(lambda: lw.LiberoFrameStack(penv, env_runner.frame_stack), 1)
    obs, _ = penv.reset()
    batch = bc_policy.preprocess_input(
        bc_policy._make_batch({k: v for k, v in obs.items()}, TASK, **task_emb), train_mode=False)
    with torch.no_grad():
        cond = bc_policy.get_cond(batch)
    penv.close()
    cond = cond.expand(B, -1, -1).contiguous().to(dev)

    H, A = student.horizon_steps, student.action_dim
    critic = student.critic

    def _rep(x, m):
        return x.unsqueeze(1).expand(-1, m, -1, -1).reshape(B * m, x.shape[1], x.shape[2])
    state_K = _rep(cond, K)
    z = torch.randn(B * K, H, A, device=dev)

    @torch.no_grad()
    def measure():
        a = student._teacher_fn(state_K, z) + student.actor(state_K, z)
        return critic(state_K, z, a).mean().item(), student.actor(state_K, z).pow(2).mean().sqrt().item()

    init_actor = {k: v.clone() for k, v in student.actor.state_dict().items()}

    print("=" * 78)
    print("UNIFIED ADAPTIVE-LAMBDA FIELD — one knob (res_target) auto-balances PG vs BC")
    print("  expect: residual_final ~= res_target, lambda self-adjusts, Q climbs")
    print("=" * 78)
    print(f"{'res_target':>10s} | {'Q0':>7s} {'Qf':>7s} {'dQ':>7s} | {'res_f':>6s} | "
          f"{'lam0':>5s} {'lam_f':>6s} | converged?")
    print("-" * 78)
    for target in (0.10, 0.20, 0.35):
        student.actor.load_state_dict(init_actor)
        student._field_lambda = 1.0
        student._field_res_ema = None
        opt = torch.optim.Adam(student.actor.parameters(), lr=1e-3)
        q0, r0 = measure()
        lam0 = student._field_lambda
        for step in range(STEPS):
            m = student.field_actor_loss(cond, mode="field_distributional", q_source="grad",
                                         q_step=0.05, q_max_norm=1.0, total_max_norm=1.0,
                                         num_particles=K, num_bc_particles=K, topk=4,
                                         unit_normalize=True, adaptive_bc=True,
                                         res_target=target, lambda_lr=2.0, lambda_max=50.0)
            opt.zero_grad(set_to_none=True); m["actor_total"].backward(); opt.step()
        q1, r1 = measure()
        lamf = student._field_lambda
        ok = "YES" if abs(r1 - target) < 0.4 * target else "off"
        print(f"{target:>10.2f} | {q0:>7.3f} {q1:>7.3f} {q1-q0:>+7.3f} | {r1:>6.3f} | "
              f"{lam0:>5.1f} {lamf:>6.2f} | {ok}")
    print("=" * 78)
    print("READ: if residual_final tracks res_target across rows, the dual balance works")
    print("      -> one dimensionless knob replaces the q_step/bc_step/cap G-sweep.")
    print("=" * 78)


if __name__ == "__main__":
    main()
