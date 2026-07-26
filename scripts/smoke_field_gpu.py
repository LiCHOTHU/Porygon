"""Local-GPU smoke for the two-field residual actor update.

Builds the REAL model from a DICE checkpoint (frozen FM teacher + critic ensemble),
grabs a real cond from one env reset, and runs field_actor_loss through all modes:
  - field_pointwise   / grad
  - field_distributional / grad     (proposed, gradient Q)
  - field_distributional / zeroth   (proposed, zeroth-order Q — survives flat critic)
Checks: finite loss, gradients reach the ACTOR only (critic ~0), and prints the
real ||V_Q|| / ||V_BC|| / cosine so we can see on the actual critic whether grad_a Q
has any magnitude.  Run with MUJOCO_GL=egl on a GPU node.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
import imitation.utils.utils as utils
import imitation.envs.libero.wrappers as lw
from powered_eval_matched import _build_dice_policy

CK = "/storage/scratch1/8/lwang831/imitation/experiments_dice/libero/libero_90/fm_dice_cmp_t32/dice_latest.pth"
TASK = 32
B = 8


def main():
    dev = torch.device("cuda")
    dice_ck = utils.load_checkpoint(CK)
    er_cfg = OmegaConf.create(dice_ck["config"])["task"]["env_runner"]
    er_cfg["rollouts_per_env"] = 2
    env_runner = instantiate(er_cfg)
    bc_policy, student = _build_dice_policy(dice_ck, env_runner, [TASK], dev,
                                            best_of_k=1, det_base=True, return_student=True)
    # _build_dice_policy freezes everything for deploy; re-enable actor grads (training path).
    for p in student.actor.parameters():
        p.requires_grad_(True)

    # one real cond from a single env reset (no full rollout)
    task_emb = {k: v.repeat(1, 1) for k, v in env_runner.benchmark.get_task_emb(TASK).items()}
    penv = env_runner.env_factory(task_id=TASK, benchmark=env_runner.benchmark)
    penv = lw.LiberoVectorWrapper(lambda: lw.LiberoFrameStack(penv, env_runner.frame_stack), 1)
    obs, _ = penv.reset()
    batch = bc_policy._make_batch({k: v for k, v in obs.items()}, TASK, **task_emb)
    batch = bc_policy.preprocess_input(batch, train_mode=False)
    with torch.no_grad():
        cond = bc_policy.get_cond(batch)
    penv.close()
    cond = cond.expand(B, -1, -1).contiguous().to(dev)
    print(f"cond shape {tuple(cond.shape)}  device {cond.device}", flush=True)

    configs = [("field_pointwise", "grad"),
               ("field_distributional", "grad"),
               ("field_distributional", "zeroth"),
               ("field_distributional", "tilted")]
    ok = True
    for mode, qsrc in configs:
        student.actor.zero_grad(set_to_none=True)
        student.critic.zero_grad(set_to_none=True)
        m = student.field_actor_loss(cond, mode=mode, q_source=qsrc,
                                     q_step=0.01, bc_step=0.01,
                                     restore_step=1.0, restore_radius=0.1,
                                     num_particles=16, num_bc_particles=16)
        loss = m["actor_total"]
        loss.backward()
        ag = sum(float(p.grad.abs().sum()) for p in student.actor.parameters() if p.grad is not None)
        cg = sum(float(p.grad.abs().sum()) for p in student.critic.parameters() if p.grad is not None)
        finite = bool(torch.isfinite(loss))
        print(f"\n[{mode} / {qsrc}]", flush=True)
        print(f"  loss={float(loss):.6f} finite={finite}  actor_grad={ag:.4e}  critic_grad={cg:.4e}", flush=True)
        print(f"  ||V_Q||raw={float(m['field_q_raw_norm']):.4e}  ||V_Q||={float(m['field_q_norm']):.4e}  "
              f"||V_BC||={float(m['field_bc_norm']):.4e}  cos(V_Q,V_BC)={float(m['field_q_bc_cosine']):+.3f}", flush=True)
        print(f"  q_delta={float(m['field_q_delta_norm']):.4e}  bc_delta={float(m['field_bc_delta_norm']):.4e}  "
              f"total_delta={float(m['field_total_delta_norm']):.4e}  residual_norm={float(m['residual_norm']):.4e}", flush=True)
        if not finite or ag <= 0 or cg > 1e-3:
            ok = False
            print(f"  *** CHECK FAILED: finite={finite} actor_grad>0={ag>0} critic_grad~0={cg<=1e-3}", flush=True)

    print(f"\n{'SMOKE OK' if ok else 'SMOKE FAILED'}", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
