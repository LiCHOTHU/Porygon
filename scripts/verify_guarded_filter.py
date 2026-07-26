"""Local-GPU verification of the GUARDED BC-loss filter (paper Eq. 6) before the
Tier-1 guarded-filter launches (2026-07-06).

Paper Eq. 6: FILTER = [Q(a_cur) >= Q(a_pre)] AND [Q(a_cur) - G_hat(s) <= eps],
i.e. drop the BC anchor only where the critic endorses the edit AND is not
overestimating vs. the Monte-Carlo return. In our port this is
use_soft_q_filtering=True + q_overestimation passed (threshold eps = -0.5).

Checks, on a real trained drift-DICE checkpoint post-warmup:
  A) q_overestimation = -1.0 (deep underestimation, guard PASSES)
     -> keep fraction should equal the unguarded hard filter's (~= 1 - better%)
  B) q_overestimation = 0.0 (guard BLOCKS: 0.0 > eps=-0.5)
     -> keep fraction must be exactly 1.0 (BC anchor everywhere)
  C) q_overestimation = None (official fallback path)
     -> keep ~= 1 - better% (hard filter)

Run:  MUJOCO_GL=egl python scripts/verify_guarded_filter.py
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
CK = sys.argv[1] if len(sys.argv) > 1 else \
    f"{CEDAR}/imitation/experiments_dice/libero/libero_90/rl_hard8_drift_dice_bcw10/dice_latest.pth"
TASK, B = 32, 16


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
    student.use_soft_q_filtering = True
    student.q_underestimation_threshold = -0.5

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

    def probe(q_over):
        d = student.actor_loss(cond, training_step=10**9, q_overestimation=q_over)
        return float(d["q_filtering_active"]), float(d["better_than_expert_percentage"])

    torch.manual_seed(0)
    keep_a, better_a = probe(torch.full((B, 1), -1.0, device=dev))   # guard passes
    torch.manual_seed(0)
    keep_b, better_b = probe(torch.zeros(B, 1, device=dev))          # guard blocks
    torch.manual_seed(0)
    keep_c, better_c = probe(None)                                   # fallback = hard

    print("=" * 72)
    print("GUARDED BC-FILTER VERIFICATION (use_soft_q_filtering=True, eps=-0.5)")
    print(f"  A) q_over=-1.0 (guard passes): keep={keep_a:.4f}  expect ~{1-better_a:.4f}")
    print(f"  B) q_over= 0.0 (guard blocks): keep={keep_b:.4f}  expect 1.0000")
    print(f"  C) q_over= None (fallback)   : keep={keep_c:.4f}  expect ~{1-better_c:.4f}")
    ok = (abs(keep_a - (1 - better_a)) < 0.05 and better_a > 0.0
          and keep_b > 0.999
          and abs(keep_c - (1 - better_c)) < 0.05)
    print("  RESULT:", "PASS — guarded filter implements both Eq. 6 conditions"
          if ok else "FAIL")
    print("=" * 72)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
