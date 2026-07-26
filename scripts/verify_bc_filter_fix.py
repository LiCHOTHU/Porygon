"""Local-GPU verification of the BC-filter semantics fix (2026-07-04).

BUG: with use_soft_q_filtering=False (the official robomimic setting used in ALL our
runs), our port applied the BC anchor unconditionally (bc_filter=ones), while the
official code applies the HARD filter (drop BC where Q(student) > Q(teacher)).
Result: the paper's "selective behavior regularization" never fired -> qfilt_keep
was identically 1.000 in every log -> residual pinned at ~0.003 -> DICE == BC.

This script loads a REAL trained drift-DICE checkpoint and calls actor_loss
post-warmup with use_soft_q_filtering=False:
  PASS if q_filtering_active < 1.0 and consistent with better_than_expert_percentage
  (keep fraction ~= 1 - better%), i.e. the filter now actually releases the anchor.

Run:  MUJOCO_GL=egl python scripts/verify_bc_filter_fix.py
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
CK = f"{CEDAR}/imitation/experiments_dice/libero/libero_90/rl_hard8_drift_dice_bcw10/dice_latest.pth"
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
    student.use_soft_q_filtering = False       # official robomimic setting (the buggy path)

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

    # post-warmup call (training_step >> q_filtering_warmup_steps)
    d = student.actor_loss(cond, training_step=10**9)
    keep = float(d["q_filtering_active"])           # mean BC-keep fraction
    better = float(d["better_than_expert_percentage"])
    qadv = float(d["q_advantage_mean"])

    print("=" * 70)
    print("BC-FILTER FIX VERIFICATION (use_soft_q_filtering=False, post-warmup)")
    print(f"  better_than_expert_percentage = {better:.4f}")
    print(f"  q_filtering_active (BC keep)  = {keep:.4f}   (pre-fix: identically 1.0000)")
    print(f"  expected keep ~= 1 - better   = {1.0 - better:.4f}")
    print(f"  q_advantage_mean              = {qadv:+.5f}")
    ok = keep < 0.999 and abs(keep - (1.0 - better)) < 0.05 and better > 0.0
    print("  RESULT:", "PASS — filter now releases the BC anchor per official semantics"
          if ok else "FAIL — filter still not firing (or better%=0 on this batch)")
    print("=" * 70)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
