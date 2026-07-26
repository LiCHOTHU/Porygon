"""Local-GPU simple test: does the BC-anchor TRUST REGION control the grad residual?

Diag [5] showed unclamped grad ascent on the drift critic runs residual 0.002 -> 1.54
(critic exploitation). This tests whether tightening (q_step, bc_step, total_cap)
keeps the residual in a healthy band (~0.05-0.5) while Q still climbs. Mirrors the
cluster G1/G2/G3 configs so we can read the residual-control BEFORE the jobs finish.

Run:  MUJOCO_GL=egl python scripts/test_drift_grad_trust.py
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
TASK, B, K, STEPS = 32, 8, 16, 200

# (label, q_step, bc_step, total_cap) -- G1/G2/G3 mirror the cluster sweep; UNCLAMPED = diag[5]
CONFIGS = [
    ("UNCLAMPED (diag[5])", 0.2, 0.05, 1.0),
    ("G1 tight",            0.1, 0.10, 0.10),
    ("G2 medium",           0.2, 0.10, 0.20),
    ("G3 loose",            0.3, 0.05, 0.30),
    ("G0 very-tight",       0.05, 0.20, 0.05),
]


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

    print("=" * 74)
    print("TRUST-REGION TEST — grad channel on drift critic (Q is a proxy; success needs rollouts)")
    print(f"  target: residual lands in ~0.05-0.5 (moved but not runaway); Q rises")
    print("=" * 74)
    print(f"{'config':22s} {'Q0':>8s} {'Q_final':>8s} {'dQ':>8s} | {'res0':>7s} {'res_mid':>8s} {'res_final':>9s}")
    print("-" * 74)
    for label, q_step, bc_step, cap in CONFIGS:
        student.actor.load_state_dict(init_actor)
        opt = torch.optim.Adam(student.actor.parameters(), lr=1e-3)
        q0, r0 = measure()
        rmid = None
        for step in range(STEPS):
            m = student.field_actor_loss(cond, mode="field_distributional", q_source="grad",
                                         q_step=q_step, bc_step=bc_step, q_max_norm=1.0,
                                         total_max_norm=cap, num_particles=K, num_bc_particles=K, topk=4)
            opt.zero_grad(set_to_none=True); m["actor_total"].backward(); opt.step()
            if step == STEPS // 2 - 1:
                _, rmid = measure()
        q1, r1 = measure()
        flag = "  <-- healthy" if 0.03 <= r1 <= 0.6 else ("  runaway" if r1 > 0.6 else "  inert")
        print(f"{label:22s} {q0:>8.4f} {q1:>8.4f} {q1-q0:>+8.4f} | {r0:>7.4f} {rmid:>8.4f} {r1:>9.4f}{flag}")
    print("=" * 74)


if __name__ == "__main__":
    main()
