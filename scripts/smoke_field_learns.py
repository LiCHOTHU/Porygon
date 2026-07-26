"""Local-GPU smoke: does the two-field actor actually LEARN from reward?

On the real critic grad_aQ~=0 (flat), so a null result there is ambiguous. To test
the ALGORITHM, give it a SYNTHETIC informative reward Q(a)=<a,g> (fixed unit g):
then grad_aQ=g (nonzero) and top-K-by-Q is meaningful. If the field update is
correct, the residual should move actions along +g and the measured reward should
CLIMB for BOTH q_source=grad and q_source=zeroth.

Control: with the REAL critic, the same loop should NOT climb (flat critic) -- this
separates "algorithm learns from reward" (yes) from "our critic discriminates" (no).

Run with MUJOCO_GL=egl on a GPU node.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import torch.nn as nn
from hydra.utils import instantiate
from omegaconf import OmegaConf
import imitation.utils.utils as utils
import imitation.envs.libero.wrappers as lw
from powered_eval_matched import _build_dice_policy

CK = "/storage/scratch1/8/lwang831/imitation/experiments_dice/libero/libero_90/fm_dice_cmp_t32/dice_latest.pth"
TASK, B, K, STEPS = 32, 8, 16, 300


class SynCritic(nn.Module):
    """Q(s,a) = <a, g> / s  -- informative, known optimum (push a toward sign(g))."""
    def __init__(self, dim, device):
        super().__init__()
        g = torch.randn(dim, generator=torch.Generator().manual_seed(0))
        self.register_buffer("g", (g / g.norm()).to(device))
    def forward(self, state, noise, action, return_all=False, return_mean=False):
        bsz = action.shape[0]
        q = (action.reshape(bsz, -1) * self.g).sum(-1, keepdim=True)
        return [q, q] if return_all else q


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
    student.use_q_normalization = False   # raw field for a clean synthetic test

    # one real cond, tiled
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
    S = H * A
    real_critic = student.critic
    syn = SynCritic(S, dev)

    # fixed eval particles to measure reward comparably across steps
    cond_K = cond.unsqueeze(1).expand(-1, K, -1, -1).reshape(B * K, *cond.shape[1:])
    z_eval = torch.randn(B * K, H, A, device=dev)

    @torch.no_grad()
    def measure(critic):
        a = student._teacher_fn(cond_K, z_eval) + student.actor(cond_K, z_eval)
        return float(critic(cond_K, z_eval, a).mean()), float(student.actor(cond_K, z_eval).pow(2).mean().sqrt())

    init_actor = {k: v.clone() for k, v in student.actor.state_dict().items()}

    def run(critic, q_source, q_step, bc_step, label):
        student.actor.load_state_dict(init_actor)        # fresh start
        student.critic = critic
        opt = torch.optim.Adam(student.actor.parameters(), lr=1e-3)
        q0, r0 = measure(critic)
        traj = [(0, q0, r0)]
        for step in range(1, STEPS + 1):
            m = student.field_actor_loss(cond, mode="field_distributional", q_source=q_source,
                                         q_step=q_step, bc_step=bc_step, q_max_norm=1.0,
                                         total_max_norm=1.0, num_particles=K, num_bc_particles=K, topk=4)
            opt.zero_grad(set_to_none=True); m["actor_total"].backward(); opt.step()
            if step % 75 == 0:
                q, r = measure(critic); traj.append((step, q, r))
        print(f"\n[{label}]  reward(step0)={q0:+.4f} -> reward(step{STEPS})={traj[-1][1]:+.4f}   "
              f"(Δ={traj[-1][1]-q0:+.4f})   residual_norm {r0:.4f}->{traj[-1][2]:.4f}", flush=True)
        for s, q, r in traj:
            print(f"     step {s:4d}  reward={q:+.4f}  residual_norm={r:.4f}", flush=True)
        return q0, traj[-1][1]

    print("=== SYNTHETIC informative reward Q(a)=<a,g> : algorithm SHOULD climb ===", flush=True)
    sg0, sg1 = run(syn, "grad",   q_step=0.05, bc_step=0.02, label="synthetic / grad")
    sz0, sz1 = run(syn, "zeroth", q_step=0.05, bc_step=0.02, label="synthetic / zeroth")
    print("\n=== REAL critic (flat grad_aQ): control, should NOT climb ===", flush=True)
    rg0, rg1 = run(real_critic, "grad", q_step=0.2, bc_step=0.05, label="real / grad")

    syn_learns = (sg1 - sg0 > 0.05) and (sz1 - sz0 > 0.05)
    print(f"\nSYN grad Δ={sg1-sg0:+.4f}  SYN zeroth Δ={sz1-sz0:+.4f}  REAL grad Δ={rg1-rg0:+.4f}", flush=True)
    print("LEARNS-FROM-REWARD:", "YES (synthetic reward climbed)" if syn_learns else "NO (did not climb)", flush=True)
    sys.exit(0 if syn_learns else 1)


if __name__ == "__main__":
    main()
