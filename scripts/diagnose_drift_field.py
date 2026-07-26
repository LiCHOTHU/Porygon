"""Local-GPU diagnosis: WHY is the drifting-field update inert on the DRIFT base?

Loads a real DRIFT DICE checkpoint (task 32, one-step drift base + trained critic)
and measures the 5 things that gate whether the two-field update can move the policy:

  1. CRITIC ACTION-SENSITIVITY  ||grad_a Q||  (+ finite-diff SNR).  If ~0 -> grad
     channel is dead (no reward direction).
  2. Q-SPREAD across the K candidates  std(Q)/|Q|.  If ~0 -> zeroth top-K == mean,
     nowhere to drift (the zeroth diversity bottleneck).
  3. CANDIDATE DISPERSION  rms spread of base+residual particles.
  4. FIELD -> REWARD ALIGNMENT: step particles along the (grad / zeroth) field and
     check whether the REAL critic's Q actually goes UP.
  5. IN-LOOP: run field_actor_loss for 150 steps against the REAL critic; does Q(a)
     climb and does residual_norm grow off ~0.003?

Run:  MUJOCO_GL=egl python scripts/diagnose_drift_field.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
import imitation.utils.utils as utils
import imitation.envs.libero.wrappers as lw
from powered_eval_matched import _build_dice_policy
from imitation.algos.dice.drift_field import compute_q_gradient_field, compute_drift_field, clip_field_norm

CEDAR = "/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/imitation_scratch"
CK = f"{CEDAR}/imitation/experiments_dice/libero/libero_90/field_drift_t32_C_distrib_zeroth/dice_latest.pth"
TASK, B, K = 32, 8, 16


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

    # one real task-32 cond, tiled to B
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
    critic = student.critic

    def _rep(x, m):
        return x.unsqueeze(1).expand(-1, m, -1, -1).reshape(B * m, x.shape[1], x.shape[2])

    state_K = _rep(cond, K)
    z = torch.randn(B * K, H, A, device=dev)
    with torch.no_grad():
        base = student._teacher_fn(state_K, z)
        res = student.actor(state_K, z)
        cur = base + res
        q_cur = critic(state_K, z, cur)            # (B*K,1) conservative min
        q_scale = q_cur.abs().mean().clamp_min(1e-8)

    print("=" * 68)
    print(f"DRIFT-FIELD DIAGNOSIS on {os.path.basename(os.path.dirname(CK))}")
    print(f"  base ckpt one-step drift | B={B} K={K} action_dim={A} horizon={H} S={S}")
    print("=" * 68)

    # ---- 1. critic action-sensitivity ----
    qf = compute_q_gradient_field(critic, state_K, z, cur, q_scale=q_scale, max_norm=None)
    grad_raw = qf.raw_norm.mean().item()
    # finite-diff SNR: dQ for a unit-rms action perturbation of size eps
    eps = 0.05
    u = torch.randn_like(cur); u = u / u.reshape(B * K, -1).norm(dim=-1, keepdim=True).view(-1, 1, 1) * (eps * (S ** 0.5))
    with torch.no_grad():
        dq = (critic(state_K, z, cur + u) - critic(state_K, z, cur)).abs().mean().item()
    print(f"\n[1] CRITIC ACTION-SENSITIVITY")
    print(f"    ||grad_a Q|| (raw, per-particle mean) = {grad_raw:.4e}")
    print(f"    |dQ| for eps={eps} rms action step     = {dq:.4e}   (|Q|mean={q_scale.item():.3f})")
    print(f"    -> grad channel is {'DEAD (flat critic)' if grad_raw < 1e-2 else 'alive'}")

    # ---- 2. Q-spread across candidates (zeroth signal) ----
    q_b = q_cur.reshape(B, K)
    q_std = q_b.std(dim=1).mean().item()
    q_range = (q_b.max(1).values - q_b.min(1).values).mean().item()
    print(f"\n[2] Q-SPREAD across K candidates (zeroth needs this)")
    print(f"    std(Q)/|Q|    = {q_std / q_scale.item():.4f}   (std={q_std:.4e})")
    print(f"    (Qmax-Qmin)   = {q_range:.4e}")
    print(f"    -> zeroth signal is {'WEAK (candidates ~ same Q)' if q_std / q_scale.item() < 0.05 else 'usable'}")

    # ---- 3. candidate dispersion ----
    disp = cur.reshape(B, K, S).std(dim=1).mean().item()
    base_disp = base.reshape(B, K, S).std(dim=1).mean().item()
    print(f"\n[3] CANDIDATE DISPERSION (rms across K)")
    print(f"    base+res particles = {disp:.4f}   (base alone = {base_disp:.4f})")

    # ---- 4. field -> reward alignment (does stepping along the field raise Q?) ----
    print(f"\n[4] FIELD -> REWARD (step along field, does REAL Q rise?)")
    for qs in ("grad", "zeroth"):
        if qs == "grad":
            f = compute_q_gradient_field(critic, state_K, z, cur, q_scale=q_scale, max_norm=1.0).field
        else:
            cur_b = cur.reshape(B, K, S); qb = q_cur.reshape(B, K)
            top_idx = qb.topk(4, dim=1).indices
            pos = torch.gather(cur_b, 1, top_idx.unsqueeze(-1).expand(-1, -1, S))
            out = compute_drift_field(cur_b, pos, cur_b, radii=(0.02, 0.05, 0.2),
                                      exclude_negative_self=True, normalize_per_radius=False)
            f = clip_field_norm(out.field.reshape(B * K, 1, S), 1.0).reshape(B * K, H, A)
        with torch.no_grad():
            for step_sz in (0.05, 0.2):
                dQ = (critic(state_K, z, cur + step_sz * f) - q_cur).mean().item()
                print(f"    {qs:6s} step={step_sz}:  dQ = {dQ:+.4e}   ||field||={f.reshape(B*K,-1).norm(dim=-1).mean():.3f}")

    # ---- 5. in-loop: 150 field updates vs REAL critic ----
    print(f"\n[5] IN-LOOP field_actor_loss (real critic, 150 steps) — does Q climb & residual grow?")
    init_actor = {k: v.clone() for k, v in student.actor.state_dict().items()}

    @torch.no_grad()
    def measure():
        a = student._teacher_fn(state_K, z) + student.actor(state_K, z)
        return critic(state_K, z, a).mean().item(), student.actor(state_K, z).pow(2).mean().sqrt().item()

    for qs in ("zeroth", "grad"):
        student.actor.load_state_dict(init_actor)
        opt = torch.optim.Adam(student.actor.parameters(), lr=1e-3)
        q0, r0 = measure()
        for step in range(150):
            m = student.field_actor_loss(cond, mode="field_distributional", q_source=qs,
                                         q_step=0.2, bc_step=0.05, q_max_norm=1.0,
                                         total_max_norm=0.2, num_particles=K, num_bc_particles=K, topk=4)
            opt.zero_grad(set_to_none=True); m["actor_total"].backward(); opt.step()
        q1, r1 = measure()
        print(f"    {qs:6s}:  Q {q0:+.4f} -> {q1:+.4f}  (dQ={q1-q0:+.4f})   residual {r0:.4f} -> {r1:.4f}")

    print("\n" + "=" * 68)
    print("READ: [1] dead grad + [2] weak Q-spread => neither channel has a reward")
    print("      direction on THIS drift critic. [4]/[5] confirm whether ANY step helps.")
    print("=" * 68)


if __name__ == "__main__":
    main()
