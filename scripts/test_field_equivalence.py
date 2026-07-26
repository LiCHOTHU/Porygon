"""Pre-rollout gate: field-regression must reproduce the DICE actor gradient
DIRECTION (magnitude differs by the MSE 2*eta factor — only direction matters).

Test 1: V_Q = grad_a Q  vs  direct -Q.
Test 2: V_BC = -r        vs  lambda*||r||^2.

Uses a random-init critic (nonzero grad_a Q, unlike the trained flat one) and a
mock teacher — this validates the FIELD PLUMBING, independent of whether the real
critic has signal. Run: python scripts/test_field_equivalence.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from imitation.algos.dice.distill_rl import DistilledRLModel
from imitation.algos.dice.drift_field import compute_q_gradient_field


def _cos(gs_a, gs_b):
    fa = torch.cat([g.reshape(-1) for g in gs_a])
    fb = torch.cat([g.reshape(-1) for g in gs_b])
    return torch.nn.functional.cosine_similarity(fa, fb, dim=0).item()


def _model(D, A, H):
    m = DistilledRLModel(state_dim=D, action_dim=A, horizon_steps=H,
                         actor_hidden=(64, 64), critic_hidden=(64, 64),
                         ensemble_size=2, conservative="min",
                         use_q_normalization=False, device="cpu").to("cpu")
    m.attach_teacher(lambda s, z: torch.zeros(s.shape[0], H, A))   # a0 = 0 (math test)
    return m


def test_q_field_equivalence():
    torch.manual_seed(0)
    B, D, A, H, q_step = 5, 12, 7, 4, 1e-3
    m = _model(D, A, H)
    state = torch.randn(B, D); z = torch.randn(B, H, A)

    # direct DICE-Q: L = -mean Q(s, a_theta)
    a = m.get_action(state, z)
    L_dice = -m.critic(state, z, a).mean()
    g_dice = torch.autograd.grad(L_dice, list(m.actor.parameters()), retain_graph=False)

    # field-Q: target = sg[r + q_step * grad_a Q]; L = ||r - target||^2
    old_res = m.actor(state, z).detach()
    qf = compute_q_gradient_field(m.critic, state, z, a.detach(), q_scale=1.0, max_norm=None)
    target = (old_res + q_step * qf.field).detach()
    pred = m.actor(state, z)
    L_field = ((pred - target) ** 2).mean()
    g_field = torch.autograd.grad(L_field, list(m.actor.parameters()))

    c = _cos(g_dice, g_field)
    assert c > 0.99, f"Q-field gradient direction mismatch: cos={c:.4f}"
    return f"cos(grad_DICE-Q, grad_field-Q) = {c:.5f}"


def test_bc_pointwise_equivalence():
    torch.manual_seed(1)
    B, D, A, H, bc_step = 5, 12, 7, 4, 1e-3
    m = _model(D, A, H)
    state = torch.randn(B, D); z = torch.randn(B, H, A)

    # direct BC anchor: L = ||r||^2  (a0=0 => residual = action; lambda absorbed in step)
    r = m.actor(state, z)
    L_bc = (r ** 2).mean()
    g_bc = torch.autograd.grad(L_bc, list(m.actor.parameters()))

    # field-BC pointwise: V_BC = -r ; target = sg[r - bc_step*r]; L = ||r - target||^2
    old_res = m.actor(state, z).detach()
    target = (old_res - bc_step * old_res).detach()
    pred = m.actor(state, z)
    L_field = ((pred - target) ** 2).mean()
    g_field = torch.autograd.grad(L_field, list(m.actor.parameters()))

    c = _cos(g_bc, g_field)
    assert c > 0.99, f"BC-pointwise gradient direction mismatch: cos={c:.4f}"
    return f"cos(grad_||r||^2, grad_field-BC) = {c:.5f}"


def test_no_critic_grad_leak():
    """compute_q_gradient_field must leave critic params with no grad and restore flags."""
    torch.manual_seed(2)
    B, D, A, H = 4, 12, 7, 4
    m = _model(D, A, H)
    state = torch.randn(B, D); z = torch.randn(B, H, A)
    a = m.get_action(state, z).detach()
    before = [p.requires_grad for p in m.critic.parameters()]
    _ = compute_q_gradient_field(m.critic, state, z, a, q_scale=1.0, max_norm=1.0)
    after = [p.requires_grad for p in m.critic.parameters()]
    assert before == after, "critic requires_grad flags not restored"
    assert all(p.grad is None or p.grad.abs().sum() == 0 for p in m.critic.parameters()), \
        "critic accumulated gradient from field construction"
    return "critic flags restored, no grad leak"


if __name__ == "__main__":
    tests = [test_q_field_equivalence, test_bc_pointwise_equivalence, test_no_critic_grad_leak]
    fails = 0
    for t in tests:
        try:
            print(f"  PASS  {t.__name__:30s} {t()}")
        except AssertionError as e:
            fails += 1; print(f"  FAIL  {t.__name__:30s} {e}")
    print(f"\n{'ALL PASS' if fails == 0 else str(fails) + ' FAILED'}")
    sys.exit(1 if fails else 0)
