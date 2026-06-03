"""Offline smoke test for the lambertae-port PolicyDrifting.

Verifies:
  (1) _drift_loss: shape OK, finite, nonzero loss, grad on gen, info populated
  (2) Conditional model + drift loss learns cond -> own_pos (BC use case)
  (3) Many-step training: loss stays finite, no collapse
  (4) Dry-run at LIBERO shapes (B=32, G=4, chunk=16, A=7)
"""

import sys
sys.path.insert(0, "/storage/home/hcoda1/8/lwang831/workspace/imitation")

import torch
import torch.nn as nn

from imitation.algos.policy_drifting import PolicyDrifting


class _Bag:
    pass


def _bind(obj, **kw):
    obj.drift_R_list = tuple(kw.get("drift_R_list", (0.02, 0.05, 0.2)))
    obj.drift_t = float(kw.get("drift_t", 0.0))
    obj.drift_num_gen = int(kw.get("drift_num_gen", 4))
    obj.action_clamp = bool(kw.get("action_clamp", True))
    obj._batched_cdist = PolicyDrifting._batched_cdist
    obj._drift_loss = PolicyDrifting._drift_loss.__get__(obj)


def test_drift_loss_basic():
    print("[1] _drift_loss basic: shape, finite, grad ...")
    obj = _Bag(); _bind(obj)
    B, G, S = 16, 4, 16 * 7
    gen = torch.randn(B, G, S, requires_grad=True)
    pos = torch.randn(B, 1, S)
    loss, info = obj._drift_loss(gen, pos)
    assert torch.isfinite(loss) and loss.item() > 1e-6, loss
    loss.backward()
    g = gen.grad
    assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0
    print(f"    OK  loss={loss.item():.4f}  drift_scale={info['drift_scale']:.4f}  "
          f"force_norm={info['drift_force_norm']:.4f}  grad.rms={g.pow(2).mean().sqrt():.4e}")


def test_drift_conditional_learning():
    """A conditioned model (cond -> G generator outputs per state) trained with
    drift loss should learn to produce gen_i close to pos_i."""
    print("[2] conditional model + drift loss learns cond -> own_pos ...")
    obj = _Bag(); _bind(obj)
    torch.manual_seed(0)
    N_STATES = 64
    G = 4
    S = 16 * 7
    COND = 32
    conds = torch.randn(N_STATES, COND)
    poss = torch.randn(N_STATES, S).clamp(-1, 1)
    model = nn.Sequential(nn.Linear(COND + S, 256), nn.GELU(),
                          nn.Linear(256, 256), nn.GELU(),
                          nn.Linear(256, S))
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    with torch.no_grad():
        noise = torch.randn(N_STATES, G, S)
        cond_g = conds.unsqueeze(1).expand(N_STATES, G, COND)
        gen0 = model(torch.cat([cond_g, noise], dim=-1)).mean(1)
        d_before = (gen0 - poss).pow(2).mean().sqrt().item()
    losses = []
    action_mses = []
    for step in range(2000):
        noise = torch.randn(N_STATES, G, S)
        cond_g = conds.unsqueeze(1).expand(N_STATES, G, COND)
        gen = model(torch.cat([cond_g, noise], dim=-1))   # (N, G, S)
        pos = poss.unsqueeze(1)                            # (N, 1, S)
        loss, _info = obj._drift_loss(gen, pos)
        # also track inferenced-action MSE (what we'll log per epoch)
        with torch.no_grad():
            action_mses.append(((gen.detach().mean(1) - poss) ** 2).mean().item())
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
    with torch.no_grad():
        noise = torch.randn(N_STATES, G, S)
        cond_g = conds.unsqueeze(1).expand(N_STATES, G, COND)
        gen1 = model(torch.cat([cond_g, noise], dim=-1)).mean(1)
        d_after = (gen1 - poss).pow(2).mean().sqrt().item()
    print(f"    rms(gen_mean - own_pos): {d_before:.4f}  ->  {d_after:.4f}")
    print(f"    drift loss[0]={losses[0]:.4f}  loss[500]={losses[500]:.4f}  loss[-1]={losses[-1]:.4f}")
    print(f"    action_mse[0]={action_mses[0]:.4f}  [500]={action_mses[500]:.4f}  [-1]={action_mses[-1]:.4f}")
    assert d_after < d_before * 0.5, f"failed to learn: {d_before} -> {d_after}"
    assert action_mses[-1] < action_mses[0] * 0.5, (
        f"action MSE did not decrease: {action_mses[0]} -> {action_mses[-1]}"
    )
    print("    OK")


def test_no_collapse_many_steps():
    print("[3] long training: drift loss stays finite, no collapse ...")
    obj = _Bag(); _bind(obj)
    torch.manual_seed(0)
    B, G, S, COND = 32, 4, 16 * 7, 64
    cond = torch.randn(B, COND)
    pos = torch.randn(B, 1, S)
    proj = nn.Sequential(nn.Linear(COND + S, 128), nn.GELU(), nn.Linear(128, S))
    opt = torch.optim.Adam(proj.parameters(), lr=3e-3)
    losses = []
    for step in range(500):
        noise = torch.randn(B, G, S)
        cond_g = cond.unsqueeze(1).expand(B, G, COND)
        gen = proj(torch.cat([cond_g, noise], dim=-1))
        loss, info = obj._drift_loss(gen, pos)
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
    assert all(torch.isfinite(torch.tensor(l)) for l in losses), "non-finite loss"
    n_zero = sum(1 for l in losses if l < 1e-8)
    print(f"    losses[0]={losses[0]:.4f}  losses[-1]={losses[-1]:.4f}  min={min(losses):.4f}  "
          f"#~0={n_zero}/{len(losses)}")
    assert n_zero == 0, "loss collapsed to 0"
    print("    OK")


def test_libero_shapes_dryrun():
    print("[4] dry-run at LIBERO shapes (B=32, G=4, chunk=16, A=7) ...")
    obj = _Bag(); _bind(obj)
    B, G, H, A = 32, 4, 16, 7
    S = H * A
    v = nn.Parameter(torch.randn(B * G, H, A) * 0.1)
    x0 = torch.randn(B * G, H, A)
    a_pred = (x0 + v).reshape(B, G, S)
    pos = torch.randn(B, H, A).clamp(-1, 1).reshape(B, 1, S)
    loss, info = obj._drift_loss(a_pred, pos)
    assert torch.isfinite(loss) and loss.item() > 1e-6, loss
    loss.backward()
    assert v.grad is not None and v.grad.abs().sum() > 0
    print(f"    OK  loss={loss.item():.4f}  drift_scale={info['drift_scale']:.4f}  "
          f"force_norm={info['drift_force_norm']:.4f}  v.grad.rms={v.grad.pow(2).mean().sqrt():.4e}")


if __name__ == "__main__":
    torch.manual_seed(42)
    test_drift_loss_basic()
    test_drift_conditional_learning()
    test_no_collapse_many_steps()
    test_libero_shapes_dryrun()
    print("\nAll smoke tests PASSED")
