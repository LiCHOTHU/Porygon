"""Unit tests for imitation/algos/dice/drift_field.py (CPU, no model).

Run:  python scripts/test_drift_field.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from imitation.algos.dice.drift_field import (
    compute_drift_field, clip_field_norm, aggregate_q, R_LIST_DEFAULT)


def _mk(B=3, K=16, H=8, A=7, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(B, K, H, A, generator=g)


def test_p_eq_n_zero():
    """positive == negative == query, self masked on BOTH blocks => V ~= 0.
    Catches the self-term asymmetry: masking only one block leaves a residual field."""
    q = _mk()
    out = compute_drift_field(q, q.clone(), q.clone(),
                              exclude_negative_self=True, exclude_positive_self=True)
    rel = out.field.norm(dim=-1).mean().item() / q.norm(dim=-1).mean().item()
    assert rel < 0.02, f"P+=P-=query field not ~0: rel={rel:.4f}"
    return f"rel field mag = {rel:.5f}"


def test_distributional_fixedpoint():
    """REAL Test-3: current==base distributionally (independent noise). positive =
    independent same-distribution sample, negative = query. V_BC should be small."""
    q = _mk(seed=10)
    pos = _mk(seed=11)                 # same distribution, independent particles
    # raw restoring field (normalize_per_radius=False) -> shrinks as distributions match
    out = compute_drift_field(q, pos, q.clone(), exclude_negative_self=True,
                              normalize_per_radius=False)
    rel = out.field.norm(dim=-1).mean().item() / q.norm(dim=-1).mean().item()
    assert rel < 0.6, f"raw fixed-point field too large: rel={rel:.4f}"
    return f"rel field mag = {rel:.4f} (raw restoring, same-dist)"


def test_translation_equivariance():
    """shifting query+pos+neg by a constant leaves the field unchanged."""
    q = _mk(); pos = _mk(seed=1); neg = q.clone()
    c = torch.randn(1, 1, 8, 7) * 0.3
    f0 = compute_drift_field(q, pos, neg).field
    f1 = compute_drift_field(q + c, pos + c, neg + c).field
    d = (f0 - f1).abs().max().item()
    assert d < 1e-4, f"not translation-equivariant: max diff {d:.2e}"
    return f"max diff = {d:.2e}"


def test_permutation_equivariance():
    """permuting the query particles permutes the field rows identically.
    (independent neg => no self-alignment, exclude_negative_self=False)."""
    q = _mk(); pos = _mk(seed=1); neg = _mk(seed=2)
    perm = torch.randperm(q.shape[1])
    f0 = compute_drift_field(q, pos, neg, exclude_negative_self=False).field[:, perm]
    f1 = compute_drift_field(q[:, perm], pos, neg, exclude_negative_self=False).field
    d = (f0 - f1).abs().max().item()
    assert d < 1e-4, f"not permutation-equivariant: max diff {d:.2e}"
    return f"max diff = {d:.2e}"


def test_perturbation_recovery():
    """query = bc + delta, positive = bc, negative = query.
    The BC field should OPPOSE the imposed shift: mean_i V_i . delta < 0."""
    bc = _mk(seed=3)
    delta = torch.randn(1, 1, 8, 7) * 0.15
    q = bc + delta
    out = compute_drift_field(q, bc, q.clone(), normalize_per_radius=False)
    Vflat = out.field.reshape(out.field.shape[0], out.field.shape[1], -1)
    dflat = delta.reshape(1, 1, -1)
    proj = (Vflat * dflat).sum(-1).mean().item()
    assert proj < 0, f"BC field does not oppose the shift: proj={proj:.4f}"
    return f"mean V.delta = {proj:.4f} (<0 good)"


def test_no_grad():
    """field construction must not leak gradients to the query."""
    q = _mk().requires_grad_(True); pos = _mk(seed=1); neg = q.detach().clone()
    out = compute_drift_field(q, pos, neg)
    assert not out.field.requires_grad, "field should be detached (no_grad)"
    return "field.requires_grad = False"


def test_clip_and_aggregate():
    f = torch.randn(2, 4, 56) * 5.0
    c = clip_field_norm(f, max_norm=1.0)
    mx = c.norm(dim=-1).max().item()
    assert mx <= 1.0 + 1e-5, f"clip failed: max norm {mx}"
    qh = [torch.tensor([[0.2], [0.5]]), torch.tensor([[0.1], [0.9]])]
    amin = aggregate_q(qh, "min")
    assert torch.allclose(amin, torch.tensor([[0.1], [0.5]])), amin
    return f"clip max={mx:.3f}, agg-min ok"


if __name__ == "__main__":
    tests = [test_p_eq_n_zero, test_distributional_fixedpoint,
             test_translation_equivariance,
             test_permutation_equivariance, test_perturbation_recovery,
             test_no_grad, test_clip_and_aggregate]
    fails = 0
    for t in tests:
        try:
            msg = t()
            print(f"  PASS  {t.__name__:32s} {msg}")
        except AssertionError as e:
            fails += 1
            print(f"  FAIL  {t.__name__:32s} {e}")
    print(f"\n{'ALL PASS' if fails == 0 else str(fails) + ' FAILED'}")
    sys.exit(1 if fails else 0)
