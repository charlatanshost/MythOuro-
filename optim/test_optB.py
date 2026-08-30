"""OPT-B: the sparse top-K+tail loss must reduce to the dense loss at K=V."""
import sys, pathlib, torch
sys.path.insert(0, str(pathlib.Path(__file__).parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import pytest as _pt
_pt.importorskip("mythouro.sparse_kd", reason="OPT-B not applied (optim/apply.sh B)")
from mythouro.sparse_kd import sparse_distillation_loss
from mythouro.training_utils import distillation_loss

torch.manual_seed(0)
B, Tq, V = 2, 6, 64

def make(K, temperature, s, t):
    tv, ti = t.topk(K, dim=-1)
    if K == V:
        tail = torch.full((B, Tq), float("-inf"))
    else:
        mask = torch.zeros_like(t, dtype=torch.bool).scatter(-1, ti, True)
        tail = torch.where(mask, torch.full_like(t, -1e30), t / temperature)
        tail = torch.logsumexp(tail, dim=-1)
    return ti, tv, tail

def test_exact_at_full_K():
    for div in ("fwd_kl", "rev_kl", "jsd"):
        for temp in (1.0, 2.0):
            for targets in (None, torch.randint(0, V, (B, Tq))):
                s = torch.randn(B, Tq, V, requires_grad=True)
                t = torch.randn(B, Tq, V)
                dense, _ = distillation_loss(
                    s, t, targets=targets, temperature=temp,
                    alpha=0.5, divergence=div)
                ti, tv, tail = make(V, temp, s, t)
                sparse, _ = sparse_distillation_loss(
                    s, ti, tv, tail, targets=targets, temperature=temp,
                    alpha=0.5, divergence=div)
                d = abs(float(dense) - float(sparse))
                assert d < 2e-4, f"{div} T={temp} targets={targets is not None}: {d}"
    print("  ok: K=V reproduces dense loss for fwd_kl/rev_kl/jsd, T=1&2, ±targets")

def test_gradients_flow():
    s = torch.randn(B, Tq, V, requires_grad=True)
    t = torch.randn(B, Tq, V)
    ti, tv, tail = make(16, 2.0, s, t)
    loss, _ = sparse_distillation_loss(s, ti, tv, tail, temperature=2.0,
                                       divergence="rev_kl")
    loss.backward()
    assert s.grad is not None and torch.isfinite(s.grad).all()
    print("  ok: gradients finite through the sparse path")

def test_lower_bound_and_convergence():
    """Coarsening reduces KL -> sparse <= dense, and closes as K grows."""
    s = torch.randn(B, Tq, V)
    t = torch.randn(B, Tq, V) * 2.0
    dense, _ = distillation_loss(s, t, temperature=2.0, divergence="fwd_kl")
    prev = -1.0
    print(f"    dense (K=V={V}) = {float(dense):.6f}")
    for K in (1, 2, 4, 8, 16, 32, 64):
        ti, tv, tail = make(K, 2.0, s, t)
        sp, _ = sparse_distillation_loss(s, ti, tv, tail, temperature=2.0,
                                         divergence="fwd_kl")
        v = float(sp)
        assert v <= float(dense) + 1e-4, f"K={K}: {v} > dense {float(dense)}"
        assert v >= prev - 1e-6, f"K={K} not monotone: {v} < {prev}"
        prev = v
        print(f"      K={K:3d}  {v:.6f}   ({v/float(dense)*100:5.1f}% of dense)")
    print("  ok: sparse <= dense (data-processing bound), monotone in K")

def test_misalignment_rejected():
    s = torch.randn(B, Tq, V)
    t = torch.randn(B, Tq, V)
    ti, tv, tail = make(8, 2.0, s, t)
    try:
        sparse_distillation_loss(torch.randn(B, Tq + 1, V), ti, tv, tail)
    except ValueError:
        print("  ok: misaligned cache is refused, not silently broadcast")
        return
    raise AssertionError("misalignment guard missed")

for f in (test_exact_at_full_K, test_gradients_flow,
          test_lower_bound_and_convergence, test_misalignment_rejected):
    f()
print("\n  OPT-B loss: all tests pass")
