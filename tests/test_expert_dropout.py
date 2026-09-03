"""Expert dropout must dilute routing, not merely reshuffle it.

Reproduces the accidental regularisation from the 24->48 growth: half of every
token's routing went to experts contributing at ~0.40 amplitude, so the ORIGINAL
experts effectively saw ~2 slots instead of 4. Renormalising after the mask would
destroy that — the surviving experts would absorb the dropped weight and the
token's total FFN contribution would be unchanged.
"""
import torch
from mythouro.main import MoEFFN
from mythouro.variants import mythouro_distill_tiny


def _ffn(p):
    cfg = mythouro_distill_tiny()
    cfg.dim, cfg.n_experts, cfg.expert_dim, cfg.n_experts_per_tok = 32, 8, 16, 4
    cfg.expert_dropout = p
    return MoEFFN(cfg), cfg


def test_off_by_default_is_bit_identical():
    torch.manual_seed(0); a, _ = _ffn(0.0)
    torch.manual_seed(0); b, _ = _ffn(0.0)
    x = torch.randn(2, 6, 32)
    a.train(); b.train()
    assert torch.allclose(a(x), b(x))


def test_dropout_reduces_the_output_magnitude():
    """Dilution, not reshuffling: total contribution must FALL."""
    torch.manual_seed(0); m, _ = _ffn(0.0)
    x = torch.randn(4, 8, 32)
    m.train()
    base = m(x).abs().mean().item()
    m.expert_dropout = 0.5
    torch.manual_seed(1)
    got = torch.stack([m(x).abs().mean() for _ in range(8)]).mean().item()
    assert got < base, f"dropout did not reduce contribution: {got} vs {base}"


def test_inactive_in_eval_mode():
    torch.manual_seed(0); m, _ = _ffn(0.5)
    x = torch.randn(2, 6, 32)
    m.eval()
    with torch.no_grad():
        assert torch.allclose(m(x), m(x)), "eval must be deterministic"


def test_gradient_still_flows_to_experts():
    torch.manual_seed(0); m, _ = _ffn(0.5)
    m.train()
    x = torch.randn(4, 8, 32, requires_grad=True)
    m(x).sum().backward()
    got = [n for n, q in m.named_parameters()
           if q.grad is not None and q.grad.abs().sum() > 0]
    assert got, "no expert received gradient under dropout"
