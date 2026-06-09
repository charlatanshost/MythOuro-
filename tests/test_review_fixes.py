"""
Invariant tests for the external code-review fixes (P0.1, P0.2, P0.3).

These are *invariant* tests (not logic tests) — the kind the 303-test suite
lacked, which let P0.1 and P0.2 slip through despite "all green". See
docs/roadmap.md failure modes / the review findings.
"""

from __future__ import annotations

import torch

from mythouro.main import MythOuro, MythOuroConfig, CrossLoopAttention


def _cfg(**kw) -> MythOuroConfig:
    d = dict(
        vocab_size=128, dim=64, n_heads=4, n_kv_heads=2, max_seq_len=64,
        max_loop_iters=4, prelude_layers=1, coda_layers=1, attn_type="gqa",
        n_experts=4, n_shared_experts=1, n_experts_per_tok=2, expert_dim=16,
        lora_rank=4, kv_lora_rank=16, q_lora_rank=16,
        qk_rope_head_dim=8, qk_nope_head_dim=8, v_head_dim=8, dropout=0.0,
        use_cross_loop_attention=True,
    )
    d.update(kw)
    return MythOuroConfig(**d)


# ---------------------------------------------------------------------------
# P0.1 — deliberate zero-inits must survive _init_weights
# ---------------------------------------------------------------------------


def test_zero_inits_survive_construction():
    m = MythOuro(_cfg())

    # CrossLoopAttention.o_proj — zero so it starts as an identity residual.
    found_cross = False
    for mod in m.modules():
        if isinstance(mod, CrossLoopAttention):
            found_cross = True
            assert float(mod.o_proj.weight.detach().abs().sum()) == 0.0, (
                "cross-loop o_proj zero-init was clobbered by _init_weights"
            )
    assert found_cross, "cross-loop attention not present in test model"

    # UncertaintyHead output layer — zero weight + bias.
    assert float(m.uncertainty.net[-1].weight.detach().abs().sum()) == 0.0
    assert float(m.uncertainty.net[-1].bias.detach().abs().sum()) == 0.0


def test_uncertainty_head_starts_neutral_half():
    # With the output layer zero, the head outputs sigmoid(0) = 0.5 for ANY
    # input — the calibrated-neutral starting point its docstring claims.
    m = MythOuro(_cfg()).eval()
    h = torch.randn(2, 5, m.cfg.dim)
    u = m.uncertainty(m.norm(h))
    assert torch.allclose(u, torch.full_like(u, 0.5), atol=1e-6), (
        u.min().item(), u.max().item()
    )
