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


# ---------------------------------------------------------------------------
# P0.2 — router telemetry must cover ALL loops, not just the last
# ---------------------------------------------------------------------------


def test_router_telemetry_covers_all_loops():
    from mythouro.training_utils import collect_router_logits, collect_expert_counts

    K = 4
    m = MythOuro(_cfg(max_loop_iters=K)).train()   # train mode → no early exit
    topk = m.cfg.n_experts_per_tok
    m(torch.randint(0, 128, (2, 5)), n_loops=K)

    logits = collect_router_logits(m)
    assert len(logits) == K, (
        f"expected one router-logits tensor per loop ({K}), got {len(logits)} "
        "— telemetry is still capturing only the last loop (P0.2)"
    )
    rows_per_loop = logits[0].shape[0]

    counts = collect_expert_counts(m)
    total = float(next(iter(counts.values())).sum().item())
    assert total == K * rows_per_loop * topk, (total, K, rows_per_loop, topk)


def test_aux_loss_grad_live_under_gradient_checkpointing():
    # The fragile path the review flagged: telemetry stashed inside the
    # checkpointed _loop_body. After the P0.2 fix it flows through the
    # checkpoint return, so aux-loss gradient reaches the router weights.
    from mythouro.training_utils import (
        collect_router_logits, load_balance_loss, sparse_activation_loss,
    )

    m = MythOuro(_cfg(gradient_checkpointing=True, max_loop_iters=3)).train()
    logits, _ = m(torch.randint(0, 128, (2, 6)), n_loops=3)
    rbuf = collect_router_logits(m)
    loss = (
        logits.float().mean()
        + load_balance_loss(rbuf, topk=m.cfg.n_experts_per_tok)
        + sparse_activation_loss(rbuf)
    )
    loss.backward()
    g = m.recurrent.block.ffn.router.weight.grad
    assert g is not None and torch.isfinite(g).all() and float(g.abs().sum()) > 0


# ---------------------------------------------------------------------------
# P0.3 — eval emits h_K (same path training optimized), not the h_out blend
# ---------------------------------------------------------------------------


def test_train_eval_emission_parity():
    # With ACT early-exit disabled (threshold unreachable, convergence off) all
    # loops run in both modes, and both now return the final loop state h_K — so
    # train-mode and eval-mode forwards must produce identical logits (dropout
    # is 0). Before P0.3, eval returned the under-summed h_out → they'd differ.
    m = MythOuro(_cfg(
        act_threshold=1e9, convergence_eps=0.0, gradient_checkpointing=False,
    ))
    x = torch.randint(0, 128, (2, 6))
    with torch.no_grad():
        m.train()
        lt, _ = m(x, n_loops=4)
        m.eval()
        le, _ = m(x, n_loops=4)
    assert torch.allclose(lt, le, atol=1e-5), (lt - le).abs().max().item()
