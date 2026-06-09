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


# ---------------------------------------------------------------------------
# P1.3 — sorted MoE dispatch must equal a naive per-token reference
# ---------------------------------------------------------------------------


def test_moe_sorted_dispatch_matches_naive_reference():
    import torch.nn.functional as F
    from mythouro.main import MoEFFN

    torch.manual_seed(0)
    cfg = _cfg(n_experts=6, n_experts_per_tok=3, expert_dim=16)
    ffn = MoEFFN(cfg).eval()
    # Non-zero router bias so the biased-topk path is exercised too.
    ffn.router_bias.uniform_(-0.5, 0.5)

    x = torch.randn(2, 5, cfg.dim)
    out = ffn(x)

    # Independent reference: recompute the routing decisions and apply each
    # expert token by token (no sorting, no batching).
    flat = x.view(-1, cfg.dim)
    logits = ffn.router(flat)
    scores = F.softmax(logits, dim=-1)
    _, topk_idx = (logits + ffn.router_bias).topk(ffn.topk, dim=-1)
    topk_scores = scores.gather(-1, topk_idx)
    topk_scores = topk_scores / topk_scores.sum(dim=-1, keepdim=True)

    ref = torch.zeros_like(flat)
    for i in range(flat.shape[0]):
        for k in range(ffn.topk):
            eid = int(topk_idx[i, k])
            ref[i] += ffn.routed_experts[eid](flat[i : i + 1])[0] * topk_scores[i, k]
    for shared in ffn.shared_experts:
        ref += shared(flat)

    assert torch.allclose(out.view(-1, cfg.dim), ref, atol=1e-5), (
        (out.view(-1, cfg.dim) - ref).abs().max().item()
    )


def test_moe_dispatch_gradient_flows_through_gates_and_experts():
    from mythouro.main import MoEFFN

    torch.manual_seed(1)
    ffn = MoEFFN(_cfg()).train()
    x = torch.randn(2, 4, 64, requires_grad=True)
    ffn(x).sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert ffn.router.weight.grad is not None          # via the gate scores
    assert any(
        p.grad is not None and float(p.grad.abs().sum()) > 0
        for e in ffn.routed_experts for p in e.parameters()
    )


# ---------------------------------------------------------------------------
# P1.4 — multi-scale injection hoist: precompute+blend == per-loop forward
# ---------------------------------------------------------------------------


def test_ms_injection_precompute_blend_matches_forward():
    from mythouro.main import MultiScaleInjection

    torch.manual_seed(0)
    ms = MultiScaleInjection(dim=64, max_loops=4, window_size=4).eval()
    e = torch.randn(2, 10, 64)
    views = ms.precompute(e)
    for t in range(4):
        assert torch.allclose(ms.blend_views(views, t), ms(e, t), atol=1e-6)


# ---------------------------------------------------------------------------
# P1.1 — MLA caches the COMPACT shared rope keys, and cached decode still
# matches the no-cache forward (the numerical contract that matters).
# ---------------------------------------------------------------------------


def test_mla_cache_stores_compact_rope_keys():
    m = MythOuro(_cfg(attn_type="mla")).eval()
    kv: dict = {}
    x = torch.randint(0, 128, (2, 6))
    m(x, n_loops=2, kv_cache=kv, start_pos=0)
    rope_entries = [v["k_rope"] for v in kv.values() if "k_rope" in v]
    assert rope_entries, "no MLA cache entries written"
    for t in rope_entries:
        # (B, S, 1, rope_dim) — shared across heads, NOT expanded to n_heads.
        assert t.shape[2] == 1, t.shape


def test_mla_cached_decode_matches_full_forward():
    # NOTE: multi-scale injection and cross-loop attention are turned OFF here
    # — both are position-context-dependent (window pooling / all-position loop
    # snapshots), so cached single-token decode legitimately differs from a
    # full-sequence forward when they're on (pre-existing Part-2 semantics,
    # independent of the cache layout this test pins).
    torch.manual_seed(0)
    m = MythOuro(_cfg(
        attn_type="mla", convergence_eps=0.0,
        use_multiscale_injection=False, use_cross_loop_attention=False,
    )).eval()
    ids = torch.randint(0, 128, (1, 7))

    # Reference: one full no-cache forward over the whole sequence.
    ref, _ = m(ids, n_loops=2)

    # Cached: prefill on the first 6 tokens, then decode the 7th.
    kv: dict = {}
    m(ids[:, :6], n_loops=2, kv_cache=kv, start_pos=0)
    step, _ = m(ids[:, 6:7], n_loops=2, kv_cache=kv, start_pos=6)

    assert torch.allclose(step[0, -1], ref[0, -1], atol=1e-4), (
        (step[0, -1] - ref[0, -1]).abs().max().item()
    )


# ---------------------------------------------------------------------------
# P1.6 — zero-copy KV-cache rewind in UncertaintyGatedGenerator
# ---------------------------------------------------------------------------


def test_structure_snapshot_is_a_correct_cache_rewind():
    # The rewind contract: attention layers REPLACE cache entries (cat -> new
    # tensor -> store), never mutating stored tensors in place — so a
    # structure-only snapshot (shared refs, no clones) must restore the exact
    # pre-step cache after a forward has grown the live cache.
    m = MythOuro(_cfg()).eval()
    kv: dict = {}
    prompt = torch.randint(0, 128, (1, 5))
    m(prompt, n_loops=2, kv_cache=kv, start_pos=0)             # prefill

    reference = {
        k: {kk: vv.clone() for kk, vv in v.items()} for k, v in kv.items()
    }
    snapshot = {k: dict(v) for k, v in kv.items()}             # zero-copy

    nxt = torch.randint(0, 128, (1, 1))
    m(nxt, n_loops=2, kv_cache=kv, start_pos=5)                # grows the cache

    # Live cache grew; snapshot still holds the exact pre-step tensors.
    for key, entry in reference.items():
        for kk, ref_t in entry.items():
            assert kv[key][kk].shape[1] > ref_t.shape[1]       # grew
            assert torch.equal(snapshot[key][kk], ref_t), (key, kk)


def test_uncertainty_gated_always_redo_path():
    from mythouro.inference import UncertaintyGatedGenerator

    m = MythOuro(_cfg()).eval()
    prompt = torch.randint(0, 128, (1, 6))
    gen = UncertaintyGatedGenerator(m, min_loops=2, max_loops=4, threshold=0.0)
    out = gen.generate(prompt, max_new_tokens=4)
    assert out.shape == (1, 10)
    assert torch.equal(out[:, :6], prompt)


# ---------------------------------------------------------------------------
# P0.4 — ContinuousDepthwiseBatcher with cross-loop attention and a shrinking
# active set: no ragged-buffer crash, and per-row outputs match a single-row
# reference forward (rows must never see another sequence's history).
# ---------------------------------------------------------------------------


class _ScheduledHalt(torch.nn.Module):
    """ACT stub emitting a fixed halt schedule: halt_loops[i] is the loop at
    which original row i emits p=1.0 (None = never halts). Tracks the loop
    index by call count; assumes callers pass active rows in original order
    (active_idx from nonzero() is sorted, so they do)."""

    def __init__(self, halt_loops: "list[int | None]"):
        super().__init__()
        self.halt_loops = halt_loops
        self.t = 0

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        B_act, T, _ = h.shape
        active = [
            i for i, hl in enumerate(self.halt_loops)
            if hl is None or hl >= self.t
        ]
        assert len(active) == B_act, (self.t, len(active), B_act)
        p = torch.zeros(B_act, T, device=h.device)
        for j, orig in enumerate(active):
            if self.halt_loops[orig] == self.t:
                p[j] = 1.0
        self.t += 1
        return p


def _batcher_cfg():
    return _cfg(
        max_loop_iters=4,
        cross_loop_store_every=1,   # snapshot every loop → stresses the buffer
        convergence_eps=0.0,        # disable convergence early-exit
        dropout=0.0,
    )


def test_batcher_cross_loop_shrinking_active_set_no_crash():
    from mythouro.inference import ContinuousDepthwiseBatcher

    torch.manual_seed(0)
    m = MythOuro(_batcher_cfg()).eval()
    # Rows halt at loops 0, 2, never → active set shrinks 3 → 2 → 2 → 1.
    m.recurrent.act = _ScheduledHalt([0, 2, None])

    x = torch.randint(0, 128, (3, 6))
    logits, unc = ContinuousDepthwiseBatcher(m).forward(x, n_loops=4)
    assert logits.shape == (3, 6, 128)
    assert unc.shape == (3, 6)
    assert torch.isfinite(logits.float()).all()


def test_batcher_rows_match_single_row_reference():
    from mythouro.inference import ContinuousDepthwiseBatcher

    torch.manual_seed(0)
    m = MythOuro(_batcher_cfg()).eval()
    x = torch.randint(0, 128, (3, 6))
    halt_loops: "list[int | None]" = [0, 2, None]

    m.recurrent.act = _ScheduledHalt(halt_loops)
    batched, _ = ContinuousDepthwiseBatcher(m).forward(x, n_loops=4)

    # Reference: each row alone through the plain forward (eval mode halts via
    # the halt-all break at the same loop the batcher froze it). Fresh stub per
    # run so the schedule's call counter restarts.
    for i in range(3):
        m.recurrent.act = _ScheduledHalt([halt_loops[i]])
        ref, _ = m(x[i : i + 1], n_loops=4)
        assert torch.allclose(batched[i : i + 1], ref, atol=1e-4), (
            f"row {i} (halt_loop={halt_loops[i]}): max diff "
            f"{(batched[i:i+1] - ref).abs().max().item()}"
        )


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
