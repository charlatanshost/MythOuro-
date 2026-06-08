"""
Tests for the MoE-vs-dense recurrent-FFN ablation (`recurrent_dense`).

Pins the contract the ablation depends on:
  1. the flag actually swaps MoEFFN -> a dense Expert in the recurrent block,
  2. the dense FFN is sized so its params == the MoE arm's *activated* FFN per
     token (the matched-compute invariant — the whole point of the comparison),
  3. the dense model trains (forward + backward), and
  4. the MoE-only aux helpers no-op on a dense model, so the existing training
     scripts run the dense arm unchanged (the MoE loss terms vanish to 0).

See docs/roadmap.md "Gating experiment: MoE-vs-dense ablation".
"""

from __future__ import annotations

from dataclasses import asdict, replace

import torch
import torch.nn.functional as F

from mythouro.main import MythOuro, MythOuroConfig, Expert, MoEFFN
from mythouro.variants import mythouro_distill_tiny, mythouro_distill_tiny_dense
from mythouro.training_utils import (
    collect_router_logits,
    load_balance_loss,
    sparse_activation_loss,
    collect_expert_counts,
    update_router_bias_from_counts,
)


def _cfg(**overrides) -> MythOuroConfig:
    defaults = dict(
        vocab_size=128, dim=64, n_heads=4, n_kv_heads=2, max_seq_len=64,
        max_loop_iters=3, prelude_layers=1, coda_layers=1, attn_type="gqa",
        n_experts=4, n_shared_experts=1, n_experts_per_tok=2, expert_dim=16,
        lora_rank=4, kv_lora_rank=16, q_lora_rank=16,
        qk_rope_head_dim=8, qk_nope_head_dim=8, v_head_dim=8, dropout=0.0,
    )
    defaults.update(overrides)
    return MythOuroConfig(**defaults)


# ---------------------------------------------------------------------------
# Flag wiring
# ---------------------------------------------------------------------------


def test_moe_arm_uses_moeffn():
    model = MythOuro(_cfg(recurrent_dense=False)).eval()
    assert isinstance(model.recurrent.block.ffn, MoEFFN)


def test_dense_arm_uses_expert():
    model = MythOuro(_cfg(recurrent_dense=True)).eval()
    ffn = model.recurrent.block.ffn
    assert isinstance(ffn, Expert)
    assert not isinstance(ffn, MoEFFN)


def test_explicit_dense_width_is_respected():
    model = MythOuro(_cfg(recurrent_dense=True, recurrent_dense_ffn_dim=40))
    # gate: Linear(dim, d_ff) -> out_features is the inner width
    assert model.recurrent.block.ffn.gate.out_features == 40


# ---------------------------------------------------------------------------
# The matched-compute invariant
# ---------------------------------------------------------------------------


def test_dense_params_equal_moe_active_params():
    base = _cfg()
    moe = MythOuro(replace(base, recurrent_dense=False))
    dense = MythOuro(replace(base, recurrent_dense=True))  # auto width

    moe_ffn = moe.recurrent.block.ffn
    per_routed = sum(p.numel() for p in moe_ffn.routed_experts[0].parameters())
    per_shared = sum(p.numel() for p in moe_ffn.shared_experts[0].parameters())
    moe_active = moe_ffn.topk * per_routed + moe_ffn.n_shared * per_shared

    dense_params = sum(p.numel() for p in dense.recurrent.block.ffn.parameters())

    # The core ablation guarantee: same FLOPs/params per token.
    assert dense_params == moe_active
    # And the auto width matches the documented formula.
    expected = base.expert_dim * base.n_experts_per_tok * (1 + base.n_shared_experts)
    assert dense.recurrent.block.ffn.gate.out_features == expected


def test_dense_has_fewer_total_params_than_moe():
    base = _cfg()
    moe = sum(p.numel() for p in MythOuro(replace(base, recurrent_dense=False)).parameters())
    dense = sum(p.numel() for p in MythOuro(replace(base, recurrent_dense=True)).parameters())
    # Dense drops the idle routed experts -> strictly smaller total.
    assert dense < moe


# ---------------------------------------------------------------------------
# Trains
# ---------------------------------------------------------------------------


def test_dense_forward_and_backward():
    model = MythOuro(_cfg(recurrent_dense=True)).train()
    x = torch.randint(0, 128, (2, 6))
    logits, unc = model(x)
    assert logits.shape == (2, 6, 128)
    assert unc.shape == (2, 6)
    loss = F.cross_entropy(logits.reshape(-1, 128), x.reshape(-1))
    loss.backward()
    # The dense recurrent FFN actually received gradient.
    assert model.recurrent.block.ffn.gate.weight.grad is not None


# ---------------------------------------------------------------------------
# MoE-only aux helpers no-op on a dense model (training scripts run unchanged)
# ---------------------------------------------------------------------------


def test_moe_aux_helpers_noop_on_dense():
    model = MythOuro(_cfg(recurrent_dense=True)).train()
    model(torch.randint(0, 128, (2, 6)))  # populate any stashed state

    assert collect_router_logits(model) == []
    assert collect_expert_counts(model) == {}
    assert float(load_balance_loss([], topk=2).item()) == 0.0
    assert float(sparse_activation_loss([]).item()) == 0.0
    # Empty counts -> no-op, no crash.
    update_router_bias_from_counts(model, {}, bias_lr=1e-3, ddp=False)


# ---------------------------------------------------------------------------
# The shipped variant: identical to distill_tiny except the FFN
# ---------------------------------------------------------------------------


def test_distill_tiny_dense_differs_only_in_ffn_fields():
    tiny = asdict(mythouro_distill_tiny())
    dense = asdict(mythouro_distill_tiny_dense())
    diff = {k for k in tiny if tiny[k] != dense[k]}
    assert diff == {"recurrent_dense", "recurrent_dense_ffn_dim"}
    assert dense["recurrent_dense"] is True
    # 1280 * 4 * (1 + 2) = 15360 — matched-active width for distill_tiny.
    assert dense["recurrent_dense_ffn_dim"] == 15360
