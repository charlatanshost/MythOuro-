"""
Tests for the §1 attention-kernel cascade (FA2 → SDPA → manual).

What these pin
--------------
1. The CAPABILITIES probe enforces CC ≥ 8.0 for FA2 — Volta / Turing
   must NOT report `fa2_usable` even with flash-attn installed (this is
   the actual bug §1 was opened to fix).
2. `warn_once` deduplicates messages: hot attention layers would
   otherwise spam the log every forward pass.
3. The SDPA and manual paths produce numerically equivalent attention
   outputs for both GQAttention and MLAttention. Without this guarantee
   training behavior would depend on the deploy hardware.

We can't easily test the live FA2 path here (this dev box doesn't have
flash-attn installed), but the FA2 ↔ SDPA equivalence is enforced by
PyTorch — both kernels compute the same scaled dot-product. The high-
risk pair is SDPA ↔ manual, which is what we exercise.
"""

from __future__ import annotations

import torch
import pytest

from mythouro.main import (
    CAPABILITIES,
    GQAttention,
    MLAttention,
    MythOuroConfig,
    precompute_rope_freqs,
)


B, T = 2, 8


def _gqa_cfg(**overrides) -> MythOuroConfig:
    defaults = dict(
        vocab_size=200, dim=64, n_heads=4, n_kv_heads=2,
        max_seq_len=32, max_loop_iters=2,
        prelude_layers=1, coda_layers=1, attn_type="gqa",
        n_experts=4, n_shared_experts=1, n_experts_per_tok=2, expert_dim=16,
        lora_rank=4, kv_lora_rank=16, q_lora_rank=16,
        qk_rope_head_dim=8, qk_nope_head_dim=8, v_head_dim=8,
        dropout=0.0,  # determinism: dropout off so SDPA == manual exactly
    )
    defaults.update(overrides)
    return MythOuroConfig(**defaults)


def _mla_cfg(**overrides) -> MythOuroConfig:
    return _gqa_cfg(attn_type="mla", **overrides)


# ---------------------------------------------------------------------------
# CAPABILITIES probe
# ---------------------------------------------------------------------------


class TestCapabilities:
    def test_has_sdpa_on_modern_torch(self):
        # PyTorch ≥ 2.0 exposes F.scaled_dot_product_attention. The whole
        # test suite assumes a modern torch — if this fails the cascade
        # has lost its middle tier and would always fall to manual.
        assert CAPABILITIES.has_sdpa is True

    def test_fa2_requires_compute_capability_8(self, monkeypatch):
        # The whole point of §1: flash-attn imports cleanly on Volta
        # (V100, CC 7.0) but kernels crash. The capability gate must
        # refuse to use FA2 below CC 8.0 even when import_ok is True.
        monkeypatch.setattr(CAPABILITIES, "has_flash_attn_import", True)
        monkeypatch.setattr(CAPABILITIES, "cuda_cc", (7, 0))
        assert CAPABILITIES._fa2_usable() is False
        monkeypatch.setattr(CAPABILITIES, "cuda_cc", (7, 5))   # Turing
        assert CAPABILITIES._fa2_usable() is False
        monkeypatch.setattr(CAPABILITIES, "cuda_cc", (8, 0))   # Ampere
        assert CAPABILITIES._fa2_usable() is True
        monkeypatch.setattr(CAPABILITIES, "cuda_cc", (12, 0))  # Blackwell
        assert CAPABILITIES._fa2_usable() is True

    def test_fa2_requires_import(self, monkeypatch):
        monkeypatch.setattr(CAPABILITIES, "has_flash_attn_import", False)
        monkeypatch.setattr(CAPABILITIES, "cuda_cc", (12, 0))
        assert CAPABILITIES._fa2_usable() is False

    def test_fa2_needs_cuda(self, monkeypatch):
        monkeypatch.setattr(CAPABILITIES, "has_flash_attn_import", True)
        monkeypatch.setattr(CAPABILITIES, "cuda_cc", None)
        assert CAPABILITIES._fa2_usable() is False

    def test_warn_once_emits_once(self, monkeypatch):
        CAPABILITIES.reset_warnings()
        events: list[str] = []
        # Patch loguru.logger.warning to capture calls without spamming
        # the test output. We patch on the module the cascade actually
        # uses, not the global loguru.logger.
        from mythouro import main as om_main
        monkeypatch.setattr(om_main.logger, "warning",
                             lambda msg: events.append(msg))
        CAPABILITIES.warn_once("xyz", "first")
        CAPABILITIES.warn_once("xyz", "second")
        CAPABILITIES.warn_once("xyz", "third")
        assert events == ["first"]
        CAPABILITIES.warn_once("other", "different key fires once")
        assert events == ["first", "different key fires once"]


# ---------------------------------------------------------------------------
# GQA SDPA ↔ manual equivalence
# ---------------------------------------------------------------------------


class TestGQAFallbackEquivalence:
    """
    SDPA and manual paths must produce numerically equivalent outputs
    on the same input. This is what frees training to be hardware-
    portable: a checkpoint trained on Ampere with SDPA and resumed on
    a CPU box on the manual path must compute the same gradients.
    """

    def setup_method(self):
        self.cfg = _gqa_cfg()
        self.attn = GQAttention(self.cfg).eval()        # eval → no dropout RNG
        self.freqs = precompute_rope_freqs(
            self.cfg.dim // self.cfg.n_heads, self.cfg.max_seq_len,
        )[:T]
        self.x = torch.randn(B, T, self.cfg.dim)
        self.mask = torch.full((1, 1, T, T), float("-inf"))
        self.mask = torch.triu(self.mask, diagonal=1)

    @pytest.mark.parametrize("with_mask", [True, False])
    def test_sdpa_matches_manual(self, monkeypatch, with_mask):
        # Force the FA2 tier off so we compare SDPA against manual.
        monkeypatch.setattr(CAPABILITIES, "fa2_usable", False)

        # SDPA path
        monkeypatch.setattr(CAPABILITIES, "has_sdpa", True)
        with torch.no_grad():
            out_sdpa = self.attn(
                self.x, self.freqs,
                mask=self.mask if with_mask else None,
            )
        # Manual path
        monkeypatch.setattr(CAPABILITIES, "has_sdpa", False)
        with torch.no_grad():
            out_manual = self.attn(
                self.x, self.freqs,
                mask=self.mask if with_mask else None,
            )
        assert torch.allclose(out_sdpa, out_manual, atol=1e-5), (
            f"SDPA vs manual diverged: max diff = "
            f"{(out_sdpa - out_manual).abs().max().item():.2e}"
        )


# ---------------------------------------------------------------------------
# MLA SDPA ↔ manual equivalence
# ---------------------------------------------------------------------------


class TestMLAFallbackEquivalence:
    def setup_method(self):
        self.cfg = _mla_cfg()
        self.attn = MLAttention(self.cfg).eval()
        self.freqs = precompute_rope_freqs(
            self.cfg.qk_rope_head_dim, self.cfg.max_seq_len,
        )[:T]
        self.x = torch.randn(B, T, self.cfg.dim)
        self.mask = torch.full((1, 1, T, T), float("-inf"))
        self.mask = torch.triu(self.mask, diagonal=1)

    @pytest.mark.parametrize("with_mask", [True, False])
    def test_sdpa_matches_manual(self, monkeypatch, with_mask):
        monkeypatch.setattr(CAPABILITIES, "has_sdpa", True)
        with torch.no_grad():
            out_sdpa = self.attn(
                self.x, self.freqs,
                mask=self.mask if with_mask else None,
            )
        monkeypatch.setattr(CAPABILITIES, "has_sdpa", False)
        with torch.no_grad():
            out_manual = self.attn(
                self.x, self.freqs,
                mask=self.mask if with_mask else None,
            )
        assert torch.allclose(out_sdpa, out_manual, atol=1e-5), (
            f"MLA SDPA vs manual diverged: max diff = "
            f"{(out_sdpa - out_manual).abs().max().item():.2e}"
        )
