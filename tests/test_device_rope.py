"""
Tests for the CUDA/XPU/CPU device abstraction and the real-valued RoPE fallback.

The load-bearing guarantee: the real (cos/sin) RoPE path computes the **same**
rotation as the default complex path, so flipping `rope_real` (e.g. for Intel XPU,
which may lack complex-tensor ops) is safe on a checkpoint trained either way.
"""

from __future__ import annotations

import torch

from mythouro import device as dev
from mythouro.main import (
    MythOuro,
    MythOuroConfig,
    apply_rope,
    precompute_rope_freqs,
)


# ---------------------------------------------------------------------------
# Device abstraction
# ---------------------------------------------------------------------------


def test_backend_and_autocast_mapping():
    assert dev.backend("cuda:0") == "cuda"
    assert dev.backend("cuda") == "cuda"
    assert dev.backend("xpu") == "xpu"
    assert dev.backend("xpu:1") == "xpu"
    assert dev.backend("cpu") == "cpu"
    # autocast device_type mirrors the backend name
    assert dev.autocast_type("cuda:0") == "cuda"
    assert dev.autocast_type("xpu") == "xpu"
    assert dev.autocast_type("cpu") == "cpu"


def test_pick_device_honours_explicit_and_falls_back():
    assert dev.pick_device("cpu") == "cpu"
    assert dev.pick_device("xpu") == "xpu"
    # No preference → some valid backend (cuda:0 / xpu / cpu depending on host)
    assert dev.pick_device(None) in ("cuda:0", "xpu", "cpu")


def test_cpu_helpers_are_safe_noops():
    assert dev.is_accelerator("cpu") is False
    assert dev.is_accelerator("cuda:0") is True
    assert dev.is_accelerator("xpu") is True
    assert dev.bf16_supported("cpu") is False
    assert dev.fused_adam_supported("cpu") is False
    assert dev.fused_adam_supported("xpu") is False     # fused AdamW is CUDA-only
    assert dev.fused_adam_supported("cuda:0") is True
    dev.synchronize("cpu")                               # must not raise


# ---------------------------------------------------------------------------
# RoPE: real path == complex path
# ---------------------------------------------------------------------------


def test_precompute_shapes():
    dim, T = 16, 8
    fc = precompute_rope_freqs(dim, T, real=False)
    fr = precompute_rope_freqs(dim, T, real=True)
    assert fc.shape == (T, dim // 2) and fc.is_complex()
    assert fr.shape == (T, dim // 2, 2) and not fr.is_complex()


def test_apply_rope_real_matches_complex():
    dim, T, H, B = 16, 8, 4, 2
    x = torch.randn(B, T, H, dim)
    fc = precompute_rope_freqs(dim, T, real=False)[:T]
    fr = precompute_rope_freqs(dim, T, real=True)[:T]
    out_c = apply_rope(x, fc)
    out_r = apply_rope(x, fr)
    assert out_c.shape == x.shape
    assert torch.allclose(out_c, out_r, atol=1e-5), (
        (out_c - out_r).abs().max().item()
    )


def _tiny(**kw) -> dict:
    d = dict(
        vocab_size=128, dim=64, n_heads=4, n_kv_heads=2, max_seq_len=64,
        max_loop_iters=2, prelude_layers=1, coda_layers=1, attn_type="gqa",
        n_experts=4, n_shared_experts=1, n_experts_per_tok=2, expert_dim=16,
        lora_rank=4, kv_lora_rank=16, q_lora_rank=16,
        qk_rope_head_dim=8, qk_nope_head_dim=8, v_head_dim=8, dropout=0.0,
    )
    d.update(kw)
    return d


def test_model_forward_rope_real_matches_complex():
    # Same seed → identical weights; only the RoPE table differs (complex vs
    # real). Outputs must match within fp tolerance.
    torch.manual_seed(0)
    m_c = MythOuro(MythOuroConfig(**_tiny(rope_real=False))).eval()
    torch.manual_seed(0)
    m_r = MythOuro(MythOuroConfig(**_tiny(rope_real=True))).eval()

    x = torch.randint(0, 128, (1, 8))
    with torch.no_grad():
        lc, _ = m_c(x)
        lr, _ = m_r(x)
    assert torch.allclose(lc, lr, atol=1e-4), (lc - lr).abs().max().item()


def test_model_forward_rope_real_mla():
    # The MLA attention path also routes through apply_rope — check the real
    # table works there too.
    torch.manual_seed(0)
    m = MythOuro(MythOuroConfig(**_tiny(attn_type="mla", rope_real=True))).eval()
    x = torch.randint(0, 128, (1, 6))
    with torch.no_grad():
        logits, _ = m(x)
    assert logits.shape == (1, 6, 128)
    assert torch.isfinite(logits.float()).all()
