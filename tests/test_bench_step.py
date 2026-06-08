"""Smoke test for tools/bench_step.run_benchmark."""

from __future__ import annotations

import torch

from mythouro.main import MythOuro, MythOuroConfig
from tools.bench_step import run_benchmark


def _tiny() -> MythOuroConfig:
    return MythOuroConfig(
        vocab_size=128, dim=64, n_heads=4, n_kv_heads=2, max_seq_len=64,
        max_loop_iters=3, prelude_layers=1, coda_layers=1, attn_type="gqa",
        n_experts=4, n_shared_experts=1, n_experts_per_tok=2, expert_dim=16,
        lora_rank=4, kv_lora_rank=16, q_lora_rank=16,
        qk_rope_head_dim=8, qk_nope_head_dim=8, v_head_dim=8, dropout=0.0,
    )


def test_benchmark_returns_positive_throughput():
    model = MythOuro(_tiny())
    r = run_benchmark(
        model, "cpu", batch=1, seq_len=16, steps=2, warmup=1,
        backward=True, dtype=torch.float32,
    )
    assert r["ms_per_step"] > 0
    assert r["tokens_per_s"] > 0
    assert r["n_loops"] == 3


def test_benchmark_forward_only():
    model = MythOuro(_tiny())
    r = run_benchmark(
        model, "cpu", batch=2, seq_len=16, steps=2, warmup=1,
        backward=False, dtype=torch.float32,
    )
    assert r["tokens_per_s"] > 0
