"""
Tests for batched + KV-cached on-policy rollouts (onpolicy_plan.md phase 5).

The load-bearing guarantees:
1. The cached student decode (`use_kv_cache=True`) produces the IDENTICAL
   token sequence to the legacy full-recompute path under greedy sampling
   — CPU fp32 exact. (XPU bf16 tolerance variant runs only when XPU exists.)
2. The teacher cached path is gated: it validates against the uncached path
   on first use and falls back (never crashes, never silently diverges).
3. `RolloutBuffer` enforces its reuse budget, wrap-around slicing, and
   staleness cap. `rollout_with_retry` retries RuntimeError exactly once.
"""

from __future__ import annotations

import pytest
import torch

import mythouro.training_utils as tu
from mythouro.main import MythOuro, MythOuroConfig
from mythouro.rollout import RolloutBuffer, rollout_with_retry
from mythouro.training_utils import generate_rollout, teacher_logits


def _tiny_student(**kw) -> MythOuro:
    d = dict(
        vocab_size=128, dim=64, n_heads=4, n_kv_heads=2, max_seq_len=128,
        max_loop_iters=2, prelude_layers=1, coda_layers=1, attn_type="gqa",
        n_experts=4, n_shared_experts=1, n_experts_per_tok=2, expert_dim=16,
        lora_rank=4, kv_lora_rank=16, q_lora_rank=16,
        qk_rope_head_dim=8, qk_nope_head_dim=8, v_head_dim=8, dropout=0.0,
    )
    d.update(kw)
    torch.manual_seed(0)
    return MythOuro(MythOuroConfig(**d)).eval()


@pytest.fixture(autouse=True)
def _fresh_teacher_gate():
    # The teacher-cache verdict is module-level (one per run); isolate tests.
    tu._reset_teacher_cache_gate()
    yield
    tu._reset_teacher_cache_gate()


# ---------------------------------------------------------------------------
# 1. Student cached decode == legacy full recompute (greedy)
# ---------------------------------------------------------------------------


class TestStudentCacheEquivalence:
    def _greedy_rollout(self, student, prompt, *, use_kv_cache, n_new=16):
        # temperature ~0 → softmax collapses to argmax one-hot → multinomial
        # is deterministic; top_k=0 disables filtering. Fixed torch seed for
        # the (degenerate) multinomial draw.
        torch.manual_seed(1234)
        return generate_rollout(
            student, None, prompt,
            n_loops=2, max_new_tokens=n_new,
            teacher_mix_alpha=0.0, temperature=1e-8, top_k=0,
            use_kv_cache=use_kv_cache,
        )

    def test_greedy_sequences_identical_gqa(self):
        student = _tiny_student()
        prompt = torch.randint(0, 128, (2, 12))
        legacy = self._greedy_rollout(student, prompt, use_kv_cache=False)
        cached = self._greedy_rollout(student, prompt, use_kv_cache=True)
        assert torch.equal(legacy, cached), (
            f"diverged at position "
            f"{(legacy != cached).any(dim=0).nonzero()[0].item()}"
        )

    def test_greedy_sequences_identical_mla(self):
        student = _tiny_student(attn_type="mla")
        prompt = torch.randint(0, 128, (2, 12))
        legacy = self._greedy_rollout(student, prompt, use_kv_cache=False)
        cached = self._greedy_rollout(student, prompt, use_kv_cache=True)
        assert torch.equal(legacy, cached)

    def test_cached_shape_and_prompt_preserved(self):
        student = _tiny_student()
        prompt = torch.randint(0, 128, (3, 9))
        out = self._greedy_rollout(student, prompt, use_kv_cache=True, n_new=7)
        assert out.shape == (3, 16)
        assert torch.equal(out[:, :9], prompt)

    def test_train_mode_restored(self):
        student = _tiny_student().train()
        prompt = torch.randint(0, 128, (1, 8))
        self._greedy_rollout(student, prompt, use_kv_cache=True, n_new=2)
        assert student.training

    @pytest.mark.skipif(
        not (hasattr(torch, "xpu") and torch.xpu.is_available()),
        reason="XPU not available",
    )
    def test_greedy_sequences_match_on_xpu_bf16(self):
        # bf16 autocast on real silicon: token-sequence match over >=64 new
        # tokens (acceptance criterion 1, XPU half).
        student = _tiny_student().to("xpu")
        prompt = torch.randint(0, 128, (2, 12), device="xpu")
        with torch.autocast(device_type="xpu", dtype=torch.bfloat16):
            legacy = self._greedy_rollout(
                student, prompt, use_kv_cache=False, n_new=64
            )
            cached = self._greedy_rollout(
                student, prompt, use_kv_cache=True, n_new=64
            )
        assert torch.equal(legacy.cpu(), cached.cpu())


# ---------------------------------------------------------------------------
# 2. Teacher cached path + validation gate
# ---------------------------------------------------------------------------


def _tiny_hf_teacher(vocab=128):
    transformers = pytest.importorskip("transformers")
    cfg = transformers.GPT2Config(
        vocab_size=vocab, n_positions=256, n_embd=32, n_layer=2, n_head=2,
    )
    torch.manual_seed(7)
    return transformers.GPT2LMHeadModel(cfg).eval()


class _KwargRejectingTeacher(torch.nn.Module):
    """Stub that dies on cache kwargs — the gate must catch it and fall back."""

    def __init__(self, vocab=128, dim=16):
        super().__init__()
        self.emb = torch.nn.Embedding(vocab, dim)
        self.head = torch.nn.Linear(dim, vocab)

    def forward(self, input_ids=None, **kwargs):
        if kwargs.get("use_cache"):
            raise TypeError("no cache support here")
        return self.head(self.emb(input_ids))


class TestTeacherCache:
    def test_cached_matches_uncached_on_real_hf_model(self):
        teacher = _tiny_hf_teacher()
        ids = torch.randint(0, 128, (1, 10))

        # Full-sequence reference vs incremental cached decode.
        ref = teacher_logits(teacher, ids)
        got_prefill, past = tu.teacher_logits_cached(teacher, ids, None, 0)
        assert torch.allclose(ref, got_prefill, atol=1e-5)

        nxt = ref[:, -1, :].argmax(dim=-1, keepdim=True)
        ids2 = torch.cat([ids, nxt], dim=1)
        ref2 = teacher_logits(teacher, ids2)[:, -1, :]
        got2, _ = tu.teacher_logits_cached(teacher, nxt, past, ids.shape[1])
        assert torch.allclose(ref2, got2[:, -1, :], atol=1e-5)

    def test_validation_gate_passes_for_real_hf_model(self):
        teacher = _tiny_hf_teacher()
        ids = torch.randint(0, 128, (2, 10))
        assert tu._teacher_cache_usable(teacher, ids) is True
        # Verdict memoised module-level.
        assert tu._TEACHER_CACHE_OK is True

    def test_validation_gate_falls_back_for_cacheless_stub(self):
        teacher = _KwargRejectingTeacher()
        ids = torch.randint(0, 128, (1, 10))
        assert tu._teacher_cache_usable(teacher, ids) is False
        assert tu._TEACHER_CACHE_OK is False

    def test_rollout_with_cacheless_teacher_still_works(self):
        # End-to-end: gate fails -> uncached teacher mixing -> rollout completes.
        student = _tiny_student()
        teacher = _KwargRejectingTeacher()
        prompt = torch.randint(0, 128, (2, 8))
        out = generate_rollout(
            student, teacher, prompt,
            n_loops=2, max_new_tokens=4,
            teacher_mix_alpha=0.5, temperature=1.0, top_k=5,
            use_kv_cache=True,
        )
        assert out.shape == (2, 12)
        assert tu._TEACHER_CACHE_OK is False

    def test_rollout_with_cached_teacher_end_to_end(self):
        student = _tiny_student()
        teacher = _tiny_hf_teacher()
        prompt = torch.randint(0, 128, (2, 8))
        out = generate_rollout(
            student, teacher, prompt,
            n_loops=2, max_new_tokens=4,
            teacher_mix_alpha=0.5, temperature=1.0, top_k=5,
            use_kv_cache=True,
        )
        assert out.shape == (2, 12)
        assert tu._TEACHER_CACHE_OK is True


# ---------------------------------------------------------------------------
# 3. RolloutBuffer + retry wrapper
# ---------------------------------------------------------------------------


class TestRolloutBuffer:
    def test_reuse_budget_and_refill_cycle(self):
        buf = RolloutBuffer(8, 2, reuse=2, max_age_steps=100)
        assert buf.needs_refill(0)
        buf.fill(torch.arange(8 * 4).reshape(8, 4), current_step=0)
        # 8 rows / micro 2 = 4 slices, x2 reuse = 8 draws before refill.
        for _ in range(8):
            assert not buf.needs_refill(1)
            assert buf.draw().shape == (2, 4)
        assert buf.needs_refill(1)
        with pytest.raises(RuntimeError):
            buf.draw()

    def test_slices_cycle_through_all_rows(self):
        buf = RolloutBuffer(4, 2, reuse=1, max_age_steps=100)
        data = torch.arange(4 * 3).reshape(4, 3)
        buf.fill(data, current_step=0)
        seen = torch.cat([buf.draw(), buf.draw()], dim=0)
        assert torch.equal(seen, data)         # each row served exactly once

    def test_staleness_cap_forces_refill(self):
        buf = RolloutBuffer(4, 2, reuse=100, max_age_steps=10)
        buf.fill(torch.zeros(4, 3, dtype=torch.long), current_step=5)
        assert not buf.needs_refill(14)
        assert buf.needs_refill(15)            # 15 - 5 >= 10

    def test_rollout_batch_rounded_down_to_micro_multiple(self):
        buf = RolloutBuffer(7, 2, reuse=1)
        assert buf.rollout_batch == 6

    def test_rejects_rollout_batch_below_micro(self):
        with pytest.raises(ValueError):
            RolloutBuffer(1, 2)


class TestRetry:
    def test_retries_runtime_error_once(self):
        calls = []

        def flaky():
            calls.append(1)
            if len(calls) == 1:
                raise RuntimeError("transient XPU abort")
            return "ok"

        assert rollout_with_retry(flaky) == "ok"
        assert len(calls) == 2

    def test_second_failure_raises(self):
        def always_fails():
            raise RuntimeError("real failure")

        with pytest.raises(RuntimeError, match="real failure"):
            rollout_with_retry(always_fails)
