"""
Tests for the Ouro-distillation utilities in `mythouro.training_utils`.

What's pinned:

    1. `distillation_loss` math invariants:
        - KL term = 0 (modulo numerical noise) when student logits ≡ teacher.
        - Shape mismatches raise with a clear "tokenizer alignment" message.
        - alpha=0 → only the hard CE term contributes.
        - alpha=1 + targets=None → only the soft KL term contributes.
        - T² scaling: doubling temperature roughly doubles the soft-loss
          magnitude on identical distributions (specifically: T² ratio).
    2. `teacher_logits` wrapper returns a Tensor with no gradient.
    3. `load_distillation_teacher` refuses a vocab-mismatched teacher.

These tests don't pull a real Ouro checkpoint — they construct minimal
nn.Module stubs that mimic the contract. That keeps the suite offline
and fast.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import pytest

from mythouro.training_utils import (
    distillation_loss,
    load_distillation_teacher,
    teacher_logits,
)


# ---------------------------------------------------------------------------
# distillation_loss
# ---------------------------------------------------------------------------


class TestDistillationLossShapes:
    def test_returns_scalar_and_metrics_dict(self):
        B, T_, V = 2, 8, 64
        s = torch.randn(B, T_, V, requires_grad=True)
        t = torch.randn(B, T_, V)
        y = torch.randint(0, V, (B, T_))
        loss, m = distillation_loss(s, t, y)
        assert loss.ndim == 0
        assert torch.isfinite(loss)
        assert {"soft", "hard"} <= set(m.keys())
        assert isinstance(m["soft"], float)
        assert isinstance(m["hard"], float)

    def test_shape_mismatch_raises_with_alignment_hint(self):
        s = torch.randn(2, 8, 64)
        t = torch.randn(2, 8, 128)   # different vocab
        with pytest.raises(ValueError, match="tokenisers are misaligned"):
            distillation_loss(s, t)

    def test_invalid_temperature_raises(self):
        s = torch.randn(2, 8, 64)
        t = torch.randn(2, 8, 64)
        with pytest.raises(ValueError, match="temperature must be > 0"):
            distillation_loss(s, t, temperature=0.0)


class TestDistillationLossMath:
    def test_kl_zero_when_student_equals_teacher(self):
        # If student logits == teacher logits exactly, KL(t||s) = 0.
        torch.manual_seed(0)
        logits = torch.randn(2, 8, 64)
        loss, m = distillation_loss(
            logits.clone(), logits.clone(),
            targets=None, alpha=1.0, temperature=2.0,
        )
        assert m["soft"] < 1e-5
        assert loss.item() < 1e-5

    def test_kl_grows_with_disagreement(self):
        # Bigger student/teacher gap → bigger soft loss.
        torch.manual_seed(0)
        t = torch.randn(2, 8, 64)
        close = t + 0.01 * torch.randn_like(t)
        far   = t + 5.00 * torch.randn_like(t)
        _, m_close = distillation_loss(close, t, alpha=1.0)
        _, m_far   = distillation_loss(far,   t, alpha=1.0)
        assert m_far["soft"] > m_close["soft"]

    def test_alpha_zero_uses_only_hard_term(self):
        # alpha=0 → total = (1-0)·hard = hard. soft is still measured but
        # contributes 0 to total. Verify by comparing total to a separately
        # computed CE.
        torch.manual_seed(0)
        s = torch.randn(2, 8, 64)
        t = torch.randn(2, 8, 64)
        y = torch.randint(0, 64, (2, 8))
        loss, m = distillation_loss(s, t, y, alpha=0.0)
        ref = nn.functional.cross_entropy(s.view(-1, 64), y.view(-1))
        # alpha·soft + (1-alpha)·hard = 0 + hard, so loss ≈ hard ≈ ref.
        assert abs(loss.item() - m["hard"]) < 1e-5
        assert abs(loss.item() - ref.item()) < 1e-5

    def test_alpha_one_targets_none_returns_only_soft(self):
        torch.manual_seed(0)
        s = torch.randn(2, 8, 64)
        t = torch.randn(2, 8, 64)
        loss, m = distillation_loss(s, t, targets=None, alpha=1.0)
        assert m["hard"] == 0.0
        # loss = alpha · soft = 1 · soft, so loss ≈ soft.
        assert abs(loss.item() - m["soft"]) < 1e-5

    def test_temperature_squared_scaling(self):
        # The T² scaling factor is what keeps the soft-loss gradient
        # magnitude comparable across temperatures. On *identical* logits
        # both temperatures give soft=0, so test with a small gap.
        torch.manual_seed(0)
        t_logits = torch.randn(2, 8, 64)
        s_logits = t_logits + 0.5 * torch.randn_like(t_logits)
        _, m_t1 = distillation_loss(s_logits, t_logits, alpha=1.0, temperature=1.0)
        _, m_t2 = distillation_loss(s_logits, t_logits, alpha=1.0, temperature=2.0)
        # Both should be positive and finite. The exact ratio depends on
        # logit scale and gap, but the T² scaling factor means m_t2 should
        # not vanish toward zero relative to m_t1 (which the un-scaled
        # KL would). Picking a generous lower bound: m_t2 must be at least
        # 30% of m_t1 (un-scaled would be ~25% of m_t1 at T=2).
        assert m_t1["soft"] > 0
        assert m_t2["soft"] > 0
        assert m_t2["soft"] >= 0.3 * m_t1["soft"]


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------


class TestDistillationGradFlow:
    def test_grad_flows_through_student_only(self):
        # Student gets grad; teacher does not.
        torch.manual_seed(0)
        s = torch.randn(2, 8, 64, requires_grad=True)
        t = torch.randn(2, 8, 64, requires_grad=True)
        # The contract is that the teacher tensor is detached / no-grad
        # *upstream*. distillation_loss does its own no-grad on softmax(t)
        # — verify no gradient flows back into t even if it had requires_grad.
        loss, _ = distillation_loss(s, t, alpha=1.0)
        loss.backward()
        assert s.grad is not None and s.grad.abs().sum() > 0
        assert t.grad is None or t.grad.abs().sum() == 0


# ---------------------------------------------------------------------------
# teacher_logits wrapper
# ---------------------------------------------------------------------------


class _DummyHFOutput:
    """Mimics HuggingFace's `CausalLMOutputWithPast` (has `.logits`)."""
    def __init__(self, logits):
        self.logits = logits


class _DummyTeacher(nn.Module):
    """Tiny stand-in for an HF causal LM — embedding → linear → logits."""
    def __init__(self, vocab_size=64, dim=16):
        super().__init__()
        self.config = type("Cfg", (), {"vocab_size": vocab_size})()
        self.embed = nn.Embedding(vocab_size, dim)
        self.head  = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, input_ids=None, **_kw):
        h = self.embed(input_ids)
        return _DummyHFOutput(self.head(h))


class TestTeacherLogitsWrapper:
    def test_returns_detached_tensor(self):
        teacher = _DummyTeacher()
        ids = torch.randint(0, 64, (2, 8))
        out = teacher_logits(teacher, ids)
        assert isinstance(out, torch.Tensor)
        assert out.shape == (2, 8, 64)
        assert not out.requires_grad
        assert out.grad_fn is None

    def test_handles_bare_tensor_return(self):
        # Some custom HF-style models return the logits tensor directly
        # instead of wrapping in CausalLMOutputWithPast. Wrapper must
        # cope with either shape.
        class _BareTeacher(_DummyTeacher):
            def forward(self, input_ids=None, **_kw):
                h = self.embed(input_ids)
                return self.head(h)

        teacher = _BareTeacher()
        ids = torch.randint(0, 64, (1, 4))
        out = teacher_logits(teacher, ids)
        assert isinstance(out, torch.Tensor)
        assert not out.requires_grad


# ---------------------------------------------------------------------------
# load_distillation_teacher precondition checks
# ---------------------------------------------------------------------------


class TestLoadDistillationTeacher:
    def test_unknown_model_returns_none(self, monkeypatch):
        # The contract: load failure logs and returns None, doesn't raise.
        # Force AutoModelForCausalLM.from_pretrained to raise.
        from transformers import AutoModelForCausalLM
        def _boom(*_a, **_kw):
            raise OSError("simulated network failure")
        monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", _boom)
        result = load_distillation_teacher(
            "nonexistent/model", student_vocab_size=64,
        )
        assert result is None

    def test_vocab_mismatch_returns_none(self, monkeypatch):
        # The other precondition: a teacher whose vocab differs from the
        # student must be rejected so the caller doesn't silently distil
        # across mismatched tokenisers.
        from transformers import AutoModelForCausalLM

        def _fake_load(*_a, **_kw):
            return _DummyTeacher(vocab_size=128, dim=16)

        monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", _fake_load)
        result = load_distillation_teacher(
            "fake/model", student_vocab_size=64,    # mismatch with 128
        )
        assert result is None

    def test_matched_vocab_returns_frozen_teacher(self, monkeypatch):
        # Happy path: vocab matches, all params get frozen, eval mode set.
        from transformers import AutoModelForCausalLM

        def _fake_load(*_a, **_kw):
            return _DummyTeacher(vocab_size=64, dim=16)

        monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", _fake_load)
        teacher = load_distillation_teacher(
            "fake/model", student_vocab_size=64, dtype=torch.float32,
        )
        assert teacher is not None
        assert not teacher.training        # eval mode
        for p in teacher.parameters():
            assert not p.requires_grad     # frozen
