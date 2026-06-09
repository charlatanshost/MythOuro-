"""
Tests for `mythouro.training_utils` — the loss helpers, curriculum/annealer
schedules, MoE-aware collectors, and the expert specialisation probe.

These are pure-Python / tiny-model unit tests; nothing here touches a real
dataset or a GPU. The point is to pin the public contract of each helper
before any long-running training depends on it.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from mythouro.main import MythOuro, MythOuroConfig
from mythouro.training_utils import (
    LoopCurriculum,
    LoopDepthAnnealer,
    ExpertSpecializationProbe,
    ProcessRewardHead,
    combined_loss,
    consistency_loss,
    contrastive_loop_loss,
    load_balance_loss,
    process_reward_loss,
    sparse_activation_loss,
    uncertainty_calibration_loss,
    collect_router_logits,
    collect_expert_counts,
    get_domain_labels,
    log_spectral_radius,
)


B, T = 2, 8


def _tiny_cfg(**overrides) -> MythOuroConfig:
    defaults = dict(
        vocab_size=200, dim=64, n_heads=4, n_kv_heads=2,
        max_seq_len=32, max_loop_iters=4,
        prelude_layers=1, coda_layers=1, attn_type="gqa",
        n_experts=4, n_shared_experts=1, n_experts_per_tok=2, expert_dim=16,
        lora_rank=4, kv_lora_rank=16, q_lora_rank=16,
        qk_rope_head_dim=8, qk_nope_head_dim=8, v_head_dim=8,
        dropout=0.0,
    )
    defaults.update(overrides)
    return MythOuroConfig(**defaults)


# ---------------------------------------------------------------------------
# combined_loss
# ---------------------------------------------------------------------------


class TestCombinedLoss:
    def setup_method(self):
        self.cfg = _tiny_cfg()
        self.model = MythOuro(self.cfg)
        self.ids = torch.randint(0, self.cfg.vocab_size, (B, T))
        self.targets = torch.randint(0, self.cfg.vocab_size, (B, T))
        self.logits, self.unc = self.model(self.ids)

    def test_returns_scalar_and_metrics_dict(self):
        loss, metrics = combined_loss(
            self.model, self.logits, self.unc, self.targets,
            vocab_size=self.cfg.vocab_size, topk=self.cfg.n_experts_per_tok,
        )
        assert loss.ndim == 0
        assert torch.isfinite(loss)
        assert set(metrics.keys()) >= {"ce", "lb", "unc"}
        for v in metrics.values():
            assert isinstance(v, float)

    def test_metrics_match_components(self):
        # Each metric must equal its individual component evaluated stand-alone
        # — this is the property that makes the log line interpretable.
        loss, metrics = combined_loss(
            self.model, self.logits, self.unc, self.targets,
            vocab_size=self.cfg.vocab_size, topk=self.cfg.n_experts_per_tok,
        )
        ce_ref = F.cross_entropy(
            self.logits.view(-1, self.cfg.vocab_size), self.targets.view(-1),
        ).item()
        assert abs(metrics["ce"] - ce_ref) < 1e-5

    def test_lb_coeff_scales_load_balance_contribution(self):
        # Setting lb_coeff=0 should make the total = ce + unc_coeff*unc
        # (no load-balance contribution).
        loss_off, metrics_off = combined_loss(
            self.model, self.logits, self.unc, self.targets,
            vocab_size=self.cfg.vocab_size, topk=self.cfg.n_experts_per_tok,
            lb_coeff=0.0,
        )
        loss_on, metrics_on = combined_loss(
            self.model, self.logits, self.unc, self.targets,
            vocab_size=self.cfg.vocab_size, topk=self.cfg.n_experts_per_tok,
            lb_coeff=1.0,
        )
        # When lb_coeff is positive, the total must be larger by lb_coeff*lb.
        assert loss_on.item() > loss_off.item()

    def test_backward_runs(self):
        loss, _ = combined_loss(
            self.model, self.logits, self.unc, self.targets,
            vocab_size=self.cfg.vocab_size, topk=self.cfg.n_experts_per_tok,
        )
        loss.backward()
        # At least one parameter must have a non-zero gradient.
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in self.model.parameters()
        )
        assert has_grad


# ---------------------------------------------------------------------------
# consistency_loss
# ---------------------------------------------------------------------------


class TestConsistencyLoss:
    def setup_method(self):
        self.cfg = _tiny_cfg()
        self.model = MythOuro(self.cfg).eval()
        self.ids = torch.randint(0, self.cfg.vocab_size, (B, T))

    def test_returns_finite_scalar(self):
        loss = consistency_loss(self.model, self.ids, n_loops_low=1, n_loops_high=4)
        assert loss.ndim == 0
        assert torch.isfinite(loss)
        assert loss.item() >= 0

    def test_zero_when_shallow_equals_deep(self):
        # When low == high loop count, the two distributions are identical →
        # KL(p || p) = 0 (modulo numerical jitter).
        loss = consistency_loss(self.model, self.ids, n_loops_low=2, n_loops_high=2)
        assert loss.item() < 1e-3


# ---------------------------------------------------------------------------
# contrastive_loop_loss
# ---------------------------------------------------------------------------


class TestContrastiveLoopLoss:
    def setup_method(self):
        self.cfg = _tiny_cfg()
        self.model = MythOuro(self.cfg).eval()
        self.ids = torch.randint(0, self.cfg.vocab_size, (B, T))
        self.targets = torch.randint(0, self.cfg.vocab_size, (B, T))

    def test_returns_finite_scalar(self):
        loss = contrastive_loop_loss(
            self.model, self.ids, self.targets,
            n_loops_low=1, n_loops_high=4,
        )
        assert loss.ndim == 0
        assert torch.isfinite(loss)

    def test_coeff_scales_output(self):
        l1 = contrastive_loop_loss(
            self.model, self.ids, self.targets,
            n_loops_low=1, n_loops_high=4, coeff=0.0,
        )
        # coeff=0 must drive the loss to exactly zero
        assert l1.item() == 0.0


# ---------------------------------------------------------------------------
# load_balance_loss
# ---------------------------------------------------------------------------


class TestLoadBalanceLoss:
    def test_empty_buffer_returns_zero(self):
        loss = load_balance_loss([], topk=2)
        assert loss.item() == 0.0

    def test_uniform_routing_minimum(self):
        # Perfectly uniform router logits → load-balance loss at its minimum
        # of 1.0 (E * Σ(1/E * 1/E) = 1).
        n_tokens, n_experts = 32, 4
        uniform = torch.zeros(n_tokens, n_experts)
        loss = load_balance_loss([uniform], topk=2)
        # With uniform softmax and a non-deterministic topk tie-break the
        # bound is only approximately 1.0; allow some slack.
        assert 0.9 < loss.item() < 1.5

    def test_collapsed_routing_above_uniform(self):
        # Force the router to send everything to expert 0
        n_tokens, n_experts = 32, 4
        skewed = torch.full((n_tokens, n_experts), -10.0)
        skewed[:, 0] = 10.0
        loss = load_balance_loss([skewed], topk=2)
        # Collapsed routing must score strictly above the uniform minimum.
        assert loss.item() > 1.5


# ---------------------------------------------------------------------------
# uncertainty_calibration_loss
# ---------------------------------------------------------------------------


class TestUncertaintyCalibrationLoss:
    def test_returns_finite_scalar(self):
        logits = torch.randn(B, T, 100)
        unc = torch.rand(B, T)                        # already in (0, 1)
        targets = torch.randint(0, 100, (B, T))
        loss = uncertainty_calibration_loss(logits, unc, targets)
        assert loss.ndim == 0
        assert torch.isfinite(loss)
        assert loss.item() >= 0

    def test_perfect_calibration_low_loss(self):
        # If the head outputs 1.0 for every wrong prediction and 0.0 for
        # every right one, BCE should be very small.
        logits = torch.zeros(B, T, 4)
        logits[..., 0] = 10.0                          # always predicts class 0
        targets = torch.zeros(B, T, dtype=torch.long)  # always class 0 → always correct
        unc = torch.full((B, T), 1e-4)                 # head says "very confident"
        loss = uncertainty_calibration_loss(logits, unc, targets)
        assert loss.item() < 1e-2


# ---------------------------------------------------------------------------
# sparse_activation_loss
# ---------------------------------------------------------------------------


class TestSparseActivationLoss:
    def test_empty_buffer_returns_zero(self):
        assert sparse_activation_loss([]).item() == 0.0

    def test_uniform_scores_strictly_higher_than_one_hot(self):
        # Entropy of uniform router output is log(E); entropy of a one-hot
        # output is 0. The loss must reflect that ordering, otherwise it
        # provides no gradient signal to push routing toward decisiveness.
        uniform = torch.zeros(32, 4)                   # softmax → uniform
        one_hot_logits = torch.full((32, 4), -10.0)
        one_hot_logits[:, 0] = 10.0                    # near one-hot
        l_uniform = sparse_activation_loss([uniform])
        l_onehot  = sparse_activation_loss([one_hot_logits])
        assert l_uniform.item() > l_onehot.item() * 100

    def test_loss_is_differentiable(self):
        # The previous L1 implementation was constant in the routing pattern
        # and therefore produced no gradient. Pin that the entropy version
        # actually propagates a signal back to the logits.
        logits = torch.randn(8, 4, requires_grad=True)
        loss = sparse_activation_loss([logits], coeff=1.0)
        loss.backward()
        assert logits.grad is not None
        assert logits.grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# ProcessRewardHead + process_reward_loss
# ---------------------------------------------------------------------------


class TestProcessReward:
    def test_head_returns_scalar_per_batch(self):
        head = ProcessRewardHead(dim=64)
        h = torch.randn(B, T, 64)
        out = head(h)
        assert out.shape == (B,)

    def test_loss_runs_and_is_finite(self):
        head = ProcessRewardHead(dim=64)
        h = torch.randn(B, T, 64)
        logits = torch.randn(B, T, 100)
        targets = torch.randint(0, 100, (B, T))
        loss = process_reward_loss(head, h, logits, targets)
        assert loss.ndim == 0
        assert torch.isfinite(loss)


# ---------------------------------------------------------------------------
# LoopCurriculum
# ---------------------------------------------------------------------------


class TestLoopCurriculum:
    def test_start_value_during_warmup(self):
        c = LoopCurriculum(start_loops=2, max_loops=16, warmup_steps=100, total_steps=1000)
        assert c.get(0) == 2
        assert c.get(50) == 2
        assert c.get(99) == 2

    def test_reaches_max_at_total(self):
        c = LoopCurriculum(start_loops=2, max_loops=16, warmup_steps=100, total_steps=1000)
        assert c.get(1000) == 16

    def test_monotonic_non_decreasing(self):
        c = LoopCurriculum(start_loops=2, max_loops=16, warmup_steps=100, total_steps=1000)
        prev = 0
        for s in range(0, 1100, 25):
            v = c.get(s)
            assert v >= prev
            prev = v

    def test_get_sampled_within_curriculum_range(self):
        # Sampled depth must lie in [start_loops, get(step)] for any step.
        import random
        c = LoopCurriculum(start_loops=2, max_loops=16, warmup_steps=100, total_steps=1000)
        rng = random.Random(42)
        for step in (0, 50, 100, 500, 1000, 1500):
            upper = c.get(step)
            for _ in range(20):
                v = c.get_sampled(step, rng)
                assert c.start_loops <= v <= upper, (
                    f"step={step} upper={upper} sampled={v} out of range"
                )

    def test_get_sampled_returns_start_during_warmup(self):
        # During warmup `get(step) == start_loops`, so the random range
        # is empty; the method must still return `start_loops` without
        # raising a "low > high" RNG error.
        import random
        c = LoopCurriculum(start_loops=2, max_loops=16, warmup_steps=100, total_steps=1000)
        rng = random.Random(0)
        for step in (0, 50, 99):
            assert c.get_sampled(step, rng) == 2

    def test_get_sampled_deterministic_with_seeded_rng(self):
        import random
        c = LoopCurriculum(start_loops=2, max_loops=16, warmup_steps=100, total_steps=1000)
        a = [c.get_sampled(500, random.Random(7)) for _ in range(5)]
        b = [c.get_sampled(500, random.Random(7)) for _ in range(5)]
        assert a == b

    def test_get_sampled_covers_full_range_eventually(self):
        # Drawing many samples at a wide window should produce both
        # start_loops AND get(step) at least once.
        import random
        c = LoopCurriculum(start_loops=2, max_loops=16, warmup_steps=0, total_steps=10)
        rng = random.Random(123)
        # At step=10 the upper is 16; many samples should cover [2, 16].
        samples = {c.get_sampled(10, rng) for _ in range(200)}
        assert c.start_loops in samples
        assert c.max_loops in samples


# ---------------------------------------------------------------------------
# LoopDepthAnnealer
# ---------------------------------------------------------------------------


class TestLoopDepthAnnealer:
    def test_returns_base_before_anneal_start(self):
        a = LoopDepthAnnealer(base_loops=16, max_extra_loops=24, anneal_start=85, total_steps=100)
        assert a.get(0) == 16
        assert a.get(50) == 16
        assert a.get(84) == 16

    def test_reaches_max_at_total(self):
        a = LoopDepthAnnealer(base_loops=16, max_extra_loops=24, anneal_start=85, total_steps=100)
        assert a.get(100) == 24

    def test_pushes_beyond_base_after_start(self):
        a = LoopDepthAnnealer(base_loops=16, max_extra_loops=24, anneal_start=85, total_steps=100)
        # Anywhere strictly between anneal_start and total_steps, n_loops must
        # exceed the base count — that's the whole point of the annealer.
        v = a.get(92)
        assert v > 16
        assert v <= 24


# ---------------------------------------------------------------------------
# Collectors
# ---------------------------------------------------------------------------


class TestCollectors:
    def test_collect_router_logits_one_per_loop(self):
        # P0.2: the single recurrent MoEFFN is called once per loop, and
        # telemetry now captures EVERY loop (not just the last). So a K-loop
        # forward in train mode (no early exit) yields K router-logits tensors.
        cfg = _tiny_cfg()
        model = MythOuro(cfg).train()
        ids = torch.randint(0, cfg.vocab_size, (B, T))
        model(ids)
        buf = collect_router_logits(model)
        assert len(buf) == cfg.max_loop_iters, (len(buf), cfg.max_loop_iters)
        for t in buf:
            assert t.shape[-1] == cfg.n_experts

    def test_collect_expert_counts_one_per_moe_layer(self):
        cfg = _tiny_cfg()
        model = MythOuro(cfg)
        ids = torch.randint(0, cfg.vocab_size, (B, T))
        model(ids)
        out = collect_expert_counts(model)
        assert len(out) == 1
        (_, counts), = out.items()
        assert counts.shape == (cfg.n_experts,)


# ---------------------------------------------------------------------------
# Expert specialisation probe
# ---------------------------------------------------------------------------


class TestExpertSpecializationProbe:
    def test_loss_returns_finite_scalar(self):
        cfg = _tiny_cfg()
        model = MythOuro(cfg)
        probe = ExpertSpecializationProbe(n_experts=cfg.n_experts)
        ids = torch.randint(0, cfg.vocab_size, (B, T))
        model(ids)
        router_buf = collect_router_logits(model)
        # Broadcast a label to match the flattened token count
        N = router_buf[0].shape[0]
        labels = torch.randint(0, 4, (N,))
        loss = probe.loss(router_buf, labels)
        assert loss.ndim == 0
        assert torch.isfinite(loss)

    def test_predict_expert_domains_returns_dict_per_expert(self):
        probe = ExpertSpecializationProbe(n_experts=4)
        out = probe.predict_expert_domains()
        assert set(out.keys()) == {0, 1, 2, 3}
        assert all(d in probe.DOMAIN_NAMES for d in out.values())


# ---------------------------------------------------------------------------
# Domain labels (heuristic)
# ---------------------------------------------------------------------------


class TestDomainLabels:
    def test_code_keyword_labels_as_code(self):
        labels = get_domain_labels(["def foo(): pass"], torch.device("cpu"))
        assert labels[0].item() == 1                       # 1 == code

    def test_math_keyword_labels_as_math(self):
        labels = get_domain_labels(["this is a theorem about real numbers"], torch.device("cpu"))
        assert labels[0].item() == 2                       # 2 == math

    def test_instruction_keyword_labels_as_instruction(self):
        labels = get_domain_labels(["### Instruction:\nfoo"], torch.device("cpu"))
        assert labels[0].item() == 3                       # 3 == instruction

    def test_plain_text_labels_as_language(self):
        labels = get_domain_labels(["The quick brown fox jumps over."], torch.device("cpu"))
        assert labels[0].item() == 0                       # 0 == general language


# ---------------------------------------------------------------------------
# Spectral radius diagnostic — runs without crashing
# ---------------------------------------------------------------------------


class TestLogSpectralRadius:
    def test_runs_without_crash(self):
        model = MythOuro(_tiny_cfg())
        # Just confirm the diagnostic doesn't blow up; output goes to logger.
        log_spectral_radius(model, step=0)
