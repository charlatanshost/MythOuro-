"""
Tests for the PonderNet × Ouro depth regulariser.

What's pinned:

    1. `RecurrentBlock.last_halt_distribution` is populated after a
       forward pass, has shape (B, T, K), and each (B, T) row sums to 1.
    2. `collect_halt_distributions` returns one entry per RecurrentBlock.
    3. `depth_regularization_loss`:
         - Returns 0 (no grad) when no forward pass has happened yet.
         - KL = 0 when the halt distribution is exactly uniform.
         - KL > 0 when the halt distribution is peaked.
         - Differentiable: grad flows back into student parameters.
         - Only `prior="uniform"` is implemented; other priors raise
           NotImplementedError (don't accidentally ship an option we
           haven't validated).
    4. `combined_loss` integration:
         - With `depth_reg_coeff=0` (default), the depth term doesn't
           contribute to total loss AND no halt-collection walk fires.
         - With `depth_reg_coeff > 0`, total increases by ≈ coeff × depth.
         - The `"depth"` key always appears in `metrics`.
"""

from __future__ import annotations

import math

import pytest
import torch

from mythouro.main import MythOuro, MythOuroConfig
from mythouro.training_utils import (
    collect_halt_distributions,
    combined_loss,
    depth_regularization_loss,
)


B, T = 2, 8


def _tiny_cfg(**overrides) -> MythOuroConfig:
    defaults = dict(
        vocab_size=64, dim=32, n_heads=4, n_kv_heads=2,
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
# Halt distribution on RecurrentBlock
# ---------------------------------------------------------------------------


class TestHaltDistributionShape:
    def setup_method(self):
        # `act_threshold` set high enough that ACT cumulative halting never
        # short-circuits before consuming `n_loops` iterations. Untrained
        # ACTHalting outputs ~0.5/step, so 4 loops sum to ~2.0; threshold
        # 99 keeps the loop from breaking and gives us a deterministic
        # K = n_loops for the shape assertions below. The actual product
        # contract still allows K < n_loops; that's tested separately.
        self.cfg = _tiny_cfg(max_loop_iters=4, act_threshold=99.0)
        self.model = MythOuro(self.cfg).train()
        self.ids = torch.randint(0, self.cfg.vocab_size, (B, T))

    def test_initial_state_is_none(self):
        # Before any forward, the attribute should not yet exist or be None.
        # Either way `collect_halt_distributions` must return [].
        fresh = MythOuro(self.cfg).train()
        assert collect_halt_distributions(fresh) == []

    def test_populated_after_forward(self):
        self.model(self.ids, n_loops=4)
        dist = self.model.recurrent.last_halt_distribution
        assert dist is not None
        # T_ext includes sink tokens. With act_threshold=99 the loop runs
        # all 4 iterations and K matches n_loops.
        T_ext = T + self.cfg.n_sink_tokens
        assert dist.shape == (B, T_ext, 4)

    def test_distribution_sums_to_one(self):
        self.model(self.ids, n_loops=4)
        dist = self.model.recurrent.last_halt_distribution
        sums = dist.sum(dim=-1)
        # PonderNet construction forces normalisation via the residual at
        # the last step; sums must be very close to 1 per (B, T) row.
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_distribution_non_negative(self):
        self.model(self.ids, n_loops=4)
        dist = self.model.recurrent.last_halt_distribution
        # `.clamp(min=0)` in the post-process makes the residual
        # non-negative; per-step probs are sigmoid(.) so already in (0, 1).
        assert (dist >= -1e-7).all()

    def test_K_can_be_less_than_n_loops_when_ACT_short_circuits(self):
        # The *actual* contract: K is the number of loops that ran. At
        # inference, K may be < n_loops when ACT cumulative-threshold halt
        # or convergence-detection fires. During training the loop runs to
        # full depth so the regulariser always sees K = n_loops — without
        # this guarantee, ACT collapses onto K=1 (a degenerate δ with
        # KL=0, silencing depth-reg). The regulariser computes KL over
        # whatever K turned out to be.
        cfg = _tiny_cfg(max_loop_iters=4, act_threshold=0.99)  # default ACT cap
        model = MythOuro(cfg).eval()
        ids = torch.randint(0, cfg.vocab_size, (B, T))
        model(ids, n_loops=4)
        dist = model.recurrent.last_halt_distribution
        K = dist.shape[-1]
        assert 1 <= K <= 4


class TestCollectHaltDistributions:
    def test_returns_one_per_recurrent_block(self):
        # MythOuro has exactly one RecurrentBlock. We pin K via a high
        # ACT threshold (see TestHaltDistributionShape.setup_method).
        cfg = _tiny_cfg(max_loop_iters=4, act_threshold=99.0)
        model = MythOuro(cfg).train()
        ids = torch.randint(0, cfg.vocab_size, (B, T))
        model(ids, n_loops=4)
        out = collect_halt_distributions(model)
        assert len(out) == 1
        assert out[0].shape[-1] == 4


# ---------------------------------------------------------------------------
# depth_regularization_loss
# ---------------------------------------------------------------------------


class TestDepthRegularizationLoss:
    def test_returns_zero_when_no_forward(self):
        cfg = _tiny_cfg()
        model = MythOuro(cfg).eval()
        # No forward pass yet — no distributions to read.
        loss = depth_regularization_loss(model, prior="uniform")
        assert loss.ndim == 0
        assert loss.item() == 0.0

    def test_kl_is_zero_for_uniform_distribution(self, monkeypatch):
        # Construct an artificial uniform halt distribution and inject it.
        # KL(uniform || uniform) must be 0.
        cfg = _tiny_cfg()
        model = MythOuro(cfg).eval()
        K = 4
        uniform = torch.full(
            (B, T + cfg.n_sink_tokens, K), 1.0 / K,
        )
        model.recurrent.last_halt_distribution = uniform
        loss = depth_regularization_loss(model, prior="uniform", coeff=1.0)
        assert loss.item() < 1e-5

    def test_kl_is_positive_for_peaked_distribution(self):
        # A "always halt at step 0" distribution is maximally far from
        # uniform — its KL should be exactly log(K).
        cfg = _tiny_cfg()
        model = MythOuro(cfg).eval()
        K = 4
        peaked = torch.zeros((B, T + cfg.n_sink_tokens, K))
        peaked[..., 0] = 1.0
        model.recurrent.last_halt_distribution = peaked
        loss = depth_regularization_loss(model, prior="uniform", coeff=1.0)
        expected = math.log(K)
        assert abs(loss.item() - expected) < 1e-4

    def test_coeff_scales_loss_linearly(self):
        cfg = _tiny_cfg()
        model = MythOuro(cfg).eval()
        K = 4
        peaked = torch.zeros((B, T + cfg.n_sink_tokens, K))
        peaked[..., 0] = 1.0
        model.recurrent.last_halt_distribution = peaked
        l1 = depth_regularization_loss(model, coeff=1.0).item()
        l2 = depth_regularization_loss(model, coeff=2.0).item()
        assert abs(l2 - 2.0 * l1) < 1e-5

    def test_grad_flows_back_to_act_halting(self):
        # The regulariser drives ACTHalting via the lambdas captured in
        # RecurrentBlock.forward. Verify gradient flow with a real forward.
        cfg = _tiny_cfg()
        model = MythOuro(cfg).train()
        ids = torch.randint(0, cfg.vocab_size, (B, T))
        # Force a real forward pass to populate halt_distribution with a
        # grad-tracked tensor.
        logits, _ = model(ids, n_loops=4)
        # halt_distribution should be in the autograd graph (we did NOT
        # detach it on capture).
        dist = model.recurrent.last_halt_distribution
        assert dist.requires_grad
        loss = depth_regularization_loss(model, prior="uniform", coeff=1.0)
        loss.backward()
        # ACTHalting must have received gradient from the regulariser.
        halt_weight = model.recurrent.act.halt.weight
        assert halt_weight.grad is not None
        assert halt_weight.grad.abs().sum() > 0

    def test_geometric_prior_raises(self):
        cfg = _tiny_cfg()
        model = MythOuro(cfg).eval()
        K = 4
        dist = torch.full((B, T + cfg.n_sink_tokens, K), 1.0 / K)
        model.recurrent.last_halt_distribution = dist
        with pytest.raises(NotImplementedError, match="uniform"):
            depth_regularization_loss(model, prior="geometric")


# ---------------------------------------------------------------------------
# combined_loss integration
# ---------------------------------------------------------------------------


class TestCombinedLossIntegration:
    def setup_method(self):
        self.cfg = _tiny_cfg()
        self.model = MythOuro(self.cfg)
        self.ids = torch.randint(0, self.cfg.vocab_size, (B, T))
        self.targets = torch.randint(0, self.cfg.vocab_size, (B, T))
        self.logits, self.unc = self.model(self.ids)

    def test_depth_metric_present_when_disabled(self):
        loss, m = combined_loss(
            self.model, self.logits, self.unc, self.targets,
            vocab_size=self.cfg.vocab_size, topk=self.cfg.n_experts_per_tok,
            depth_reg_coeff=0.0,
        )
        # Metric is always reported so the log line shape is stable;
        # value is 0 when the regulariser is off.
        assert "depth" in m
        assert m["depth"] == 0.0

    def test_total_loss_unchanged_when_disabled(self):
        # depth_reg_coeff=0 must not change the total loss vs. the
        # version of combined_loss before this addition. We can't compare
        # against the old version directly, but we can check that loss
        # equals ce + lb_coeff*lb + unc_coeff*unc with the same metrics.
        loss, m = combined_loss(
            self.model, self.logits, self.unc, self.targets,
            vocab_size=self.cfg.vocab_size, topk=self.cfg.n_experts_per_tok,
            lb_coeff=1e-2, unc_coeff=5e-2, depth_reg_coeff=0.0,
        )
        expected = m["ce"] + 1e-2 * m["lb"] + 5e-2 * m["unc"]
        assert abs(loss.item() - expected) < 1e-4

    def test_total_loss_increases_when_enabled(self):
        _, m_off = combined_loss(
            self.model, self.logits, self.unc, self.targets,
            vocab_size=self.cfg.vocab_size, topk=self.cfg.n_experts_per_tok,
            depth_reg_coeff=0.0,
        )
        loss_on, m_on = combined_loss(
            self.model, self.logits, self.unc, self.targets,
            vocab_size=self.cfg.vocab_size, topk=self.cfg.n_experts_per_tok,
            depth_reg_coeff=1.0,
        )
        # The depth metric is reported with `coeff=1.0` (the helper-call
        # coefficient), and the cfg-driven coefficient multiplies it
        # again. Both add a non-negative contribution.
        assert m_on["depth"] >= 0.0
        # Total must rise by exactly depth_reg_coeff * m_on["depth"]
        # vs. the off case (everything else is identical).
        expected_delta = 1.0 * m_on["depth"]
        actual_delta = loss_on.item() - (
            m_off["ce"] + 1e-2 * m_off["lb"] + 5e-2 * m_off["unc"]
        )
        assert abs(actual_delta - expected_delta) < 1e-4
