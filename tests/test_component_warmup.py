"""
Tests for the §3 new-component LR warmup.

Pinning the contract:

    1. `_collect_new_component_param_ids` picks up exactly the risk
       surface — InjectionScheduler.log_scale, LoRAAdapter.down, and all
       of MultiScaleInjection. Self-warming components (CrossLoopAttention,
       UncertaintyHead, ProcessRewardHead) must NOT be picked up.
    2. `get_optimizer_groups` partitions cleanly: every model parameter
       lands in exactly one group, with extra base params (aux heads)
       appended to the base group.
    3. `ComponentWarmup.factor` ramps 0→1 linearly over `warmup_steps`
       and clamps at 1.0 afterward.
    4. `apply_component_warmup` mutates per-group LRs correctly each step.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from mythouro.main import (
    MythOuro, MythOuroConfig,
    InjectionScheduler, LoRAAdapter, MultiScaleInjection,
    CrossLoopAttention, UncertaintyHead,
)
from mythouro.training_utils import (
    ComponentWarmup,
    ProcessRewardHead,
    _collect_new_component_param_ids,
    apply_component_warmup,
    get_optimizer_groups,
)


def _tiny_cfg(**overrides) -> MythOuroConfig:
    defaults = dict(
        vocab_size=64, dim=32, n_heads=4, n_kv_heads=2, max_seq_len=32,
        max_loop_iters=2, prelude_layers=1, coda_layers=1, attn_type="gqa",
        n_experts=4, n_shared_experts=1, n_experts_per_tok=2, expert_dim=16,
        lora_rank=4, kv_lora_rank=16, q_lora_rank=16,
        qk_rope_head_dim=8, qk_nope_head_dim=8, v_head_dim=8,
    )
    defaults.update(overrides)
    return MythOuroConfig(**defaults)


# ---------------------------------------------------------------------------
# Membership: which params are in the warmup group?
# ---------------------------------------------------------------------------


class TestNewComponentMembership:
    def setup_method(self):
        self.model = MythOuro(_tiny_cfg())
        self.warmup_ids = _collect_new_component_param_ids(self.model)

    def test_injection_scheduler_log_scale_is_included(self):
        # Every InjectionScheduler's log_scale must be in the warmup group
        sched = self.model.recurrent.injection.scheduler
        assert isinstance(sched, InjectionScheduler)
        assert id(sched.log_scale) in self.warmup_ids

    def test_lora_down_is_included(self):
        # LoRAAdapter.down is the non-zero-init half that needs warming.
        lora = self.model.recurrent.lora
        assert isinstance(lora, LoRAAdapter)
        for p in lora.down.parameters():
            assert id(p) in self.warmup_ids

    def test_lora_B_and_scale_are_NOT_included(self):
        # B is zero-init → self-warms; scale only multiplies that zero,
        # so it has nothing to push around. Including them would be
        # unnecessary churn.
        lora = self.model.recurrent.lora
        assert id(lora.B) not in self.warmup_ids
        for p in lora.scale.parameters():
            assert id(p) not in self.warmup_ids

    def test_multiscale_injection_all_params_included(self):
        ms = self.model.recurrent.ms_inject
        assert isinstance(ms, MultiScaleInjection)
        for p in ms.parameters():
            assert id(p) in self.warmup_ids

    def test_cross_loop_attention_is_excluded(self):
        # `o_proj` is zero-init → starts as identity residual, self-warms.
        cla = self.model.recurrent.cross_loop_attn
        assert isinstance(cla, CrossLoopAttention)
        for p in cla.parameters():
            assert id(p) not in self.warmup_ids

    def test_uncertainty_head_is_excluded(self):
        # Final Linear is zero-init → sigmoid(0)=0.5 at start, self-warms.
        head = self.model.uncertainty
        assert isinstance(head, UncertaintyHead)
        for p in head.parameters():
            assert id(p) not in self.warmup_ids

    def test_main_transformer_blocks_are_excluded(self):
        # The whole base architecture (embed, prelude, coda, recurrent.block,
        # the LM head, etc.) must NOT be in the warmup group.
        excluded = {"embed", "prelude", "coda", "head", "norm", "sink"}
        for name, p in self.model.named_parameters():
            if any(name.startswith(e) or f".{e}." in name for e in excluded):
                assert id(p) not in self.warmup_ids, (
                    f"unexpected base param in warmup group: {name}"
                )


# ---------------------------------------------------------------------------
# Optimizer groups: partition correctness
# ---------------------------------------------------------------------------


class TestOptimizerGroups:
    def setup_method(self):
        self.cfg = _tiny_cfg()
        self.model = MythOuro(self.cfg)
        self.prm_head = ProcessRewardHead(self.cfg.dim)
        self.groups = get_optimizer_groups(
            self.model,
            base_lr=1e-3,
            weight_decay=0.1,
            extra_base_params=list(self.prm_head.parameters()),
        )

    def test_two_named_groups(self):
        names = [g["name"] for g in self.groups]
        assert names == ["base", "new_component"]

    def test_no_param_appears_twice(self):
        seen: set = set()
        for g in self.groups:
            for p in g["params"]:
                pid = id(p)
                assert pid not in seen, "parameter appears in multiple groups"
                seen.add(pid)

    def test_every_model_param_is_assigned(self):
        seen = set()
        for g in self.groups:
            for p in g["params"]:
                seen.add(id(p))
        for name, p in self.model.named_parameters():
            assert id(p) in seen, f"model param missing from optimizer: {name}"

    def test_extra_base_params_land_in_base_group(self):
        base = next(g for g in self.groups if g["name"] == "base")
        extra_ids = {id(p) for p in self.prm_head.parameters()}
        base_ids = {id(p) for p in base["params"]}
        assert extra_ids.issubset(base_ids)

    def test_weight_decay_propagates(self):
        for g in self.groups:
            assert g["weight_decay"] == 0.1

    def test_optimizer_accepts_groups(self):
        # The end-to-end smoke: can we actually construct AdamW with them
        # and run one optimizer step?
        opt = torch.optim.AdamW(self.groups)
        ids = torch.randint(0, self.cfg.vocab_size, (2, 8))
        logits, unc = self.model(ids)
        (logits.sum() + unc.sum() + self.prm_head(torch.randn(2, 8, self.cfg.dim)).sum()).backward()
        opt.step()


# ---------------------------------------------------------------------------
# Warmup factor curve
# ---------------------------------------------------------------------------


class TestComponentWarmupFactor:
    def test_zero_at_step_zero(self):
        assert ComponentWarmup(1000).factor(0) == 0.0

    def test_halfway_at_half(self):
        assert ComponentWarmup(1000).factor(500) == 0.5

    def test_one_at_warmup_end(self):
        assert ComponentWarmup(1000).factor(1000) == 1.0

    def test_clamped_after_warmup(self):
        assert ComponentWarmup(1000).factor(5000) == 1.0

    def test_zero_warmup_returns_one(self):
        # Edge case: warmup_steps=0 → no warmup, full LR immediately.
        assert ComponentWarmup(0).factor(0) == 1.0
        assert ComponentWarmup(0).factor(500) == 1.0

    def test_monotonic_non_decreasing(self):
        cw = ComponentWarmup(100)
        prev = -1.0
        for s in range(0, 200, 5):
            f = cw.factor(s)
            assert f >= prev
            prev = f


# ---------------------------------------------------------------------------
# apply_component_warmup: optimizer mutation per step
# ---------------------------------------------------------------------------


class TestApplyComponentWarmup:
    def setup_method(self):
        self.model = MythOuro(_tiny_cfg())
        self.opt = torch.optim.AdamW(
            get_optimizer_groups(self.model, base_lr=1e-3),
        )

    def _lrs(self) -> "dict[str, float]":
        return {g["name"]: g["lr"] for g in self.opt.param_groups}

    def test_step_zero_warms_new_group_only(self):
        factor = apply_component_warmup(
            self.opt, base_lr=2e-3, step=0, warmup_steps=1000,
        )
        assert factor == 0.0
        lrs = self._lrs()
        assert lrs["base"] == 2e-3
        assert lrs["new_component"] == 0.0

    def test_half_warmup(self):
        factor = apply_component_warmup(
            self.opt, base_lr=2e-3, step=500, warmup_steps=1000,
        )
        assert factor == 0.5
        lrs = self._lrs()
        assert lrs["base"] == 2e-3
        assert lrs["new_component"] == 1e-3

    def test_post_warmup_full_lr(self):
        factor = apply_component_warmup(
            self.opt, base_lr=2e-3, step=2000, warmup_steps=1000,
        )
        assert factor == 1.0
        lrs = self._lrs()
        assert lrs["base"] == 2e-3
        assert lrs["new_component"] == 2e-3

    def test_lr_can_change_between_calls(self):
        # The training loop calls apply each step with the current cosine LR;
        # both groups must track changes to base_lr correctly.
        apply_component_warmup(self.opt, base_lr=1e-3, step=2000, warmup_steps=1000)
        apply_component_warmup(self.opt, base_lr=5e-4, step=2001, warmup_steps=1000)
        lrs = self._lrs()
        assert lrs["base"] == 5e-4
        assert lrs["new_component"] == 5e-4
