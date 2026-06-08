"""
Tests for MoE expansion / model growth.

The contract is unusual for ML code: the promoted model MUST compute the
exact same function as the source at the moment of promotion, within
floating-point tolerance. If that invariant breaks, model growth introduces
a loss spike that defeats the whole point — we'd be better off training
from scratch.

Eight invariants tested:

  1. Promotion preserves the source's forward output bit-for-bit
     (function-preservation — the whole point of the design).
  2. Router weight shape is correct after promotion (E_tgt × dim).
  3. Source experts are byte-identical in target slots [0:E_src].
  4. New experts have zeroed `down.weight` (the gating trick).
  5. router_bias[:E_src] equals source bias; router_bias[E_src:] equals
     the sentinel value.
  6. Sentinel decay schedule produces expected factors at boundary steps.
  7. A training step on the promoted model runs without errors (catches
     shape mismatches in autograd / optimizer construction).
  8. The DeepSeek-V3 bias updater handles the expanded `n_experts` pool
     and pushes biases in the expected direction for synthetic counts.
"""

from __future__ import annotations

import os
import tempfile

import pytest
import torch

from mythouro.main import MythOuro, MythOuroConfig, MoEFFN
from mythouro.grow import (
    DEFAULT_SENTINEL_BIAS,
    apply_sentinel_to_router_biases,
    get_sentinel_decay_factor,
    grow_moe_checkpoint,
    _promote_state_dict,
)
from mythouro.training_utils import update_router_bias_from_counts


# ---------------------------------------------------------------------------
# Tiny config — minimum that exercises MoEFFN without taking forever
# ---------------------------------------------------------------------------


def _tiny_cfg(*, n_experts: int = 4) -> MythOuroConfig:
    """
    Minimum-sized cfg that builds a real MythOuro with MoE. Used everywhere
    in this file — keeps tests fast (sub-second forward) while exercising
    real promotion code paths.
    """
    return MythOuroConfig(
        vocab_size=64,
        dim=32,
        n_heads=2,
        n_kv_heads=2,
        max_seq_len=16,
        max_loop_iters=2,
        prelude_layers=1,
        coda_layers=1,
        attn_type="gqa",
        kv_lora_rank=8,
        q_lora_rank=8,
        qk_rope_head_dim=8,
        qk_nope_head_dim=8,
        v_head_dim=8,
        n_experts=n_experts,
        n_shared_experts=1,
        n_experts_per_tok=2,
        expert_dim=32,
        act_threshold=0.99,
        rope_theta=10000.0,
        lora_rank=4,
    )


def _make_source_model_with_random_state(seed: int = 0) -> tuple[MythOuro, MythOuroConfig]:
    """Build a source MythOuro and randomize its weights so the test
    distinguishes copies from re-initialisations."""
    torch.manual_seed(seed)
    cfg = _tiny_cfg(n_experts=4)
    model = MythOuro(cfg)

    # Perturb router_bias so the test can assert it round-trips. Default init
    # is zeros which would alias with "fresh init" in the grown checkpoint.
    with torch.no_grad():
        for mod in model.modules():
            if isinstance(mod, MoEFFN):
                mod.router_bias.copy_(torch.randn_like(mod.router_bias) * 0.1)
    return model, cfg


# ---------------------------------------------------------------------------
# 1. Function preservation — the central invariant
# ---------------------------------------------------------------------------


class TestFunctionPreservation:
    def test_promoted_model_matches_source_output(self):
        """
        The big one. Build a source, promote it, build a target model with
        the bigger cfg, load the promoted state dict, and verify that for
        the same input both models produce the same logits (within fp
        tolerance).

        If this test ever fails, model growth is broken and the design's
        bit-exact guarantee no longer holds. Investigate before training.
        """
        src_model, src_cfg = _make_source_model_with_random_state(seed=42)
        src_model.eval()

        # Run the source forward to capture reference output.
        torch.manual_seed(1)
        ids = torch.randint(0, src_cfg.vocab_size, (2, 8))
        with torch.no_grad():
            src_logits, _ = src_model(ids)

        # Promote the state dict.
        src_state = src_model.state_dict()
        tgt_state = _promote_state_dict(
            src_state,
            e_src=src_cfg.n_experts,
            e_tgt=src_cfg.n_experts * 2,
            sentinel_bias=DEFAULT_SENTINEL_BIAS,
            perturb_scale=0.0,
        )

        # Build the target model and load promoted weights.
        tgt_cfg = _tiny_cfg(n_experts=src_cfg.n_experts * 2)
        tgt_model = MythOuro(tgt_cfg)
        tgt_model.load_state_dict(tgt_state)
        tgt_model.eval()

        with torch.no_grad():
            tgt_logits, _ = tgt_model(ids)

        # The function-preservation contract: identical logits. We allow a
        # very small tolerance for floating-point reorder noise from the
        # different topk path (top-k over E_tgt vs E_src), but no expert
        # should actually be selected from the new slots, so the answer
        # should be effectively identical.
        assert torch.allclose(src_logits, tgt_logits, atol=1e-5, rtol=1e-5), (
            "Promoted model output diverged from source — function-"
            "preservation contract broken. Check that new experts have "
            "zeroed `down.weight` AND sentinel bias is large enough to "
            "exclude them from top-k."
        )


# ---------------------------------------------------------------------------
# 2-5. State-dict structural invariants
# ---------------------------------------------------------------------------


class TestStateDictShape:
    def setup_method(self):
        torch.manual_seed(0)
        self.cfg = _tiny_cfg(n_experts=4)
        self.model = MythOuro(self.cfg)
        self.src_state = self.model.state_dict()
        self.e_tgt = 8
        self.tgt_state = _promote_state_dict(
            self.src_state,
            e_src=self.cfg.n_experts,
            e_tgt=self.e_tgt,
            sentinel_bias=DEFAULT_SENTINEL_BIAS,
            perturb_scale=0.0,
        )

    def test_router_weight_shape_correct(self):
        """Every MoEFFN router weight should grow from (E_src, dim) →
        (E_tgt, dim)."""
        for k, v in self.tgt_state.items():
            if k.endswith(".router.weight"):
                assert v.shape[0] == self.e_tgt, (
                    f"{k}: expected first dim {self.e_tgt}, got {tuple(v.shape)}"
                )
                # dim should match source.
                src_dim = self.src_state[k].shape[1]
                assert v.shape[1] == src_dim

    def test_source_experts_are_byte_identical(self):
        """Target experts [0:E_src] must equal source experts byte-for-byte.
        If this drifts, the first-E_src experts have been corrupted and the
        function-preservation guarantee is gone."""
        e_src = self.cfg.n_experts
        for i in range(e_src):
            for piece in ("gate.weight", "up.weight", "down.weight"):
                # Find any matching key — there's one per MoEFFN instance.
                for k in self.tgt_state.keys():
                    if k.endswith(f".routed_experts.{i}.{piece}"):
                        assert torch.equal(self.tgt_state[k], self.src_state[k]), (
                            f"{k} drifted from source — promotion corrupted "
                            "existing experts."
                        )

    def test_new_experts_have_zero_down_projection(self):
        """The function-preservation trick: new experts' `down.weight` is
        zero so even if the router ever selected them, their contribution
        is zero. The sentinel bias is the *first* guard; this is the
        belt-and-braces second guard."""
        e_src = self.cfg.n_experts
        e_tgt = self.e_tgt
        for i in range(e_src, e_tgt):
            for k in self.tgt_state.keys():
                if k.endswith(f".routed_experts.{i}.down.weight"):
                    w = self.tgt_state[k]
                    assert torch.all(w == 0), (
                        f"{k} should be zeroed (new expert), but contains "
                        f"non-zero values; max abs = {w.abs().max().item()}"
                    )

    def test_router_bias_split(self):
        """`router_bias[:E_src]` carries source bias forward;
        `router_bias[E_src:]` is sentinel-initialized."""
        e_src = self.cfg.n_experts
        for k in self.tgt_state.keys():
            if not k.endswith(".router_bias"):
                continue
            b_tgt = self.tgt_state[k]
            b_src = self.src_state[k]
            assert torch.equal(b_tgt[:e_src], b_src), (
                f"{k}: first {e_src} entries diverged from source bias"
            )
            assert torch.all(b_tgt[e_src:] == DEFAULT_SENTINEL_BIAS), (
                f"{k}: new entries should equal sentinel "
                f"{DEFAULT_SENTINEL_BIAS}, got {b_tgt[e_src:].tolist()}"
            )


# ---------------------------------------------------------------------------
# 6. Sentinel decay schedule
# ---------------------------------------------------------------------------


class TestSentinelDecay:
    def setup_method(self):
        self.meta = {
            "n_decay_steps": 500,
            "sentinel_bias": -100.0,
            "source_n_experts": 4,
            "target_n_experts": 8,
        }

    def test_decay_factor_at_step_0(self):
        assert get_sentinel_decay_factor(0, self.meta) == 1.0

    def test_decay_factor_at_midpoint(self):
        f = get_sentinel_decay_factor(250, self.meta)
        assert abs(f - 0.5) < 1e-6, f"Expected 0.5, got {f}"

    def test_decay_factor_at_endpoint(self):
        # At exactly the decay end, factor is 0 (training-script-side guard
        # means subsequent steps also stay 0).
        assert get_sentinel_decay_factor(500, self.meta) == 0.0

    def test_decay_factor_past_endpoint(self):
        assert get_sentinel_decay_factor(1000, self.meta) == 0.0

    def test_apply_sentinel_writes_to_router_bias(self):
        """`apply_sentinel_to_router_biases` should overwrite the new-expert
        slice of every MoEFFN's router_bias with sentinel * decay_factor.
        Without this, the DeepSeek-V3 updater would freely drift new
        experts' biases during the warm-in window."""
        torch.manual_seed(0)
        cfg = _tiny_cfg(n_experts=8)  # post-promotion size
        model = MythOuro(cfg)

        # Pretend the first 4 experts are "source" and the rest are new.
        meta = {
            **self.meta,
            "source_n_experts": 4,
            "target_n_experts": 8,
        }

        # Pre-set the new-expert slice to arbitrary nonzero values to check
        # they get overwritten.
        for mod in model.modules():
            if isinstance(mod, MoEFFN):
                mod.router_bias[4:] = 99.0

        # Apply at step 100 (factor = 1 - 100/500 = 0.8 → bias = -80)
        apply_sentinel_to_router_biases(model, meta, step=100)

        for mod in model.modules():
            if isinstance(mod, MoEFFN):
                assert torch.all(mod.router_bias[4:] == -80.0)

    def test_apply_sentinel_noop_after_decay(self):
        """After decay finishes, the function returns early without touching
        the router_bias. This is the handoff to the DeepSeek-V3 updater."""
        torch.manual_seed(0)
        cfg = _tiny_cfg(n_experts=8)
        model = MythOuro(cfg)

        # Set router_bias to something the updater might have left.
        for mod in model.modules():
            if isinstance(mod, MoEFFN):
                mod.router_bias[4:] = 42.0

        apply_sentinel_to_router_biases(
            model, {**self.meta, "source_n_experts": 4}, step=10_000,
        )

        # Untouched — the updater's values survive.
        for mod in model.modules():
            if isinstance(mod, MoEFFN):
                assert torch.all(mod.router_bias[4:] == 42.0)


# ---------------------------------------------------------------------------
# 7. Training-step smoke
# ---------------------------------------------------------------------------


class TestTrainingStep:
    def test_promoted_model_supports_a_train_step(self):
        """End-to-end: build a promoted model, run a forward + backward +
        optimizer step. Catches shape mismatches in the optimizer state
        construction, autograd graph through the zero-`down` experts, etc.
        """
        src_model, src_cfg = _make_source_model_with_random_state(seed=0)
        src_state = src_model.state_dict()
        tgt_state = _promote_state_dict(
            src_state,
            e_src=src_cfg.n_experts,
            e_tgt=src_cfg.n_experts * 2,
            sentinel_bias=DEFAULT_SENTINEL_BIAS,
            perturb_scale=0.0,
        )

        tgt_cfg = _tiny_cfg(n_experts=src_cfg.n_experts * 2)
        tgt_model = MythOuro(tgt_cfg)
        tgt_model.load_state_dict(tgt_state)
        tgt_model.train()

        opt = torch.optim.AdamW(tgt_model.parameters(), lr=1e-4)
        ids = torch.randint(0, tgt_cfg.vocab_size, (2, 8))

        opt.zero_grad()
        logits, _ = tgt_model(ids)
        # Simple synthetic loss — content doesn't matter, just that it
        # backprops through the whole graph.
        loss = logits.float().square().mean()
        loss.backward()
        opt.step()
        # If we got here without an exception, the shapes line up end-to-end.


# ---------------------------------------------------------------------------
# 8. DeepSeek-V3 bias updater handles the expanded pool
# ---------------------------------------------------------------------------


class TestBiasUpdaterCompat:
    def test_updater_runs_on_promoted_pool_and_moves_bias(self):
        """After sentinel decay, the regular bias updater takes over for
        the expanded pool. Verify it runs without shape errors and pushes
        bias for underused experts in the correct direction."""
        torch.manual_seed(0)
        cfg = _tiny_cfg(n_experts=8)
        model = MythOuro(cfg)

        # Find the MoEFFN module name so we can construct synthetic counts.
        moe_names = [
            n for n, m in model.named_modules() if isinstance(m, MoEFFN)
        ]
        assert moe_names, "Test setup error: no MoEFFN in model"

        # Synthetic counts: first 4 experts heavily used, last 4 ignored.
        counts_by_layer = {
            name: torch.tensor([100, 100, 100, 100, 0, 0, 0, 0])
            for name in moe_names
        }

        # Capture starting bias.
        before = {n: m.router_bias.clone()
                  for n, m in model.named_modules() if isinstance(m, MoEFFN)}

        update_router_bias_from_counts(
            model, counts_by_layer, bias_lr=1e-2, ddp=False,
        )

        # Underused experts (last 4) should have their bias INCREASED;
        # overused (first 4) decreased. The DeepSeek-V3 rule is
        # `bias[i] += lr * sign(target - count[i])`.
        for n, m in model.named_modules():
            if not isinstance(m, MoEFFN):
                continue
            delta = m.router_bias - before[n]
            assert (delta[:4] < 0).all(), (
                f"{n}: overused experts' bias should have decreased, "
                f"got delta {delta[:4].tolist()}"
            )
            assert (delta[4:] > 0).all(), (
                f"{n}: underused experts' bias should have increased, "
                f"got delta {delta[4:].tolist()}"
            )


# ---------------------------------------------------------------------------
# Integration: end-to-end grow_moe_checkpoint via disk roundtrip
# ---------------------------------------------------------------------------


class TestEndToEndCheckpointRoundtrip:
    def test_grow_moe_checkpoint_writes_loadable_file(self, tmp_path):
        """Write a source checkpoint to disk, run grow_moe_checkpoint, and
        verify the output loads into a `mythouro_distill_small`-style
        target model successfully.

        This is the integration boundary: if the CLI path is broken even
        though the unit tests pass, this catches it.
        """
        src_model, src_cfg = _make_source_model_with_random_state(seed=7)

        src_path = os.path.join(tmp_path, "src.pt")
        dst_path = os.path.join(tmp_path, "dst.pt")

        # Build a minimal source checkpoint payload matching what
        # `save_checkpoint` produces.
        from dataclasses import asdict
        torch.save(
            {
                "checkpoint_version": 2,
                "step": 1234,
                "model": src_model.state_dict(),
                "optimizer": {},
                "cfg": src_cfg,
                "cfg_dict": asdict(src_cfg),
                "vocab_size": src_cfg.vocab_size,
                "rng_state": None,
                "scaler_state": None,
                "extra": {},
            },
            src_path,
        )

        metadata = grow_moe_checkpoint(
            src_path=src_path,
            dst_path=dst_path,
            expansion_factor=2,
        )

        assert metadata["source_n_experts"] == src_cfg.n_experts
        assert metadata["target_n_experts"] == src_cfg.n_experts * 2
        assert metadata["expansion_factor"] == 2

        # Load the promoted checkpoint and verify it ports onto a fresh
        # target model.
        dst_ckpt = torch.load(dst_path, map_location="cpu", weights_only=False)
        tgt_cfg = dst_ckpt["cfg"]
        assert tgt_cfg.n_experts == src_cfg.n_experts * 2

        tgt_model = MythOuro(tgt_cfg)
        tgt_model.load_state_dict(dst_ckpt["model"])

        # Metadata round-tripped via the file.
        assert "growth_metadata" in dst_ckpt["extra"]
        assert dst_ckpt["extra"]["growth_metadata"]["source_step"] == 1234

    def test_expansion_factor_must_be_integer_geq_2(self, tmp_path):
        """Non-integer or <2 expansion factor should be rejected before any
        IO happens — the "split downstream weights" math only works cleanly
        for integer ratios, and a factor of 1 is a no-op that probably
        indicates a caller bug."""
        src_path = os.path.join(tmp_path, "src.pt")
        # Don't bother writing a real source — error fires before IO.
        for bad in (1, 0, -1, 1.5, 2.5):
            with pytest.raises(ValueError, match="expansion_factor"):
                grow_moe_checkpoint(
                    src_path=src_path,
                    dst_path=os.path.join(tmp_path, "dst.pt"),
                    expansion_factor=bad,
                )
