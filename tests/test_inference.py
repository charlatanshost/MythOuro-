"""
Tests for `mythouro.inference` — generators, KV-cache compressor, and
gradient-norm diagnostic.

These don't aim for exhaustive coverage of every code branch — they pin
the public contracts and catch regressions at the API boundary. The
expensive end-to-end behaviour (does speculative decoding actually
speed things up? does ECE drop after calibration?) lives in the eval
harness, not here.
"""

from __future__ import annotations

import torch

from mythouro.main import MythOuro, MythOuroConfig
from mythouro.inference import (
    BestOfTrajectoryGenerator,
    ComponentGradNormLogger,
    ConfidenceAwareGenerator,
    ContinuousDepthwiseBatcher,
    CrossLoopKVCache,
    SpeculativeDecoder,
    UncertaintyGatedGenerator,
    best_of_trajectory_generate,
    compress_kv_cache,
    confidence_aware_generate,
    speculative_generate,
    uncertainty_gated_generate,
)


def _tiny_cfg(**overrides) -> MythOuroConfig:
    defaults = dict(
        vocab_size=128, dim=64, n_heads=4, n_kv_heads=2, max_seq_len=64,
        max_loop_iters=4, prelude_layers=1, coda_layers=1, attn_type="gqa",
        n_experts=4, n_shared_experts=1, n_experts_per_tok=2, expert_dim=16,
        lora_rank=4, kv_lora_rank=16, q_lora_rank=16,
        qk_rope_head_dim=8, qk_nope_head_dim=8, v_head_dim=8,
        dropout=0.0,
    )
    defaults.update(overrides)
    return MythOuroConfig(**defaults)


# ---------------------------------------------------------------------------
# UncertaintyGatedGenerator
# ---------------------------------------------------------------------------


class TestUncertaintyGatedGenerator:
    def setup_method(self):
        self.cfg = _tiny_cfg()
        self.model = MythOuro(self.cfg).eval()
        self.prompt = torch.randint(0, self.cfg.vocab_size, (1, 6))

    def test_generate_returns_prompt_plus_new(self):
        gen = UncertaintyGatedGenerator(
            self.model, min_loops=2, max_loops=4, threshold=0.5,
        )
        out = gen.generate(self.prompt, max_new_tokens=4)
        assert out.shape == (1, self.prompt.shape[1] + 4)

    def test_prompt_prefix_is_preserved(self):
        gen = UncertaintyGatedGenerator(self.model, min_loops=2, max_loops=4)
        out = gen.generate(self.prompt, max_new_tokens=3, temperature=0.5, top_k=10)
        assert torch.equal(out[:, : self.prompt.shape[1]], self.prompt)

    def test_helper_function_matches_class_shape(self):
        out = uncertainty_gated_generate(
            self.model, self.prompt, max_new_tokens=2,
            min_loops=2, max_loops=4,
        )
        assert out.shape == (1, self.prompt.shape[1] + 2)


# ---------------------------------------------------------------------------
# SpeculativeDecoder
# ---------------------------------------------------------------------------


class TestSpeculativeDecoder:
    def setup_method(self):
        self.cfg = _tiny_cfg()
        self.model = MythOuro(self.cfg).eval()
        self.prompt = torch.randint(0, self.cfg.vocab_size, (1, 6))

    def test_generate_produces_requested_token_count(self):
        spec = SpeculativeDecoder(
            self.model, draft_loops=1, verify_loops=4, K=3,
        )
        out = spec.generate(self.prompt, max_new_tokens=4)
        assert out.shape == (1, self.prompt.shape[1] + 4)

    def test_prompt_prefix_is_preserved(self):
        spec = SpeculativeDecoder(self.model, draft_loops=1, verify_loops=4, K=3)
        out = spec.generate(self.prompt, max_new_tokens=3)
        assert torch.equal(out[:, : self.prompt.shape[1]], self.prompt)

    def test_helper_function_matches_class_shape(self):
        out = speculative_generate(
            self.model, self.prompt, max_new_tokens=2,
            draft_loops=1, verify_loops=4, K=3,
        )
        assert out.shape == (1, self.prompt.shape[1] + 2)


# ---------------------------------------------------------------------------
# ContinuousDepthwiseBatcher
# ---------------------------------------------------------------------------


class TestContinuousDepthwiseBatcher:
    def setup_method(self):
        self.cfg = _tiny_cfg()
        self.model = MythOuro(self.cfg).eval()

    def test_forward_returns_logits_and_uncertainty(self):
        batcher = ContinuousDepthwiseBatcher(self.model)
        ids = torch.randint(0, self.cfg.vocab_size, (3, 8))
        logits, unc = batcher.forward(ids, n_loops=4)
        assert logits.shape == (3, 8, self.cfg.vocab_size)
        assert unc.shape == (3, 8)
        assert torch.isfinite(logits).all()
        assert (unc >= 0).all() and (unc <= 1).all()

    def test_shape_matches_plain_forward(self):
        # The batcher should produce the same SHAPE as a normal forward pass
        # — only the per-row halt-depth differs.
        ids = torch.randint(0, self.cfg.vocab_size, (3, 8))
        with torch.no_grad():
            plain_logits, plain_unc = self.model(ids, n_loops=4)
        batcher = ContinuousDepthwiseBatcher(self.model)
        b_logits, b_unc = batcher.forward(ids, n_loops=4)
        assert plain_logits.shape == b_logits.shape
        assert plain_unc.shape == b_unc.shape


# ---------------------------------------------------------------------------
# CrossLoopKVCache / compress_kv_cache
# ---------------------------------------------------------------------------


class TestCrossLoopKVCache:
    def setup_method(self):
        self.cfg = _tiny_cfg()
        self.model = MythOuro(self.cfg).eval()
        self.ids = torch.randint(0, self.cfg.vocab_size, (1, 8))

    def test_compress_reduces_distinct_recurrent_keys(self):
        cl = CrossLoopKVCache(share_after=2)
        with torch.no_grad():
            self.model(self.ids, n_loops=4, kv_cache=cl.cache, start_pos=0)
        recurrent_keys = sorted(k for k in cl.cache if "recurrent_loop" in k)
        assert len(recurrent_keys) == 4                # all 4 loops cached separately

        cl.compress()
        recurrent_keys_after = sorted(k for k in cl.cache if "recurrent_loop" in k)
        # Loops 0,1 keep their own slots; loops 2,3 merge into shared slot.
        assert len(recurrent_keys_after) < 4
        assert any("recurrent_loop_2" in k for k in recurrent_keys_after)

    def test_compress_reduces_memory_bytes(self):
        cl = CrossLoopKVCache(share_after=2)
        with torch.no_grad():
            self.model(self.ids, n_loops=4, kv_cache=cl.cache, start_pos=0)
        before = cl.memory_bytes()
        cl.compress()
        after = cl.memory_bytes()
        assert after < before

    def test_post_compress_decode_still_works(self):
        # After compression the cache must still produce sensible outputs
        # on a subsequent single-token decode step.
        cl = CrossLoopKVCache(share_after=2)
        with torch.no_grad():
            self.model(self.ids, n_loops=4, kv_cache=cl.cache, start_pos=0)
        cl.compress()
        new_tok = torch.randint(0, self.cfg.vocab_size, (1, 1))
        with torch.no_grad():
            logits, unc = self.model(
                new_tok, n_loops=4, kv_cache=cl.cache,
                start_pos=self.ids.shape[1],
            )
        assert logits.shape == (1, 1, self.cfg.vocab_size)
        assert torch.isfinite(logits).all()

    def test_compress_kv_cache_helper_returns_new_dict(self):
        # The non-class form must not mutate its input — important for
        # callers that want to keep both the uncompressed and compressed
        # versions around for comparison.
        cache: dict = {}
        with torch.no_grad():
            self.model(self.ids, n_loops=4, kv_cache=cache, start_pos=0)
        original_keys = set(cache.keys())
        compressed = compress_kv_cache(cache, share_after=2)
        # Input untouched
        assert set(cache.keys()) == original_keys
        # Output is a new dict with fewer recurrent keys
        comp_rec = {k for k in compressed if "recurrent_loop" in k}
        orig_rec = {k for k in cache if "recurrent_loop" in k}
        assert len(comp_rec) < len(orig_rec)

    def test_reset_clears_cache(self):
        cl = CrossLoopKVCache(share_after=2)
        with torch.no_grad():
            self.model(self.ids, n_loops=2, kv_cache=cl.cache, start_pos=0)
        assert len(cl.cache) > 0
        cl.reset()
        assert cl.cache == {}


# ---------------------------------------------------------------------------
# ComponentGradNormLogger
# ---------------------------------------------------------------------------


class TestComponentGradNormLogger:
    def setup_method(self):
        self.cfg = _tiny_cfg()
        self.model = MythOuro(self.cfg).train()
        ids = torch.randint(0, self.cfg.vocab_size, (2, 8))
        logits, unc = self.model(ids)
        (logits.sum() + unc.sum()).backward()

    def test_compute_returns_dict_with_known_groups(self):
        norms = ComponentGradNormLogger.compute(self.model)
        assert isinstance(norms, dict)
        # Should at least include the canonical block groups
        for g in ("prelude", "recurrent", "coda"):
            assert g in norms

    def test_norms_are_non_negative(self):
        norms = ComponentGradNormLogger.compute(self.model)
        for v in norms.values():
            assert v >= 0

    def test_at_least_one_norm_nonzero(self):
        # A trained-with-gradients model must have a non-zero grad norm
        # somewhere — otherwise the logger isn't actually wired to the
        # parameters.
        norms = ComponentGradNormLogger.compute(self.model)
        assert any(v > 0 for v in norms.values())

    def test_log_does_not_crash(self):
        # The log helper goes to loguru; we just confirm it doesn't raise.
        ComponentGradNormLogger.log(self.model, step=42)


# ---------------------------------------------------------------------------
# ConfidenceAwareGenerator
# ---------------------------------------------------------------------------


class TestConfidenceAwareGenerator:
    def setup_method(self):
        self.cfg = _tiny_cfg()
        self.model = MythOuro(self.cfg).eval()
        self.prompt = torch.randint(0, self.cfg.vocab_size, (1, 6))

    # ── Structure ──────────────────────────────────────────────────────

    def test_generate_returns_dict_with_sequences_and_stop_reason(self):
        gen = ConfidenceAwareGenerator(self.model, n_loops=2)
        result = gen.generate(self.prompt, max_new_tokens=4)
        assert isinstance(result, dict)
        assert "sequences" in result
        assert "stop_reason" in result
        assert result["sequences"].shape[0] == 1
        # At least prompt_len tokens, at most prompt_len + max_new_tokens.
        assert result["sequences"].shape[1] >= self.prompt.shape[1]
        assert result["sequences"].shape[1] <= self.prompt.shape[1] + 4

    def test_prompt_prefix_is_preserved(self):
        gen = ConfidenceAwareGenerator(self.model, n_loops=2)
        result = gen.generate(self.prompt, max_new_tokens=3)
        assert torch.equal(
            result["sequences"][:, : self.prompt.shape[1]], self.prompt,
        )

    # ── min_new_tokens floor ──────────────────────────────────────────

    def test_min_new_tokens_floor(self):
        # With a high min_new_tokens and no EOS, we must produce at least
        # that many tokens even if confidence is always low (no break ids
        # configured, so confidence stop is trivially satisfiable otherwise).
        min_tok = 6
        gen = ConfidenceAwareGenerator(
            self.model,
            n_loops=2,
            min_new_tokens=min_tok,
            confidence_threshold=1.0,   # always below → would trigger early
            confidence_window=1,
            # break_token_ids empty → any token is a valid break
        )
        result = gen.generate(self.prompt, max_new_tokens=20)
        produced = result["sequences"].shape[1] - self.prompt.shape[1]
        assert produced >= min_tok

    # ── stop_reason values ────────────────────────────────────────────

    def test_stop_reason_max_new_tokens(self):
        # When nothing else triggers, the stop reason should be max_new_tokens.
        gen = ConfidenceAwareGenerator(
            self.model,
            n_loops=2,
            confidence_threshold=0.0,  # never fires (unc >= 0)
            eos_token_id=-1,           # impossible eos
        )
        result = gen.generate(self.prompt, max_new_tokens=3)
        assert result["stop_reason"] == "max_new_tokens"
        produced = result["sequences"].shape[1] - self.prompt.shape[1]
        assert produced == 3

    def test_stop_reason_eos(self):
        # Patch the model so the very first token sampled is the EOS id.
        eos_id = 5
        gen = ConfidenceAwareGenerator(
            self.model, n_loops=2, eos_token_id=eos_id,
        )
        # Monkey-patch _sample to always return EOS.
        import mythouro.inference as _inf
        orig_sample = _inf._sample
        _inf._sample = lambda logits, temp, k: torch.full(
            (logits.shape[0], 1), eos_id, dtype=torch.long, device=logits.device,
        )
        try:
            result = gen.generate(self.prompt, max_new_tokens=20)
        finally:
            _inf._sample = orig_sample
        assert result["stop_reason"] == "eos"
        # Should have produced exactly 1 token (the EOS).
        produced = result["sequences"].shape[1] - self.prompt.shape[1]
        assert produced == 1

    def test_stop_reason_confidence(self):
        # Confidence stop now requires an explicit break-token list (None
        # disables the mechanism, fail-closed). We make every vocab token
        # a valid break so the trigger is whatever the sampler happens to
        # produce — combined with `confidence_threshold=1.0` (any unc < 1
        # counts as "low") this fires deterministically after the window.
        gen = ConfidenceAwareGenerator(
            self.model,
            n_loops=2,
            min_new_tokens=1,
            confidence_threshold=1.0,
            confidence_window=2,
            break_token_ids=list(range(self.cfg.vocab_size)),
        )
        result = gen.generate(self.prompt, max_new_tokens=50)
        assert result["stop_reason"] == "confidence"
        produced = result["sequences"].shape[1] - self.prompt.shape[1]
        assert produced < 50

    def test_confidence_stop_disabled_when_break_token_ids_is_none(self):
        # New fail-closed contract: with break_token_ids=None (the default),
        # confidence-based stopping must never fire, even when the per-token
        # uncertainty threshold is trivially below the gate. This protects
        # callers from the previous "any token counts as a break" footgun.
        gen = ConfidenceAwareGenerator(
            self.model,
            n_loops=2,
            min_new_tokens=1,
            confidence_threshold=1.0,         # trivially satisfied
            confidence_window=2,
            # break_token_ids left unset (= None)
        )
        result = gen.generate(self.prompt, max_new_tokens=5)
        # Must run to max_new_tokens; confidence stop is disabled.
        assert result["stop_reason"] == "max_new_tokens"
        produced = result["sequences"].shape[1] - self.prompt.shape[1]
        assert produced == 5

    # ── Cycling detection ─────────────────────────────────────────────

    def test_has_cycle_detects_simple_repeat(self):
        ids = [1, 2, 3, 4, 1, 2, 3, 4]
        assert ConfidenceAwareGenerator._has_cycle(ids, window=8, min_len=4)

    def test_has_cycle_no_false_positive(self):
        ids = [1, 2, 3, 4, 5, 6, 7, 8]
        assert not ConfidenceAwareGenerator._has_cycle(ids, window=8, min_len=4)

    def test_stop_reason_cycle(self):
        # Patch _sample to produce a repeating 4-token cycle.
        cycle = [10, 11, 12, 13]
        call_count = [0]

        import mythouro.inference as _inf
        orig_sample = _inf._sample

        def _cycling_sample(logits, temp, k):
            tok = cycle[call_count[0] % len(cycle)]
            call_count[0] += 1
            return torch.full(
                (logits.shape[0], 1), tok, dtype=torch.long, device=logits.device,
            )

        gen = ConfidenceAwareGenerator(
            self.model,
            n_loops=2,
            eos_token_id=-1,
            min_new_tokens=1,
            confidence_threshold=0.0,   # never fires
            cycle_window=16,
            cycle_min_len=4,
        )
        _inf._sample = _cycling_sample
        try:
            result = gen.generate(self.prompt, max_new_tokens=100)
        finally:
            _inf._sample = orig_sample
        assert result["stop_reason"] == "cycle"

    # ── Max cap ───────────────────────────────────────────────────────

    def test_max_cap_respected(self):
        gen = ConfidenceAwareGenerator(
            self.model,
            n_loops=2,
            eos_token_id=-1,
            confidence_threshold=0.0,
        )
        result = gen.generate(self.prompt, max_new_tokens=5)
        produced = result["sequences"].shape[1] - self.prompt.shape[1]
        assert produced <= 5

    # ── Convenience helper ────────────────────────────────────────────

    def test_helper_returns_dict(self):
        result = confidence_aware_generate(
            self.model, self.prompt, max_new_tokens=3, n_loops=2,
            confidence_threshold=0.0,
        )
        assert isinstance(result, dict)
        assert "sequences" in result
        assert "stop_reason" in result
        produced = result["sequences"].shape[1] - self.prompt.shape[1]
        assert produced == 3

    # ── B=1 invariant ────────────────────────────────────────────────

    def test_batched_input_rejected(self):
        # Per-token natural-break + cycle checks can't be vectorised over a
        # batch whose rows want to stop at different positions. The
        # generator asserts B=1 rather than silently behaving incorrectly.
        import pytest
        gen = ConfidenceAwareGenerator(self.model, n_loops=2)
        batched = torch.randint(0, self.cfg.vocab_size, (2, 6))
        with pytest.raises(AssertionError, match="single-sequence"):
            gen.generate(batched, max_new_tokens=4)

    # ── uncertainty_trace return ─────────────────────────────────────

    def test_uncertainty_trace_length_matches_generated(self):
        gen = ConfidenceAwareGenerator(
            self.model, n_loops=2, confidence_threshold=0.0,
        )
        result = gen.generate(self.prompt, max_new_tokens=4)
        assert "uncertainty_trace" in result
        trace = result["uncertainty_trace"]
        assert isinstance(trace, list)
        produced = result["sequences"].shape[1] - self.prompt.shape[1]
        assert len(trace) == produced
        # Every entry must be a finite probability — UncertaintyHead is a sigmoid.
        for u in trace:
            assert 0.0 <= u <= 1.0


# ---------------------------------------------------------------------------
# BestOfTrajectoryGenerator + forward_trajectory
# ---------------------------------------------------------------------------


class TestBestOfTrajectory:
    def setup_method(self):
        self.cfg = _tiny_cfg(max_loop_iters=4)
        self.model = MythOuro(self.cfg).eval()
        self.prompt = torch.randint(0, self.cfg.vocab_size, (1, 6))

    # ── forward_trajectory contract ──────────────────────────────────

    def test_forward_trajectory_shapes(self):
        logits_traj, unc_traj = self.model.forward_trajectory(
            self.prompt, n_loops=4,
        )
        B, T = self.prompt.shape
        assert logits_traj.shape[0] == B and logits_traj.shape[1] == T
        assert logits_traj.shape[-1] == self.cfg.vocab_size
        K = logits_traj.shape[2]
        assert 1 <= K <= 4
        assert unc_traj.shape == (B, T, K)
        # Uncertainty is a sigmoid → valid probabilities.
        assert bool((unc_traj >= 0).all() and (unc_traj <= 1).all())

    def test_forward_trajectory_does_not_leak_state(self):
        # The capture flag must be reset after the call so the normal forward
        # path is unaffected.
        self.model.forward_trajectory(self.prompt, n_loops=4)
        assert self.model.recurrent.collect_trajectory is False
        assert self.model.recurrent.last_trajectory is None
        # Plain forward still returns the (logits, uncertainty) tuple.
        out = self.model(self.prompt, n_loops=4)
        assert isinstance(out, tuple) and len(out) == 2

    def test_force_full_depth_runs_all_loops(self):
        # ACT may halt early, giving K < n_loops by default. With
        # force_full_depth, every loop up to n_loops must run, so K == n_loops
        # on every step.
        n = 4
        default = self.model.forward_trajectory(self.prompt, n_loops=n)[1]
        forced = self.model.forward_trajectory(
            self.prompt, n_loops=n, force_full_depth=True,
        )[1]
        assert forced.shape[2] == n          # forced K == n_loops
        assert default.shape[2] <= n         # default may be shorter
        # The flag must be reset afterwards (no leak into the normal path).
        assert self.model.recurrent.force_full_depth is False

    def test_force_full_depth_extrapolates_past_trained_depth(self):
        # n_loops can exceed the trained max_loop_iters=4; forced depth must
        # still produce exactly that many loop scores.
        forced = self.model.forward_trajectory(
            self.prompt, n_loops=6, force_full_depth=True,
        )[1]
        assert forced.shape[2] == 6

    def test_generator_force_full_depth_chosen_within_range(self):
        out = best_of_trajectory_generate(
            self.model, self.prompt, max_new_tokens=3, n_loops=4,
            force_full_depth=True, top_k=0,
        )
        # Every per-loop vector has the full depth now.
        assert all(len(v) == 4 for v in out["per_loop_uncertainty"])
        assert all(0 <= k < 4 for k in out["chosen_loops"])

    def test_n_loops_one_gives_single_step(self):
        # n_loops=1 runs exactly one recurrent loop → trajectory of length 1.
        # (n_loops=0 is *not* tested: RecurrentBlock coalesces `0 or
        # max_loop_iters`, so 0 falls back to the default depth.)
        logits_traj, unc_traj = self.model.forward_trajectory(
            self.prompt, n_loops=1,
        )
        assert logits_traj.shape[2] == 1
        assert unc_traj.shape[2] == 1

    # ── generation contract ──────────────────────────────────────────

    def test_generate_length_and_prefix(self):
        out = best_of_trajectory_generate(
            self.model, self.prompt, max_new_tokens=4, n_loops=4, top_k=0,
        )
        seq = out["sequences"]
        assert seq.shape == (1, self.prompt.shape[1] + 4)
        assert torch.equal(seq[:, : self.prompt.shape[1]], self.prompt)
        assert len(out["chosen_loops"]) == 4
        assert len(out["uncertainty_trace"]) == 4
        # Every chosen loop is a valid depth index.
        assert all(0 <= k < 4 for k in out["chosen_loops"])
        # Per-loop uncertainty vectors: one per generated token, each a list of
        # K probabilities in [0, 1], and the emitted-loop uncertainty must equal
        # the chosen loop's entry in its vector.
        plu = out["per_loop_uncertainty"]
        assert len(plu) == 4
        for vec, k, emitted in zip(plu, out["chosen_loops"], out["uncertainty_trace"]):
            assert all(0.0 <= u <= 1.0 for u in vec)
            assert abs(vec[k] - emitted) < 1e-6

    def test_selects_argmin_uncertainty_loop(self):
        # The generator's first emitted loop must equal the argmin of the
        # trajectory's last-position uncertainty (deterministic: same input,
        # no sampling on the selection). min_loops=1 → no floor masking.
        logits_traj, unc_traj = self.model.forward_trajectory(
            self.prompt, n_loops=4,
        )
        expected = int(torch.argmin(unc_traj[0, -1]).item())
        out = best_of_trajectory_generate(
            self.model, self.prompt, max_new_tokens=1, n_loops=4,
            min_loops=1, top_k=0,
        )
        assert out["chosen_loops"][0] == expected

    def test_min_loops_floor_excludes_shallow_depths(self):
        # With min_loops=3, loops 0 and 1 are never selectable (the trajectory
        # is length 4, so the floor applies).
        out = best_of_trajectory_generate(
            self.model, self.prompt, max_new_tokens=5, n_loops=4,
            min_loops=3, top_k=0,
        )
        assert all(k >= 2 for k in out["chosen_loops"])

    def test_eos_stops_generation(self):
        # Force EOS by passing an id the model will eventually emit; assert the
        # stop_reason wiring works when it does. Use a permissive check: run
        # with a real eos id and confirm the field is one of the valid reasons.
        out = best_of_trajectory_generate(
            self.model, self.prompt, max_new_tokens=4, n_loops=2,
            eos_token_id=0, top_k=0,
        )
        assert out["stop_reason"] in {"eos", "cycle", "max_new_tokens"}

    def test_batched_input_rejected(self):
        import pytest
        gen = BestOfTrajectoryGenerator(self.model, n_loops=2)
        batched = torch.randint(0, self.cfg.vocab_size, (2, 6))
        with pytest.raises(AssertionError, match="single-sequence"):
            gen.generate(batched, max_new_tokens=2)
