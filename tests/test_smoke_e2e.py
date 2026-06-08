"""
End-to-end smoke test for all Phase 2B + 2C components.

Verifies that every new utility can be instantiated with a small model
and runs without error. No quality assertions — just "it doesn't crash
and shapes are sane".
"""

import torch
import pytest
from mythouro.main import MythOuro, MythOuroConfig, RecurrentBlock


# ---------------------------------------------------------------------------
# Shared small config (matches test_main.py conventions)
# ---------------------------------------------------------------------------

B, T = 2, 8


def _small_cfg(**overrides) -> MythOuroConfig:
    defaults = dict(
        vocab_size=200,
        dim=64,
        n_heads=4,
        n_kv_heads=2,
        max_seq_len=32,
        max_loop_iters=3,
        prelude_layers=1,
        coda_layers=1,
        attn_type="gqa",
        n_experts=4,
        n_shared_experts=1,
        n_experts_per_tok=2,
        expert_dim=16,
        act_threshold=0.99,
        lora_rank=4,
        kv_lora_rank=16,
        q_lora_rank=32,
        qk_rope_head_dim=8,
        qk_nope_head_dim=8,
        v_head_dim=8,
    )
    defaults.update(overrides)
    return MythOuroConfig(**defaults)


@pytest.fixture
def model():
    cfg = _small_cfg()
    m = MythOuro(cfg)
    m.eval()
    return m


@pytest.fixture
def ids():
    return torch.randint(0, 200, (B, T))


# ===========================================================================
# Phase 2B — Training utilities smoke tests
# ===========================================================================


class TestTrainingUtils:
    """Smoke tests for training_utils.py Part 2 components."""

    def test_collect_router_logits(self, model, ids):
        from mythouro.training_utils import collect_router_logits

        with torch.no_grad():
            model(ids)
        buf = collect_router_logits(model)
        # There should be at least one MoE layer in the recurrent block
        assert len(buf) >= 1
        for logits in buf:
            assert logits.ndim == 2  # (N, E)

    def test_load_balance_loss(self, model, ids):
        from mythouro.training_utils import collect_router_logits, load_balance_loss

        with torch.no_grad():
            model(ids)
        buf = collect_router_logits(model)
        loss = load_balance_loss(buf, topk=model.cfg.n_experts_per_tok)
        assert loss.shape == ()
        assert not torch.isnan(loss)

    def test_uncertainty_calibration_loss(self, model, ids):
        from mythouro.training_utils import uncertainty_calibration_loss

        targets = torch.randint(0, 200, (B, T))
        with torch.no_grad():
            logits, unc = model(ids)
        loss = uncertainty_calibration_loss(logits, unc, targets)
        assert loss.shape == ()
        assert not torch.isnan(loss)

    def test_combined_loss(self, model, ids):
        from mythouro.training_utils import combined_loss

        targets = torch.randint(0, 200, (B, T))
        logits, unc = model(ids)
        total, metrics = combined_loss(
            model, logits, unc, targets,
            vocab_size=200, topk=model.cfg.n_experts_per_tok,
        )
        assert total.shape == ()
        assert not torch.isnan(total)
        assert "ce" in metrics
        assert "lb" in metrics
        assert "unc" in metrics

    def test_loop_curriculum(self):
        from mythouro.training_utils import LoopCurriculum

        cur = LoopCurriculum(start_loops=1, max_loops=16, warmup_steps=100, total_steps=1000)
        assert cur.get(0) == 1
        assert cur.get(50) == 1
        assert cur.get(1000) == 16
        assert 1 <= cur.get(500) <= 16

    def test_log_spectral_radius(self, model):
        from mythouro.training_utils import log_spectral_radius

        # Should not crash
        log_spectral_radius(model, step=0)

    def test_contrastive_loop_loss(self, model, ids):
        from mythouro.training_utils import contrastive_loop_loss

        targets = torch.randint(0, 200, (B, T))
        loss = contrastive_loop_loss(model, ids, targets, n_loops_low=1, n_loops_high=3)
        assert loss.shape == ()
        assert not torch.isnan(loss)

    def test_process_reward_head(self):
        from mythouro.training_utils import ProcessRewardHead, process_reward_loss

        prm = ProcessRewardHead(dim=64)
        hidden = torch.randn(B, T, 64)
        logits = torch.randn(B, T, 200)
        targets = torch.randint(0, 200, (B, T))

        loss = process_reward_loss(prm, hidden, logits, targets)
        assert loss.shape == ()
        assert not torch.isnan(loss)

    def test_loop_depth_annealer(self):
        from mythouro.training_utils import LoopDepthAnnealer

        ann = LoopDepthAnnealer(base_loops=8, max_extra_loops=32, anneal_start=800, total_steps=1000)
        assert ann.get(0) == 8
        assert ann.get(799) == 8
        assert ann.get(1000) == 32
        assert 8 <= ann.get(900) <= 32

    def test_sparse_activation_loss(self, model, ids):
        from mythouro.training_utils import collect_router_logits, sparse_activation_loss

        with torch.no_grad():
            model(ids)
        buf = collect_router_logits(model)
        loss = sparse_activation_loss(buf)
        assert loss.shape == ()
        assert not torch.isnan(loss)

    def test_expert_specialization_probe(self):
        from mythouro.training_utils import ExpertSpecializationProbe

        probe = ExpertSpecializationProbe(n_experts=4, n_domains=4)
        logits = torch.randn(16, 4)  # 16 tokens, 4 experts
        domain_logits = probe(logits)
        assert domain_logits.shape == (16, 4, 4)

        labels = torch.randint(0, 4, (16,))
        loss = probe.loss([logits], labels)
        assert loss.shape == ()

        domains = probe.predict_expert_domains()
        assert len(domains) == 4

    def test_get_domain_labels(self):
        from mythouro.training_utils import get_domain_labels

        texts = [
            "def foo(): pass",        # code
            "the proof of theorem",    # math
            "### User: hello",         # instruction
            "the cat sat on the mat",  # general
        ]
        labels = get_domain_labels(texts, torch.device("cpu"))
        assert labels.shape == (4,)
        assert labels[0].item() == 1  # code
        assert labels[1].item() == 2  # math
        assert labels[2].item() == 3  # instruction
        assert labels[3].item() == 0  # general


# ===========================================================================
# Phase 2C — Inference utilities smoke tests
# ===========================================================================


class TestInferenceUtils:
    """Smoke tests for inference.py Part 2 components."""

    def test_uncertainty_gated_generator(self, model, ids):
        from mythouro.inference import UncertaintyGatedGenerator

        gen = UncertaintyGatedGenerator(model, min_loops=1, max_loops=3, threshold=0.5)
        out = gen.generate(ids, max_new_tokens=4, temperature=1.0, top_k=10)
        assert out.shape == (B, T + 4)

    def test_speculative_decoder(self, model, ids):
        from mythouro.inference import SpeculativeDecoder

        dec = SpeculativeDecoder(model, draft_loops=1, verify_loops=3, K=2, temperature=1.0)
        out = dec.generate(ids, max_new_tokens=4)
        assert out.shape[0] == B
        assert out.shape[1] >= T  # at least the prompt

    def test_cross_loop_kv_cache(self, model, ids):
        from mythouro.inference import CrossLoopKVCache

        cl = CrossLoopKVCache(share_after=2)
        with torch.no_grad():
            model(ids, kv_cache=cl.cache, start_pos=0)
        assert len(cl.cache) > 0
        pre_bytes = cl.memory_bytes()
        assert pre_bytes > 0
        cl.compress()
        post_bytes = cl.memory_bytes()
        # After compression, cache should be no larger
        assert post_bytes <= pre_bytes

    def test_compress_kv_cache(self, model, ids):
        from mythouro.inference import compress_kv_cache

        cache = {}
        with torch.no_grad():
            model(ids, kv_cache=cache, start_pos=0)
        compressed = compress_kv_cache(cache, share_after=2)
        assert isinstance(compressed, dict)
        # Non-recurrent keys should be preserved
        for k in cache:
            if not k.startswith("recurrent_loop_"):
                assert k in compressed

    def test_component_grad_norm_logger(self, model, ids):
        from mythouro.inference import ComponentGradNormLogger

        targets = torch.randint(0, 200, (B, T))
        logits, _ = model(ids)
        loss = torch.nn.functional.cross_entropy(logits.view(-1, 200), targets.view(-1))
        loss.backward()
        norms = ComponentGradNormLogger.compute(model)
        assert isinstance(norms, dict)
        assert len(norms) > 0
        # At least some groups should have non-zero norms
        assert any(v > 0 for v in norms.values())

    def test_continuous_depthwise_batcher(self, model, ids):
        from mythouro.inference import ContinuousDepthwiseBatcher

        batcher = ContinuousDepthwiseBatcher(model)
        logits, unc = batcher.forward(ids, n_loops=3)
        assert logits.shape == (B, T, model.cfg.vocab_size)
        assert unc.shape == (B, T)
        assert not torch.isnan(logits).any()
        assert not torch.isnan(unc).any()

    def test_cot_distillation_trainer(self, model, ids):
        from mythouro.inference import CoTDistillationTrainer

        trainer = CoTDistillationTrainer(model, dim_match_coeff=0.1)
        question_ids = ids
        answer_ids = torch.randint(0, 200, (B, 4))
        # Create mock CoT embeddings for each loop step
        cot_embeddings = [torch.randn(B, model.cfg.dim) for _ in range(3)]

        total_loss, metrics = trainer.loss(question_ids, cot_embeddings, answer_ids)
        assert total_loss.shape == ()
        assert not torch.isnan(total_loss)
        assert "answer_loss" in metrics
        assert "dim_match_loss" in metrics

    def test_activation_offloader(self, model):
        from mythouro.inference import ActivationOffloader

        off = ActivationOffloader(model)
        assert not off.enabled
        off.enable()
        assert off.enabled
        assert len(off._hooks) > 0
        off.disable()
        assert not off.enabled
        assert len(off._hooks) == 0

    def test_apply_int8_quantization(self, model):
        from mythouro.inference import apply_int8_quantization

        # Quantize and verify the model still has a forward method
        qmodel = apply_int8_quantization(model)
        assert qmodel is not None
        # Forward should still work on quantized model
        ids = torch.randint(0, 200, (1, 4))
        with torch.no_grad():
            logits, unc = qmodel(ids)
        assert logits.shape == (1, 4, model.cfg.vocab_size)

    def test_quantization_aware_training_hooks(self, model):
        from mythouro.inference import quantization_aware_training_hooks

        handles = quantization_aware_training_hooks(model)
        assert len(handles) > 0
        # Run a forward to trigger the hooks
        ids = torch.randint(0, 200, (1, 4))
        with torch.no_grad():
            model(ids)
        # Clean up
        for h in handles:
            h.remove()

    def test_convenience_aliases(self, model, ids):
        from mythouro.inference import speculative_generate, uncertainty_gated_generate

        out1 = speculative_generate(model, ids, max_new_tokens=2, K=2, draft_loops=1, verify_loops=2)
        assert out1.shape[0] == B

        out2 = uncertainty_gated_generate(model, ids, max_new_tokens=2, min_loops=1, max_loops=2)
        assert out2.shape == (B, T + 2)


# ===========================================================================
# End-to-end: full pipeline smoke test
# ===========================================================================


class TestEndToEnd:
    """Full pipeline: build model → forward → loss → backward → generate."""

    def test_full_training_step(self):
        cfg = _small_cfg()
        model = MythOuro(cfg)
        model.train()

        ids = torch.randint(0, cfg.vocab_size, (B, T))
        targets = torch.randint(0, cfg.vocab_size, (B, T))

        from mythouro.training_utils import combined_loss

        logits, unc = model(ids, n_loops=2)
        total, metrics = combined_loss(
            model, logits, unc, targets,
            vocab_size=cfg.vocab_size, topk=cfg.n_experts_per_tok,
        )
        total.backward()

        # Verify gradients exist on key parameters
        assert model.embed.weight.grad is not None
        assert model.recurrent.injection.log_A.grad is not None

        # Verify metrics are reasonable
        assert metrics["ce"] > 0
        assert not any(v != v for v in metrics.values())  # no NaN

    def test_full_inference_pipeline(self):
        """Build → forward → uncertainty gate → speculative decode → quantize."""
        from mythouro.inference import (
            UncertaintyGatedGenerator,
            CrossLoopKVCache,
            apply_int8_quantization,
        )

        cfg = _small_cfg()
        model = MythOuro(cfg)
        model.eval()
        ids = torch.randint(0, cfg.vocab_size, (1, T))

        # 1. Standard forward
        logits, unc = model(ids)
        assert logits.shape == (1, T, cfg.vocab_size)

        # 2. Standard generate
        gen = model.generate(ids, max_new_tokens=4, n_loops=2)
        assert gen.shape == (1, T + 4)

        # 3. Uncertainty-gated generate
        ugg = UncertaintyGatedGenerator(model, min_loops=1, max_loops=3)
        gen2 = ugg.generate(ids, max_new_tokens=4)
        assert gen2.shape == (1, T + 4)

        # 4. CrossLoopKVCache
        cl = CrossLoopKVCache(share_after=2)
        with torch.no_grad():
            model(ids, kv_cache=cl.cache)
        cl.compress()
        assert cl.memory_bytes() > 0

        # 5. Quantize and forward
        qmodel = apply_int8_quantization(model)
        with torch.no_grad():
            qlogits, qunc = qmodel(ids)
        assert qlogits.shape == (1, T, cfg.vocab_size)

    def test_mla_inference_pipeline(self):
        """Same pipeline with MLA attention type."""
        cfg = _small_cfg(attn_type="mla")
        model = MythOuro(cfg)
        model.eval()
        ids = torch.randint(0, cfg.vocab_size, (1, T))

        logits, unc = model(ids)
        assert logits.shape == (1, T, cfg.vocab_size)

        gen = model.generate(ids, max_new_tokens=4, n_loops=2)
        assert gen.shape == (1, T + 4)


if __name__ == "__main__":
    pytest.main([__file__, "--verbose"])
