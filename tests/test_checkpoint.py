"""
Tests for the checkpoint subsystem in `mythouro.checkpointing` (§4).

These pin the §4 contract:

    1. save → load preserves model state, optimizer state, and step.
    2. Loader rejects a version mismatch by default; honours
       allow_version_mismatch=True.
    3. Loader rejects a shape-incompatible cfg (raises RuntimeError).
    4. Loader accepts benign cfg drift (LR, ratios) and logs it.
    5. RNG state round-trips (single-process path).
    6. ShutdownHandler sets `requested=True` on signal.

The helpers live in `mythouro.checkpointing` rather than
`training/3b_fine_web_edu.py` so the test doesn't have to drag in
`datasets`/`pandas` (which segfaults during import collection on
Python 3.14 + Windows).
"""

from __future__ import annotations

import signal

import pytest
import torch

from mythouro.main import MythOuro, MythOuroConfig
from mythouro import checkpointing as _train


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _tiny_cfg(**overrides) -> MythOuroConfig:
    base = dict(
        vocab_size=64,
        dim=32,
        n_heads=4,
        n_kv_heads=2,
        max_seq_len=32,
        max_loop_iters=2,
        prelude_layers=1,
        coda_layers=1,
        attn_type="gqa",
        n_experts=4,
        n_shared_experts=1,
        n_experts_per_tok=2,
        expert_dim=16,
        lora_rank=4,
        kv_lora_rank=16,
        q_lora_rank=16,
        qk_rope_head_dim=8,
        qk_nope_head_dim=8,
        v_head_dim=8,
    )
    base.update(overrides)
    return MythOuroConfig(**base)


def _build_model_and_opt(cfg):
    model = MythOuro(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    # Take one step so the optimizer has non-trivial moments to round-trip.
    ids = torch.randint(0, cfg.vocab_size, (2, 8))
    logits, _ = model(ids)
    logits.sum().backward()
    opt.step()
    opt.zero_grad()
    return model, opt


# ---------------------------------------------------------------------------
# Checkpoint roundtrip
# ---------------------------------------------------------------------------


class TestCheckpointRoundtrip:
    def test_model_state_preserved(self, tmp_path):
        cfg = _tiny_cfg()
        model, opt = _build_model_and_opt(cfg)
        _train.save_checkpoint(
            model, opt, step=42, cfg=cfg, vocab_size=cfg.vocab_size,
            ckpt_dir=str(tmp_path), ddp=False, master=True,
        )
        # Mutate model weights to ensure load actually overwrites them.
        with torch.no_grad():
            for p in model.parameters():
                p.add_(1.0)

        ckpt_files = _train.list_ckpts(str(tmp_path))
        assert len(ckpt_files) == 1
        step, extra = _train.load_checkpoint(
            model, opt, ckpt_files[0], ddp=False, current_cfg=cfg,
        )
        assert step == 42
        assert extra == {}

        # Re-save into a SECOND directory and compare keys+shapes
        # rather than chasing bit-exactness through pickle.
        model2, opt2 = _build_model_and_opt(cfg)
        _train.load_checkpoint(
            model2, opt2, ckpt_files[0], ddp=False, current_cfg=cfg,
        )
        for (n1, p1), (n2, p2) in zip(
            model.named_parameters(), model2.named_parameters(),
        ):
            assert n1 == n2
            assert torch.allclose(p1, p2)

    def test_optimizer_state_preserved(self, tmp_path):
        cfg = _tiny_cfg()
        model, opt = _build_model_and_opt(cfg)
        # Capture an Adam first-moment slot for spot-check.
        first_param = next(model.parameters())
        opt_state_before = opt.state[first_param]["exp_avg"].clone()

        _train.save_checkpoint(
            model, opt, step=1, cfg=cfg, vocab_size=cfg.vocab_size,
            ckpt_dir=str(tmp_path), ddp=False, master=True,
        )

        # Rebuild from scratch
        cfg2 = _tiny_cfg()
        model2 = MythOuro(cfg2)
        opt2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
        ckpt_files = _train.list_ckpts(str(tmp_path))
        _train.load_checkpoint(
            model2, opt2, ckpt_files[0], ddp=False, current_cfg=cfg2,
        )
        # First param of the fresh model now has the restored Adam state.
        first_param2 = next(model2.parameters())
        assert torch.allclose(
            opt2.state[first_param2]["exp_avg"], opt_state_before
        )

    def test_router_bias_buffer_preserved(self, tmp_path):
        # router_bias is a non-gradient buffer — must survive the roundtrip.
        cfg = _tiny_cfg()
        model, opt = _build_model_and_opt(cfg)
        moe = model.recurrent.block.ffn
        with torch.no_grad():
            moe.router_bias.fill_(0.5)
        _train.save_checkpoint(
            model, opt, step=5, cfg=cfg, vocab_size=cfg.vocab_size,
            ckpt_dir=str(tmp_path), ddp=False, master=True,
        )
        model2 = MythOuro(_tiny_cfg())
        opt2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
        _train.load_checkpoint(
            model2, opt2,
            _train.list_ckpts(str(tmp_path))[0],
            ddp=False, current_cfg=_tiny_cfg(),
        )
        assert torch.allclose(
            model2.recurrent.block.ffn.router_bias,
            torch.full_like(model2.recurrent.block.ffn.router_bias, 0.5),
        )


# ---------------------------------------------------------------------------
# Version + cfg compatibility checks
# ---------------------------------------------------------------------------


class TestCheckpointCompat:
    def test_version_mismatch_raises_by_default(self, tmp_path):
        cfg = _tiny_cfg()
        model, opt = _build_model_and_opt(cfg)
        _train.save_checkpoint(
            model, opt, step=1, cfg=cfg, vocab_size=cfg.vocab_size,
            ckpt_dir=str(tmp_path), ddp=False, master=True,
        )
        ckpt_path = _train.list_ckpts(str(tmp_path))[0]

        # Mutate the file's version field on disk
        payload = torch.load(ckpt_path, weights_only=False)
        payload["checkpoint_version"] = 999
        torch.save(payload, ckpt_path)

        model2 = MythOuro(_tiny_cfg())
        opt2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
        with pytest.raises(RuntimeError, match="version mismatch"):
            _train.load_checkpoint(model2, opt2, ckpt_path, ddp=False,
                                    current_cfg=_tiny_cfg())

    def test_version_mismatch_override_allows_load(self, tmp_path):
        cfg = _tiny_cfg()
        model, opt = _build_model_and_opt(cfg)
        _train.save_checkpoint(
            model, opt, step=1, cfg=cfg, vocab_size=cfg.vocab_size,
            ckpt_dir=str(tmp_path), ddp=False, master=True,
        )
        ckpt_path = _train.list_ckpts(str(tmp_path))[0]
        payload = torch.load(ckpt_path, weights_only=False)
        payload["checkpoint_version"] = 999
        torch.save(payload, ckpt_path)

        model2 = MythOuro(_tiny_cfg())
        opt2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
        step, _ = _train.load_checkpoint(
            model2, opt2, ckpt_path, ddp=False, current_cfg=_tiny_cfg(),
            allow_version_mismatch=True,
        )
        assert step == 1

    def test_incompatible_cfg_raises(self, tmp_path):
        cfg = _tiny_cfg(dim=32)
        model, opt = _build_model_and_opt(cfg)
        _train.save_checkpoint(
            model, opt, step=1, cfg=cfg, vocab_size=cfg.vocab_size,
            ckpt_dir=str(tmp_path), ddp=False, master=True,
        )
        ckpt_path = _train.list_ckpts(str(tmp_path))[0]
        # Build a model with a different shape-affecting field
        cfg2 = _tiny_cfg(dim=64)
        model2 = MythOuro(cfg2)
        opt2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
        with pytest.raises(RuntimeError, match="shape-incompatible"):
            _train.load_checkpoint(
                model2, opt2, ckpt_path, ddp=False, current_cfg=cfg2,
            )

    def test_benign_cfg_drift_allowed(self, tmp_path):
        # dropout is NOT in _SHAPE_FIELDS, so changing it between
        # save and load must not raise.
        cfg = _tiny_cfg(dropout=0.0)
        model, opt = _build_model_and_opt(cfg)
        _train.save_checkpoint(
            model, opt, step=1, cfg=cfg, vocab_size=cfg.vocab_size,
            ckpt_dir=str(tmp_path), ddp=False, master=True,
        )
        ckpt_path = _train.list_ckpts(str(tmp_path))[0]
        cfg2 = _tiny_cfg(dropout=0.1)
        model2 = MythOuro(cfg2)
        opt2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
        step, _ = _train.load_checkpoint(
            model2, opt2, ckpt_path, ddp=False, current_cfg=cfg2,
        )
        assert step == 1


# ---------------------------------------------------------------------------
# RNG state roundtrip
# ---------------------------------------------------------------------------


class TestRNGRoundtrip:
    def test_rng_state_restored(self, tmp_path):
        cfg = _tiny_cfg()
        model, opt = _build_model_and_opt(cfg)

        # Seed deterministically, draw one number, save.
        torch.manual_seed(123)
        _train.save_checkpoint(
            model, opt, step=1, cfg=cfg, vocab_size=cfg.vocab_size,
            ckpt_dir=str(tmp_path), ddp=False, master=True,
        )
        expected_after_save = torch.randn(3)

        # Disturb the global RNG, then load — the saved RNG state should
        # reset us so the same draw produces the same numbers.
        torch.manual_seed(999)
        torch.randn(1000)
        model2 = MythOuro(_tiny_cfg())
        opt2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
        _train.load_checkpoint(
            model2, opt2,
            _train.list_ckpts(str(tmp_path))[0],
            ddp=False, current_cfg=_tiny_cfg(),
        )
        actual_after_load = torch.randn(3)
        assert torch.equal(actual_after_load, expected_after_save)


# ---------------------------------------------------------------------------
# ShutdownHandler
# ---------------------------------------------------------------------------


class TestShutdownHandler:
    def test_initial_state_idle(self):
        h = _train.ShutdownHandler()
        assert h.requested is False

    def test_signal_sets_requested_flag(self):
        h = _train.ShutdownHandler()
        # We don't install real handlers in the test (pytest's own
        # SIGINT handler complicates that on Windows); instead invoke the
        # internal method to verify the flag flips.
        h._on_signal(signal.SIGINT, None)
        assert h.requested is True

    def test_second_signal_raises(self):
        h = _train.ShutdownHandler()
        h._on_signal(signal.SIGINT, None)
        with pytest.raises(KeyboardInterrupt):
            h._on_signal(signal.SIGINT, None)
