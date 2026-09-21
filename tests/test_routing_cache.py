"""Activation checkpointing must not recompute MoE routing decisions.

2026-09-21: the large (dim 2048) profile died in backward with
torch.utils.checkpoint's "Recomputed values ... different metadata". The loop
body is checkpointed, upstream bf16 ops on XPU are not bitwise-stable across
calls, and a token near a routing tie landed in a different expert on the
recompute — MoE turns value drift into shape drift. The fix records topk_idx
per loop slot in the forward and reuses it on recompute. This test simulates
the drift with a per-call perturbation hook and checks both directions.
"""
import dataclasses
import torch
from mythouro.variants import mythouro_distill_tiny
from mythouro.main import MythOuro


def _model():
    torch.manual_seed(0)
    cfg = dataclasses.replace(mythouro_distill_tiny(), gradient_checkpointing=True,
                              dim=256, expert_dim=256, n_heads=4, n_kv_heads=2,
                              vocab_size=512, max_seq_len=64)
    m = MythOuro(cfg); m.train()
    return m, cfg


def _drift_hook(counter):
    def hook(mod, args):
        counter[0] += 1
        x = args[0]
        g = torch.Generator().manual_seed(counter[0])   # different every call
        return (x + 0.05 * torch.randn(x.shape, generator=g),)
    return hook


def test_recompute_reuses_forward_routing_under_drift():
    m, cfg = _model()
    ffn = m.recurrent.block.ffn
    counter = [0]
    h = ffn.register_forward_pre_hook(_drift_hook(counter))
    try:
        x = torch.randint(0, cfg.vocab_size, (2, 32))
        o = m(x, n_loops=cfg.max_loop_iters)
        l = (o[0] if isinstance(o, (tuple, list)) else o)
        l.float().logsumexp(-1).mean().backward()          # must not raise
    finally:
        h.remove()
    assert counter[0] == 2 * cfg.max_loop_iters, "expected one forward + one recompute per loop"
    assert len(ffn._route_cache) == cfg.max_loop_iters


def test_without_cache_the_drift_breaks_checkpointing():
    """The failure this guards against must still be reproducible, or the test
    above is not testing anything."""
    m, cfg = _model()
    ffn = m.recurrent.block.ffn
    orig = type(ffn).forward
    def no_cache(self, t):
        self._route_cache = None
        return orig(self, t)
    type(ffn).forward = no_cache
    h = ffn.register_forward_pre_hook(_drift_hook([0]))
    try:
        x = torch.randint(0, cfg.vocab_size, (2, 32))
        o = m(x, n_loops=cfg.max_loop_iters)
        l = (o[0] if isinstance(o, (tuple, list)) else o)
        try:
            l.float().logsumexp(-1).mean().backward()
            raised = False
        except torch.utils.checkpoint.CheckpointError:
            raised = True
    finally:
        h.remove(); type(ffn).forward = orig
    assert raised, "drift without the cache should reproduce the CheckpointError"


def test_no_cache_outside_checkpointed_training():
    m, cfg = _model()
    ffn = m.recurrent.block.ffn
    x = torch.randint(0, cfg.vocab_size, (1, 16))
    m.eval()
    with torch.no_grad():
        m(x, n_loops=cfg.max_loop_iters)
    assert ffn._route_cache is None
