"""
Unit tests for tools/prealloc_ut_cache._build_prealloc_class.

Tests the subclass MECHANICS on CPU against a stand-in base replicating the
stock UniversalTransformerCache's cat-semantics (the real class lives in the
dynamically-loaded HF module). The on-card KL equivalence gate
(`validate_cache_equivalence`) is the integration check and runs at harvest
startup — these tests guard the allocation/bookkeeping logic itself.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.prealloc_ut_cache import _build_prealloc_class  # noqa: E402


class StandInUTCache:
    """Minimal replica of UniversalTransformerCache's public behaviour."""

    def __init__(self, max_cache_size=None):
        self.key_cache = []
        self.value_cache = []
        self.layers = []
        self._seen_tokens = 0
        self.max_cache_size = max_cache_size

    def update(self, k, v, layer_idx, cache_kwargs=None):
        while len(self.key_cache) <= layer_idx:
            self.key_cache.append(None)
            self.value_cache.append(None)
        if self.key_cache[layer_idx] is None:
            self.key_cache[layer_idx] = k
            self.value_cache[layer_idx] = v
        else:
            self.key_cache[layer_idx] = torch.cat(
                [self.key_cache[layer_idx], k], dim=2)
            self.value_cache[layer_idx] = torch.cat(
                [self.value_cache[layer_idx], v], dim=2)
        self._seen_tokens = self.key_cache[layer_idx].shape[2]
        return self.key_cache[layer_idx], self.value_cache[layer_idx]


Prealloc = _build_prealloc_class(StandInUTCache)

B, H, D = 3, 4, 8


def _kv(t):
    return torch.randn(B, H, t, D), torch.randn(B, H, t, D)


def test_matches_cat_semantics_prefill_then_decode():
    torch.manual_seed(0)
    ref, pre = StandInUTCache(), Prealloc(max_len=64)
    # Prefill T=48 then 10 one-token decode steps, two entries (layer 0 / 5).
    for layer in (0, 5):
        k, v = _kv(48)
        rk, rv = ref.update(k.clone(), v.clone(), layer)
        pk, pv = pre.update(k.clone(), v.clone(), layer)
        assert torch.equal(rk, pk) and torch.equal(rv, pv)
    for _ in range(10):
        for layer in (0, 5):
            k, v = _kv(1)
            rk, rv = ref.update(k.clone(), v.clone(), layer)
            pk, pv = pre.update(k.clone(), v.clone(), layer)
            assert torch.equal(rk, pk) and torch.equal(rv, pv)
    assert pre._seen_tokens == ref._seen_tokens == 58


def test_returned_views_track_growth_without_copy_blowup():
    pre = Prealloc(max_len=32)
    k, v = _kv(8)
    pk, _ = pre.update(k, v, 0)
    assert pk.shape[2] == 8
    k1, v1 = _kv(1)
    pk1, _ = pre.update(k1, v1, 0)
    assert pk1.shape[2] == 9
    # Underlying storage is the same preallocated buffer (no cat reallocation).
    assert pk1.data_ptr() == pre._bufs_k[0].data_ptr()


def test_overflow_raises():
    pre = Prealloc(max_len=8)
    k, v = _kv(8)
    pre.update(k, v, 0)
    with pytest.raises(RuntimeError, match="overflow"):
        pre.update(*_kv(1), 0)


def test_max_cache_size_enforced():
    pre = Prealloc(max_len=8, max_cache_size=2)
    pre.update(*_kv(4), 0)
    with pytest.raises(IndexError):
        pre.update(*_kv(4), 5)


def test_reorder_cache_batch_indexing():
    pre = Prealloc(max_len=16)
    k, v = _kv(4)
    pre.update(k.clone(), v.clone(), 0)
    idx = torch.tensor([2, 0])
    pre.reorder_cache(idx)
    assert pre.key_cache[0].shape[0] == 2
    assert torch.equal(pre.key_cache[0], k[idx][:, :, :4])
