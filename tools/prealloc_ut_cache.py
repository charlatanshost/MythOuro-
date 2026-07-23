"""
Preallocated KV cache for the Ouro teacher (harvest_speedup_plan.md lever #2).

The stock `UniversalTransformerCache.update()` grows every entry with
`torch.cat` each decode step, transiently holding old+new — ~2× the cache at
peak. That transient is what capped the harvest at batch 24 × 768 tokens on
48 GB (batch 48 OOMs). Preallocating each entry to its final size and writing
in place removes the transient entirely → the same memory budget fits batch
~32–40.

The Ouro cache class lives in the dynamically-loaded HF module
(`trust_remote_code`), so it cannot be imported statically. `make_prealloc_cache`
locates it from the loaded teacher at runtime and builds a subclass — semantics
(per-UT-loop slots, view returns, seq-length bookkeeping) are preserved; only
the allocation strategy changes.

⚠ Correctness-gated by policy (the 2026-07-16 cached-decode lesson): call
`validate_cache_equivalence(...)` before real use — it greedy-decodes with the
prealloc cache vs the uncached full-recompute reference and requires argmax
match + KL under tolerance, mirroring `_validate_teacher_cache`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mythouro.training_utils import (  # noqa: E402
    teacher_logits,
    teacher_logits_cached,
)


def _build_prealloc_class(base_cls):
    """Subclass `base_cls` (a UniversalTransformerCache-compatible cache) with
    preallocated, slice-written entries. Split out for unit-testing against a
    stand-in base."""

    class PreallocUTCache(base_cls):
        def __init__(self, max_len: int, max_cache_size=None):
            super().__init__(max_cache_size=max_cache_size)
            self._max_len = max_len
            self._bufs_k: list = []
            self._bufs_v: list = []
            self._cursors: list = []

        def _ensure_entry(self, layer_idx: int, k: torch.Tensor):
            while len(self._bufs_k) <= layer_idx:
                self._bufs_k.append(None)
                self._bufs_v.append(None)
                self._cursors.append(0)
            if self._bufs_k[layer_idx] is None:
                B, H, _, D = k.shape
                self._bufs_k[layer_idx] = torch.zeros(
                    B, H, self._max_len, D, dtype=k.dtype, device=k.device)
                self._bufs_v[layer_idx] = torch.zeros(
                    B, H, self._max_len, D, dtype=k.dtype, device=k.device)

        def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
            if layer_idx < 0:
                raise ValueError(f"layer_idx must be non-negative, got {layer_idx}")
            if (self.max_cache_size is not None
                    and layer_idx >= self.max_cache_size):
                raise IndexError(
                    f"Cache index {layer_idx} exceeds max_cache_size="
                    f"{self.max_cache_size}.")
            self._ensure_entry(layer_idx, key_states)
            cur = self._cursors[layer_idx]
            T = key_states.shape[2]
            if cur + T > self._max_len:
                raise RuntimeError(
                    f"PreallocUTCache overflow: entry {layer_idx} at {cur}+{T} "
                    f"> max_len {self._max_len}.")
            self._bufs_k[layer_idx][:, :, cur:cur + T] = key_states
            self._bufs_v[layer_idx][:, :, cur:cur + T] = value_states
            self._cursors[layer_idx] = cur + T
            self._seen_tokens = self._cursors[layer_idx]
            k = self._bufs_k[layer_idx][:, :, :cur + T]
            v = self._bufs_v[layer_idx][:, :, :cur + T]
            # Keep the parent's bookkeeping lists coherent for any HF utility
            # that inspects them (get_seq_length reads key_cache).
            while len(self.key_cache) <= layer_idx:
                self.key_cache.append(None)
                self.value_cache.append(None)
            self.key_cache[layer_idx] = k
            self.value_cache[layer_idx] = v
            return k, v

        def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
            for i, buf in enumerate(self._bufs_k):
                if buf is None:
                    continue
                idx = beam_idx.to(buf.device)
                self._bufs_k[i] = self._bufs_k[i].index_select(0, idx)
                self._bufs_v[i] = self._bufs_v[i].index_select(0, idx)
                cur = self._cursors[i]
                self.key_cache[i] = self._bufs_k[i][:, :, :cur]
                self.value_cache[i] = self._bufs_v[i][:, :, :cur]

    return PreallocUTCache


def make_prealloc_cache(teacher, max_len: int):
    """Locate the teacher's own UT cache class and return a preallocated
    instance sized for `max_len` total positions (prompt + generation)."""
    mod = sys.modules[type(teacher).__module__]
    base = getattr(mod, "UniversalTransformerCache", None)
    if base is None:
        raise RuntimeError(
            f"UniversalTransformerCache not found in {type(teacher).__module__} "
            "— teacher is not an Ouro UT model; prealloc cache unsupported.")
    total_layers = getattr(teacher.config, "num_hidden_layers", None)
    ut_steps = getattr(teacher.config, "total_ut_steps", 4) or 4
    max_cache = total_layers * ut_steps if total_layers else None
    return _build_prealloc_class(base)(max_len=max_len, max_cache_size=max_cache)


def validate_cache_equivalence(teacher, sample_ids: torch.Tensor,
                               max_len: int, n_check: int = 8,
                               kl_tol: float = 5e-2) -> bool:
    """
    Startup gate, mirroring `_validate_teacher_cache`: greedy-decode `n_check`
    tokens with the PREALLOC cache vs the uncached full-recompute reference.
    Requires argmax match and KL(uncached‖prealloc) < kl_tol at every step.
    """
    try:
        ids = sample_ids[:1].detach()
        past = make_prealloc_cache(teacher, max_len=max_len)
        max_kl = 0.0
        for i in range(n_check):
            ref = teacher_logits(teacher, ids)[:, -1, :].float()
            cur = ids if i == 0 else ids[:, -1:]
            start = 0 if i == 0 else ids.shape[1] - 1
            got, past = teacher_logits_cached(teacher, cur, past, start)
            got = got[:, -1, :].float().to(ref.device)
            kl = F.kl_div(
                F.log_softmax(got, dim=-1), F.log_softmax(ref, dim=-1),
                log_target=True, reduction="sum").item()
            max_kl = max(max_kl, kl)
            ref_tok = ref.argmax(dim=-1, keepdim=True)
            if not torch.equal(ref_tok, got.argmax(dim=-1, keepdim=True)):
                raise RuntimeError(f"greedy token mismatch at step {i}")
            if not (kl < kl_tol):
                raise RuntimeError(
                    f"KL={kl:.3e} nats >= tol {kl_tol:g} at step {i}")
            ids = torch.cat([ids, ref_tok.to(ids.device)], dim=1)
        logger.info(
            f"prealloc-cache equivalence PASSED ({n_check} steps, "
            f"max KL={max_kl:.2e} nats).")
        return True
    except Exception as exc:                                    # noqa: BLE001
        logger.warning(
            f"prealloc-cache equivalence FAILED ({exc}); "
            "falling back to the stock dynamic cache.")
        return False
