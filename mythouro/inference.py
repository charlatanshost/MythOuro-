"""
Inference utilities for MythOuro (Part 1).

Components
----------
UncertaintyGatedGenerator   — adaptive loop budget per generated token.
                              Cheap path first (min_loops); if the
                              UncertaintyHead flags the new token as uncertain,
                              re-decode that single step at max_loops. Trades a
                              ~2x worst-case cost for a calibrated quality
                              floor — average cost stays near min_loops on
                              easy prompts.

SpeculativeDecoder          — single-model speculative decoding. The shallow
                              forward (draft_loops) proposes K candidate
                              tokens; the deep forward (verify_loops) accepts
                              them in parallel via the standard
                              Leviathan-Chen-Lozhkov speculative-sampling
                              acceptance test. Net 2–3x speedup at parity
                              with verify-loops sampling, no draft model
                              required.

CrossLoopKVCache            — drop-in replacement for the plain `dict` cache
                              that caps the number of distinct loop entries.
                              Loops above `share_after` write into a single
                              shared slot, recovering most of the memory of a
                              full per-loop cache while preserving the
                              early-loop KV that carries the most distinct
                              signal.

ComponentGradNormLogger     — per-block grad-norm diagnostic. Splits the
                              model into prelude / recurrent / coda / head and
                              reports the L2 norm of each group's parameter
                              gradients. Used to diagnose where instability
                              originates without parsing per-parameter logs.

All four are pure wrappers around a trained MythOuro — no extra parameters,
no state mutated on the model itself. They can be swapped in and out
freely at inference / debugging time.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger


# ---------------------------------------------------------------------------
# UncertaintyGatedGenerator
# ---------------------------------------------------------------------------


class UncertaintyGatedGenerator:
    """
    Generate with adaptive loop depth driven by the UncertaintyHead.

    Each decode step:
        1. Forward with `min_loops`, sample a candidate token.
        2. If the uncertainty at the last position exceeds `threshold`,
           re-do that single decode step with `max_loops` and replace the
           candidate. The KV cache is rolled back to the pre-step state
           before the re-decode so the deep forward sees the same context.
        3. Append the (possibly re-sampled) token and continue.

    Average cost converges to `min_loops` on easy prompts (uncertainty
    rarely triggers); worst-case cost is `min_loops + max_loops` per
    token. The expected-cost curve is flat across the run, unlike a
    fixed `n_loops = max_loops` setting whose cost is `max_loops` for
    every token regardless of difficulty.
    """

    def __init__(
        self,
        model: nn.Module,
        min_loops: int = 2,
        max_loops: int = 16,
        threshold: float = 0.5,
    ):
        self.model = model
        self.min_loops = min_loops
        self.max_loops = max_loops
        self.threshold = threshold

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 64,
        temperature: float = 1.0,
        top_k: int = 50,
    ) -> torch.Tensor:
        prompt_len = input_ids.shape[1]

        # We snapshot the pre-step cache before each new token so we can
        # rewind if the uncertainty gate fires. No tensor copies are needed
        # (P1.6 — the old code cloned EVERY cache tensor EVERY step, O(S)
        # memory traffic per token): the attention layers never mutate stored
        # tensors in place — each step they `cat` into a NEW tensor and
        # REPLACE the entry — so a structure-only snapshot holding references
        # to the pre-step tensors is a correct zero-copy rewind.
        kv_cache: dict = {}

        for step in range(max_new_tokens):
            if step == 0:
                cur_ids = input_ids
                start_pos = 0
            else:
                cur_ids = input_ids[:, -1:]
                start_pos = prompt_len + step - 1

            snapshot = {k: dict(v) for k, v in kv_cache.items()}

            logits, unc = self.model(
                cur_ids, n_loops=self.min_loops, kv_cache=kv_cache, start_pos=start_pos,
            )

            # Decide if we trust the shallow forward at the *last* position.
            need_redo = unc[:, -1].max().item() >= self.threshold

            if need_redo and step > 0:                              # only re-decode single-token steps
                kv_cache = snapshot                                 # rewind
                logits, unc = self.model(
                    cur_ids, n_loops=self.max_loops, kv_cache=kv_cache, start_pos=start_pos,
                )

            next_tok = _sample(logits[:, -1, :], temperature, top_k)
            input_ids = torch.cat([input_ids, next_tok], dim=1)

        return input_ids


def _sample(logits: torch.Tensor, temperature: float, top_k: int) -> torch.Tensor:
    """Standard top-K + temperature sampler."""
    logits = logits / max(temperature, 1e-5)
    if top_k > 0:
        v, _ = logits.topk(top_k)
        logits = logits.masked_fill(logits < v[:, -1:], float("-inf"))
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


# ---------------------------------------------------------------------------
# SpeculativeDecoder (single-model variant)
# ---------------------------------------------------------------------------


class SpeculativeDecoder:
    """
    Single-model speculative decoding (Leviathan et al., 2023; Chen et al.,
    2023). The "draft" and "verify" models are the same MythOuro, run at
    different loop depths.

    Per outer step:
        1. Draft: starting from the current sequence, generate K candidate
           tokens autoregressively at `draft_loops` depth, recording each
           token's draft probability `q(t)`.
        2. Verify: run a single forward pass at `verify_loops` over the K
           candidates appended to the sequence, producing target
           probabilities `p(t)` for each candidate position in parallel.
        3. Acceptance: walk the K candidates left-to-right; accept token t
           with probability `min(1, p(t)/q(t))`. On the first rejection,
           sample one extra token from the residual `max(0, p - q)` and
           stop. Otherwise sample one more from p_{K+1}.

    Expected acceptance rate is high when draft and verify distributions
    are similar (which they are: same model, just shallower). Net gain
    comes from amortising the verify forward over K candidate positions.
    """

    def __init__(
        self,
        model: nn.Module,
        draft_loops: int = 2,
        verify_loops: "int | None" = None,
        K: int = 8,
        temperature: float = 1.0,
    ):
        self.model = model
        self.draft_loops = draft_loops
        # P1.7: the old default (16) silently ran 4x the distill configs'
        # trained depth on every verify pass. Default to the trained depth.
        self.verify_loops = verify_loops or model.cfg.max_loop_iters
        self.K = K
        self.temperature = max(temperature, 1e-5)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 64,
    ) -> torch.Tensor:
        prompt_len = input_ids.shape[1]

        produced = 0
        # We rebuild the KV cache from scratch on every outer step. The
        # acceptance test can reject any suffix of the K candidates, and
        # rolling back partial cache state is more complex than recomputing
        # the (relatively cheap) verify forward without a cache. For
        # short K (≤16) the no-cache verify is competitive in practice.
        while produced < max_new_tokens:
            # ── 1. Draft K tokens autoregressively at draft_loops depth ──
            draft_tokens: list[torch.Tensor] = []
            draft_probs: list[torch.Tensor] = []
            draft_dists: list[torch.Tensor] = []   # full q per step (P1.7) —
            # K×(B,V) floats is trivial memory and saves a full draft forward
            # on every rejection (the residual resample needs the whole q).
            seq = input_ids
            for _ in range(self.K):
                logits, _ = self.model(seq, n_loops=self.draft_loops)
                q = F.softmax(logits[:, -1, :] / self.temperature, dim=-1)  # (B, V)
                tok = torch.multinomial(q, num_samples=1)                   # (B, 1)
                draft_tokens.append(tok)
                draft_probs.append(q.gather(-1, tok).squeeze(-1))           # (B,)
                draft_dists.append(q)
                seq = torch.cat([seq, tok], dim=1)

            # ── 2. Verify all K candidates in a single parallel forward ──
            logits_v, _ = self.model(seq, n_loops=self.verify_loops)
            # Positions of the K candidate tokens are prompt_len + i (i=0..K-1)
            # within `seq`. logits_v[:, j, :] predicts seq[:, j+1].
            # Therefore p(candidate_i) = softmax(logits_v[:, prompt_len + produced + i - 1, :])
            start = seq.shape[1] - self.K - 1
            p_logits = logits_v[:, start : start + self.K, :] / self.temperature   # (B, K, V)
            p_dist = F.softmax(p_logits, dim=-1)                                   # (B, K, V)

            # ── 3. Acceptance walk ──
            accepted = 0
            for i in range(self.K):
                tok_i = draft_tokens[i]                                # (B, 1)
                q_i = draft_probs[i]                                   # (B,)
                p_i = p_dist[:, i, :].gather(-1, tok_i).squeeze(-1)    # (B,)
                # Acceptance probability per batch element. We accept iff
                # u ~ U(0,1) < min(1, p/q); if any batch element rejects,
                # we stop the whole batch at this position (clean shared
                # state across the batch — handling per-row diverging
                # acceptances requires KV-cache surgery).
                ratio = (p_i / q_i.clamp_min(1e-12)).clamp(max=1.0)
                u = torch.rand_like(ratio)
                if (u < ratio).all():
                    accepted += 1
                else:
                    break

            # Append accepted tokens to input_ids
            for i in range(accepted):
                input_ids = torch.cat([input_ids, draft_tokens[i]], dim=1)
            produced += accepted

            if produced >= max_new_tokens:
                break

            # If a token was rejected, sample one resampled token from
            # the residual distribution (p - q)+. If all K accepted,
            # sample one extra from p_{K+1} so we always emit ≥1 token.
            if accepted < self.K:
                # Sample from the residual distribution at the rejected
                # position, using the draft distribution STORED during
                # drafting (P1.7 — identical to the old fresh-forward
                # recompute, since the draft context at that position is the
                # same; just without paying a full extra draft forward).
                q_r = draft_dists[accepted]
                p_r = p_dist[:, accepted, :]
                residual = (p_r - q_r).clamp(min=0)
                residual = residual / residual.sum(dim=-1, keepdim=True).clamp_min(1e-12)
                tok_new = torch.multinomial(residual, num_samples=1)
                input_ids = torch.cat([input_ids, tok_new], dim=1)
                produced += 1
            else:
                # All K accepted; emit one bonus token. The verify forward's
                # LAST position is cand_{K-1}, whose logits ARE the bonus
                # distribution — the old code paid an entire extra verify
                # forward to recompute logits it already had (P1.7).
                p_extra = F.softmax(logits_v[:, -1, :] / self.temperature, dim=-1)
                tok_extra = torch.multinomial(p_extra, num_samples=1)
                input_ids = torch.cat([input_ids, tok_extra], dim=1)
                produced += 1

        return input_ids[:, : prompt_len + max_new_tokens]


# ---------------------------------------------------------------------------
# CrossLoopKVCache
# ---------------------------------------------------------------------------


def compress_kv_cache(cache: dict, share_after: int = 4) -> dict:
    """
    Post-forward KV-cache compressor for the recurrent block.

    The model writes one cache entry per loop iteration
    (`recurrent_loop_0`, `recurrent_loop_1`, …). For very deep loops the
    K and V tensors at adjacent late loops converge — most of the
    distinct attention signal lives in the early iterations. This
    function collapses every `recurrent_loop_t` with `t >= share_after`
    into a single representative slot by averaging the K and V (or, for
    MLA, the `c_kv` and `k_rope` latents) across them.

    Use between decode steps:

        cache: dict = {}
        # Prefill — produces full per-loop cache
        model(prompt_ids, kv_cache=cache, start_pos=0)

        # Compress before decoding
        cache = compress_kv_cache(cache, share_after=4)

        # Subsequent decode steps use the smaller cache
        for step in range(max_new):
            model(next_tok, kv_cache=cache, start_pos=...)

    Why post-forward, not transparent-on-write
    ------------------------------------------
    Both `GQAttention` and `MLAttention` mutate the cache via
    `cat(old, new)`. If a wrapper transparently remapped multiple loop
    keys to one slot, the cat would fire multiple times per forward
    against the same slot and inflate the cache to N_loops × T length,
    not the intended T. Compressing AFTER the forward avoids that
    interaction entirely.

    Returns a new dict — the input cache is not mutated.
    """
    out: dict = {}
    to_merge: list[dict] = []
    for k, v in cache.items():
        if k.startswith("recurrent_loop_"):
            try:
                t = int(k.rsplit("_", 1)[-1])
            except ValueError:
                out[k] = v
                continue
            if t < share_after:
                out[k] = v
            else:
                to_merge.append(v)
        else:
            out[k] = v

    if to_merge:
        merged_key = f"recurrent_loop_{share_after}"
        merged = {}
        for field in to_merge[0].keys():
            stacked = torch.stack([entry[field] for entry in to_merge], dim=0)
            merged[field] = stacked.mean(dim=0)
        out[merged_key] = merged

    return out


class CrossLoopKVCache:
    """
    Convenience wrapper that pairs an underlying cache dict with a
    `share_after` setting and applies `compress_kv_cache` on demand.

    Usage:
        cl = CrossLoopKVCache(share_after=4)
        model(prompt_ids, kv_cache=cl.cache, start_pos=0)
        cl.compress()                              # collapse late loops
        for step in range(max_new):
            model(next_tok, kv_cache=cl.cache, start_pos=...)
    """

    def __init__(self, share_after: int = 4):
        self.share_after = share_after
        self.cache: dict = {}

    def compress(self) -> None:
        """Compress the underlying cache in-place."""
        self.cache = compress_kv_cache(self.cache, self.share_after)

    def reset(self) -> None:
        self.cache = {}

    def memory_bytes(self) -> int:
        """Sum bytes across every tensor in the cache. Diagnostic only."""
        total = 0
        for entry in self.cache.values():
            for t in entry.values():
                total += t.numel() * t.element_size()
        return total


# ---------------------------------------------------------------------------
# ComponentGradNormLogger
# ---------------------------------------------------------------------------


class ComponentGradNormLogger:
    """
    Per-block parameter-gradient L2 norm reporter.

    Splits the model into four named groups:
        - prelude     (TransformerBlocks before the loop)
        - recurrent   (the single looped TransformerBlock + LoRA + injection)
        - coda        (TransformerBlocks after the loop)
        - head        (LM head, output norm, uncertainty head)

    For each group, sums grad-squared across every parameter and reports
    the sqrt as the group's gradient norm. Cheap (one buffer per param,
    O(params) time) — safe to call every log step.

    Used to localise instability: if grad_norm['recurrent'] >> the others,
    the loop is the source; if grad_norm['head'] explodes, the LM head
    is overfitting noise; etc.
    """

    GROUPS = ("prelude", "recurrent", "coda", "head", "uncertainty", "sink", "embed", "norm")

    @staticmethod
    def _group(name: str) -> str:
        for g in ComponentGradNormLogger.GROUPS:
            if name.startswith(g) or f".{g}." in name or name.endswith(g):
                return g
        return "other"

    @classmethod
    def compute(cls, model: nn.Module) -> dict[str, float]:
        sq: dict[str, float] = {g: 0.0 for g in cls.GROUPS}
        sq["other"] = 0.0
        for name, p in model.named_parameters():
            if p.grad is None:
                continue
            g = cls._group(name)
            sq[g] += float(p.grad.detach().pow(2).sum().item())
        return {k: v ** 0.5 for k, v in sq.items()}

    @classmethod
    def log(cls, model: nn.Module, step: int) -> None:
        norms = cls.compute(model)
        parts = " | ".join(f"{k} {v:.3f}" for k, v in norms.items() if v > 0)
        logger.info(f"step {step:>7d} | grad norms by component: {parts}")


# ---------------------------------------------------------------------------
# Convenience aliases
# ---------------------------------------------------------------------------


def speculative_generate(
    model: nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int = 64,
    draft_loops: int = 2,
    verify_loops: int = 16,
    K: int = 8,
    temperature: float = 1.0,
) -> torch.Tensor:
    """One-shot helper that constructs SpeculativeDecoder and calls generate."""
    return SpeculativeDecoder(
        model,
        draft_loops=draft_loops,
        verify_loops=verify_loops,
        K=K,
        temperature=temperature,
    ).generate(input_ids, max_new_tokens=max_new_tokens)


def uncertainty_gated_generate(
    model: nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int = 64,
    min_loops: int = 2,
    max_loops: int = 16,
    threshold: float = 0.5,
    temperature: float = 1.0,
    top_k: int = 50,
) -> torch.Tensor:
    """One-shot helper that constructs UncertaintyGatedGenerator and calls generate."""
    return UncertaintyGatedGenerator(
        model, min_loops=min_loops, max_loops=max_loops, threshold=threshold,
    ).generate(
        input_ids, max_new_tokens=max_new_tokens,
        temperature=temperature, top_k=top_k,
    )


# ===========================================================================
# Part 2 inference utilities
# ===========================================================================


# ---------------------------------------------------------------------------
# Continuous depth-wise batching
# ---------------------------------------------------------------------------


class ContinuousDepthwiseBatcher:
    """
    Per-sequence early-exit batching for the recurrent block.

    The plain RecurrentBlock runs every sequence in a batch for the same
    number of loops — the longest one's halt depth. Easy sequences waste
    compute waiting for hard ones. This batcher masks halted sequences out
    of subsequent loop iterations, letting the model continue looping only
    on sequences that still need it.

    Expected speedup: 1.5–3× on mixed-difficulty batches (e.g. a batch
    containing both "What is 2+2?" and a multi-hop reasoning question).

    Implementation: we re-implement the recurrent loop here so we can
    slice active sequences each iteration. Sequences that fully halt have
    their `h_out` frozen at the halt point and stay out of the loop body
    until the rest catch up or `n_loops` is reached.

    Limitations:
        * KV-cache compatibility is not supported in this batcher.
          The cache mutates across loop iterations and slicing active
          rows out of it is non-trivial; for cached decode use
          UncertaintyGatedGenerator instead.
        * Halt decisions are at the sequence level (all tokens of a row
          halted), not the token level — matches the speed-up story
          while keeping the implementation tractable.
    """

    def __init__(self, model: nn.Module):
        self.model = model

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        n_loops: int = 16,
    ) -> "tuple[torch.Tensor, torch.Tensor]":
        """
        Returns `(logits, uncertainty)` matching the MythOuro.forward
        contract. The shape is identical to a normal forward pass — the
        difference is purely speed.
        """
        model = self.model
        device = input_ids.device

        # Prelude (same for all sequences) — done once.
        x = model.embed(input_ids)
        x, sink_len = model.sink.prepend(x)
        T_ext = x.shape[1]
        freqs = (
            model.freqs_cis_mla if model.cfg.attn_type == "mla" else model.freqs_cis
        )[:T_ext]
        mask = model._causal_mask(T_ext, device, x.dtype) if T_ext > 1 else None

        for i, layer in enumerate(model.prelude):
            x = layer(x, freqs, mask, None, cache_key=f"prelude_{i}")

        e = x
        h = x.clone()
        B, T, D = h.shape

        # Mirror RecurrentBlock.forward's ACT accounting but with active-row
        # slicing so finished sequences drop out of the inner forward.
        rec = model.recurrent
        halted_seq = torch.zeros(B, device=device, dtype=torch.bool)
        cumulative_p = torch.zeros(B, T, device=device)
        loop_state_buf: list[torch.Tensor] = []      # full-B snapshots (P0.4)

        for t in range(n_loops):
            active_idx = (~halted_seq).nonzero(as_tuple=True)[0]
            if active_idx.numel() == 0:
                break

            h_active = h[active_idx]
            e_active = e[active_idx]

            # Per-loop injection signal (multi-scale aware)
            e_inject = rec.ms_inject(e_active, t) if rec.use_ms else e_active

            # One loop body iteration on the active subset.
            # _loop_body returns (h_new, router_logits, expert_counts) since
            # P0.2; this batcher only needs the hidden state.
            h_new_active, _, _ = rec._loop_body(
                h_active, e_inject, freqs, mask, None, t,
            )

            # Splice updated rows back into the full-batch hidden state
            # (halted rows keep their frozen halt-time state).
            h[active_idx] = h_new_active

            # Cross-loop attention residual (P0.4). The buffer must hold
            # FULL-batch snapshots so row i of every entry is always the same
            # sequence — appending active-subset states (the old behaviour)
            # produced ragged batch dims (crash in the attention `cat`) and,
            # worse, silently attended row i to another sequence's history once
            # the active set shrank. Snapshot the spliced full-B state (clone:
            # `h` is mutated in place across loops, and _maybe_snapshot stores
            # a storage-sharing detach), then attend on a per-row slice.
            if rec.use_cross:
                cross = rec.cross_loop_attn
                if t % cross.store_every == 0:
                    cross._maybe_snapshot(h.clone(), t, loop_state_buf)
                sliced = [s[active_idx] for s in loop_state_buf]
                h_new_active = cross._attend(h_new_active, sliced)
                h[active_idx] = h_new_active

            # ACT accounting on active rows — purely a halt CRITERION (P0.3:
            # the emitted state is the per-row final h_K, not an ACT blend).
            p_active = rec.act(h_new_active)                       # (B_act, T)
            threshold = rec.cfg.act_threshold
            cumulative_p[active_idx] = cumulative_p[active_idx] + p_active

            # Mark sequences whose ALL token positions have crossed threshold
            row_halted = (cumulative_p[active_idx] >= threshold).all(dim=1)
            halted_seq[active_idx[row_halted]] = True

        # Coda on the per-row final states: each row's h is frozen at its halt
        # loop (its own h_K) — consistent with the post-P0.3 main path, which
        # emits h_K rather than the never-trained ACT-weighted blend.
        for i, layer in enumerate(model.coda):
            h = layer(h, freqs, mask, None, cache_key=f"coda_{i}")

        if sink_len:
            h = model.sink.strip(h)

        normed = model.norm(h)
        return model.head(normed), model.uncertainty(normed)


# ---------------------------------------------------------------------------
# Retrieval-augmented loop injection
# ---------------------------------------------------------------------------


class RetrievalAugmentedInjector:
    """
    Inject retrieved external context at specific loop iterations.

    Standard RAG prepends retrieved text to the prompt, mixing input
    signal and retrieved signal in the same token stream. This injector
    instead blends a retrieved embedding into the Prelude output `e` at
    specific loop depths, keeping the model's "raw input" and "retrieved
    context" representations separable inside the recurrent stack.

    Adaptive retrieval: if `adaptive=True`, retrieval only fires when
    the model's uncertainty exceeds `uncertainty_threshold` at the
    injection point — cheap queries skip the retriever entirely.

    `retriever` must be callable as `retriever(query_str) -> list[str]`.
    `tokenizer` must expose `.encode(str) -> list[int]`.

    Reasoning: this is a *single-pass* generator. It does not yet
    integrate with the KV cache for fast decode — that's deferred to
    a Part 3 enhancement once we have a real retriever to evaluate
    against.
    """

    def __init__(
        self,
        model: nn.Module,
        retriever,                # Callable[[str], list[str]]
        tokenizer,                # has .encode
        inject_at_loops: "list[int] | None" = None,
        adaptive: bool = True,
        uncertainty_threshold: float = 0.4,
        blend_weight: float = 0.1,
        max_docs: int = 4,
        max_doc_tokens: int = 128,
    ):
        self.model = model
        self.retriever = retriever
        self.tokenizer = tokenizer
        self.inject_at_loops = inject_at_loops or [4, 8]
        self.adaptive = adaptive
        self.unc_threshold = uncertainty_threshold
        self.blend_weight = blend_weight
        self.max_docs = max_docs
        self.max_doc_tokens = max_doc_tokens

        # Small projection mapping the retrieved-embedding pool back into
        # model space. Lives outside the model so it can be trained
        # separately (or kept frozen for purely-inference use).
        dim = model.cfg.dim
        device = next(model.parameters()).device
        self.retrieval_proj = nn.Linear(dim, dim, bias=False).to(device)

    def _encode_retrieved(
        self,
        docs: list,
        device: torch.device,
    ) -> "torch.Tensor | None":
        """Tokenise + mean-pool retrieved documents to a single dim-vector."""
        ids: list[int] = []
        for doc in docs[: self.max_docs]:
            tok = self.tokenizer.encode(doc)[: self.max_doc_tokens]
            ids.extend(tok)
            # Cheap document separator — single space token works for most
            # BPE tokenisers, fall back to no separator if encode returns
            # empty for a space.
            sep = self.tokenizer.encode(" ")
            if sep:
                ids.append(sep[0])
        if not ids:
            return None
        ids_tensor = torch.tensor([ids], dtype=torch.long, device=device)
        with torch.no_grad():
            emb = self.model.embed(ids_tensor)         # (1, len, dim)
        return emb.mean(dim=1)                          # (1, dim)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        query: str,
        n_loops: int = 16,
    ) -> "tuple[torch.Tensor, torch.Tensor]":
        """
        Full forward pass with retrieval injection at the configured loops.
        Returns `(logits, uncertainty)` like MythOuro.forward.
        """
        model = self.model
        device = input_ids.device

        x = model.embed(input_ids)
        x, sink_len = model.sink.prepend(x)
        T_ext = x.shape[1]
        freqs = (
            model.freqs_cis_mla if model.cfg.attn_type == "mla" else model.freqs_cis
        )[:T_ext]
        mask = model._causal_mask(T_ext, device, x.dtype) if T_ext > 1 else None

        for i, layer in enumerate(model.prelude):
            x = layer(x, freqs, mask, None, cache_key=f"prelude_{i}")

        e = x.clone()
        h = x.clone()
        rec = model.recurrent
        retrieved_emb: "torch.Tensor | None" = None
        cumulative_p = torch.zeros(h.shape[:2], device=device)
        h_out = torch.zeros_like(h)
        loop_state_buf: list[torch.Tensor] = []

        for t in range(n_loops):
            # Optional retrieval at this loop
            if t in self.inject_at_loops and retrieved_emb is None:
                fire = True
                if self.adaptive:
                    # Halt-probability is per-token in (0, 1); high value
                    # ~ "still uncertain". We average across the sequence
                    # for a single fire decision.
                    p = rec.act(h)
                    fire = bool(p.mean().item() < self.unc_threshold)
                if fire:
                    logger.debug(
                        f"RetrievalAugmentedInjector: retrieving at loop {t}"
                    )
                    docs = self.retriever(query)
                    retrieved_emb = self._encode_retrieved(docs, device)

            # Blend retrieved embedding into the injection signal
            if retrieved_emb is not None:
                r = self.retrieval_proj(retrieved_emb)                # (1, dim)
                e_step = e + r.unsqueeze(1) * self.blend_weight
            else:
                e_step = e

            e_inject = rec.ms_inject(e_step, t) if rec.use_ms else e_step
            h, _, _ = rec._loop_body(h, e_inject, freqs, mask, None, t)  # P0.2: 3-tuple
            if rec.use_cross:
                h = rec.cross_loop_attn(h, t, loop_state_buf)

            # ACT accounting (sequence-level halt check)
            p = rec.act(h)
            remainder = (1.0 - cumulative_p).clamp(min=0)
            threshold = rec.cfg.act_threshold
            weight = torch.where(
                cumulative_p + p >= threshold, remainder, p,
            )
            h_out = h_out + weight.unsqueeze(-1) * h
            cumulative_p = cumulative_p + p
            if (cumulative_p >= threshold).all():
                break

        for i, layer in enumerate(model.coda):
            h_out = layer(h_out, freqs, mask, None, cache_key=f"coda_{i}")
        if sink_len:
            h_out = model.sink.strip(h_out)
        normed = model.norm(h_out)
        return model.head(normed), model.uncertainty(normed)


# ---------------------------------------------------------------------------
# Chain-of-Thought distillation into latent loops
# ---------------------------------------------------------------------------


class CoTDistillationTrainer:
    """
    Distil explicit CoT reasoning traces into MythOuro's latent loops.

    Idea: for each `(question, [CoT_step_1, …, CoT_step_K], answer)`
    triple, run MythOuro at K loops and at each loop t pull the
    hidden state toward the embedding of CoT step t (cosine similarity
    in dim-space). Combined with the standard answer cross-entropy,
    this teaches the loops to perform structured reasoning steps
    internally — at inference the model reasons in N latent steps
    without emitting any CoT tokens.

    Requirements:
        - CoT dataset: (question_ids, cot_step_embeddings, answer_ids)
          where cot_step_embeddings is a list of (B, dim) tensors, one
          per reasoning step. Use any teacher encoder to produce them.

    This trainer only computes the loss — the user owns the dataset,
    optimiser, and training loop.
    """

    def __init__(
        self,
        model: nn.Module,
        dim_match_coeff: float = 0.1,
        answer_coeff: float = 1.0,
    ):
        self.model = model
        self.dim_match_coeff = dim_match_coeff
        self.answer_coeff = answer_coeff

        # Small projection mapping recurrent-hidden → CoT-embedding space.
        # Allows the teacher encoder's dim to differ from cfg.dim if needed.
        dim = model.cfg.dim
        device = next(model.parameters()).device
        self.cot_proj = nn.Linear(dim, dim, bias=False).to(device)

    def loss(
        self,
        question_ids: torch.Tensor,       # (B, T_q)
        cot_embeddings: list,             # list of (B, dim), one per CoT step
        answer_ids: torch.Tensor,         # (B, T_a)
    ) -> "tuple[torch.Tensor, dict]":
        """
        Compute the combined answer + CoT-alignment loss.

        Captures the recurrent hidden state at each loop index via a
        forward hook on the RecurrentBlock — no model surgery required.
        Returns `(total_loss, metrics)`.
        """
        from mythouro.main import RecurrentBlock

        n_steps = len(cot_embeddings)
        n_loops = min(n_steps, self.model.cfg.max_loop_iters)

        # Hook to capture per-loop hidden state. The RecurrentBlock only
        # exposes its final h_out from a single forward; to get per-loop
        # states we patch its `_loop_body` to record into a buffer.
        loop_states: dict[int, torch.Tensor] = {}
        rec_block: RecurrentBlock | None = None
        for mod in self.model.modules():
            if isinstance(mod, RecurrentBlock):
                rec_block = mod
                break
        if rec_block is None:
            return torch.tensor(0.0, device=question_ids.device), {}

        original = rec_block._loop_body

        def _patched(h, e_inject, freqs_cis, mask, kv_cache, t):
            # _loop_body returns (h_new, router_logits, expert_counts) since
            # P0.2; record the hidden state, pass the full tuple through.
            out = original(h, e_inject, freqs_cis, mask, kv_cache, t)
            loop_states[t] = out[0]
            return out

        rec_block._loop_body = _patched
        full_ids = torch.cat([question_ids, answer_ids], dim=1)
        logits, _ = self.model(full_ids, n_loops=n_loops)
        rec_block._loop_body = original

        # Answer cross-entropy over the answer positions.
        T_q = question_ids.shape[1]
        T_a = answer_ids.shape[1]
        ans_logits = logits[:, T_q - 1 : T_q + T_a - 1, :]
        V = ans_logits.shape[-1]
        answer_loss = F.cross_entropy(
            ans_logits.contiguous().view(-1, V),
            answer_ids.view(-1),
        )

        # CoT alignment: pull mean-pooled loop-t hidden state toward CoT step t
        dim_match_loss = torch.tensor(0.0, device=question_ids.device)
        n_matched = 0
        for t in range(n_loops):
            if t not in loop_states:
                continue
            h_t = loop_states[t]
            h_mean = self.cot_proj(h_t.mean(dim=1))                 # (B, dim)
            cot_emb = cot_embeddings[t]                              # (B, dim)
            cos = F.cosine_similarity(h_mean, cot_emb, dim=-1)
            dim_match_loss = dim_match_loss + (1.0 - cos).mean()
            n_matched += 1
        if n_matched:
            dim_match_loss = dim_match_loss / n_matched

        total = (
            self.answer_coeff * answer_loss
            + self.dim_match_coeff * dim_match_loss
        )
        return total, {
            "answer_loss": answer_loss.item(),
            "dim_match_loss": float(dim_match_loss.item()),
        }


# ---------------------------------------------------------------------------
# Activation offloader
# ---------------------------------------------------------------------------


class ActivationOffloader:
    """
    Offload Prelude / Coda forward activations to CPU between forward and
    backward passes.

    The RecurrentBlock already uses `torch.utils.checkpoint` (Part 1) so
    its activations are O(1) in `n_loops`. The Prelude and Coda layers
    still hold their activations — at seq_len=2048 those can be 2–4 GB
    for a 3B model. Moving them to CPU during forward and back to GPU
    just before backward trades PCIe bandwidth for VRAM.

    On V100 SXM2 (PCIe 3.0 x16 ≈ 16 GB/s) the transfer for a 3B model is
    ~125–250ms — acceptable when overlapped with computation, and
    decisive when the alternative is OOM.

    Usage:
        off = ActivationOffloader(model)
        off.enable()
        # train normally
        off.disable()
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self._hooks: list = []
        self.enabled = False

    def enable(self) -> None:
        if self.enabled:
            return

        def _to_cpu(_mod, _inp, out):
            if isinstance(out, torch.Tensor) and out.is_cuda:
                return out.cpu()
            return out

        def _to_gpu(mod, inp):
            device = next(mod.parameters()).device
            return tuple(
                x.to(device, non_blocking=True)
                if isinstance(x, torch.Tensor) and not x.is_cuda
                else x
                for x in inp
            )

        for name, mod in self.model.named_modules():
            # Offload Prelude and Coda activations only — the recurrent
            # block has its own checkpoint-based memory story.
            if "prelude" in name or "coda" in name:
                self._hooks.append(mod.register_forward_hook(_to_cpu))
                self._hooks.append(mod.register_forward_pre_hook(_to_gpu))

        self.enabled = True
        logger.info(
            f"ActivationOffloader: enabled ({len(self._hooks)} hooks)"
        )

    def disable(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks = []
        self.enabled = False
        logger.info("ActivationOffloader: disabled")


# ---------------------------------------------------------------------------
# INT8 quantization
# ---------------------------------------------------------------------------


_INT8_SKIP_PATTERNS = (
    "embed",          # vocab table
    "head",           # LM head — precision matters for sampling
    "log_A", "log_dt", "B_dir",   # LTI stability-critical params
    "router",         # routing decisions — precision matters
    "scheduler",      # per-loop schedule scalars
    "uncertainty",    # tiny head, no win
    "sink",           # tiny, no win
)


def apply_int8_quantization(
    model: nn.Module,
    extra_skip: "list[str] | None" = None,
) -> nn.Module:
    """
    Apply dynamic INT8 quantization to Linear layers post-training.

    V100s lack INT8 tensor cores (those are Turing/Ampere+), but INT8
    halves memory bandwidth — useful when bandwidth is the bottleneck
    (common at inference on large models). Expect ~1.4× inference
    speedup from memory bandwidth alone.

    Layers skipped (kept in fp16/bf16):
        - Embedding (lookup, not bandwidth-bound)
        - LM head (precision matters for sampling)
        - LTI stability params (log_A, log_dt, B_dir)
        - Router (precision matters for routing)
        - Scheduler / uncertainty / sink (tiny, no win)

    Returns the quantized model. Same forward API — no other changes.
    """
    skip = set(_INT8_SKIP_PATTERNS) | set(extra_skip or [])

    # torch.quantization.quantize_dynamic can't accept name patterns directly;
    # it quantizes ALL Linear layers. To respect the skip list we walk the
    # model and quantize selectively in-place.
    def _should_skip(name: str) -> bool:
        return any(p in name for p in skip)

    quant_count = 0
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and not _should_skip(name):
            # Locate parent and replace the child with a quantized version.
            parent_name, _, child = name.rpartition(".")
            parent = model
            if parent_name:
                for part in parent_name.split("."):
                    parent = getattr(parent, part)
            qmod = torch.quantization.quantize_dynamic(
                nn.Sequential(mod), {nn.Linear}, dtype=torch.qint8
            )[0]
            setattr(parent, child, qmod)
            quant_count += 1

    logger.info(
        f"INT8 dynamic quantization applied to {quant_count} Linear layers"
    )
    logger.info(
        "Note: V100 has no INT8 tensor cores — speedup comes from "
        "halved memory bandwidth (~1.4×), not compute throughput."
    )
    return model


def quantization_aware_training_hooks(model: nn.Module) -> list:
    """
    Attach activation-stat observers to Linear layers for QAT.

    Call in the last ~10% of training to collect distribution statistics
    that improve INT8 calibration. Run training normally; afterwards
    apply `apply_int8_quantization` to materialise the quantized model.

    Returns the list of hook handles — caller is responsible for
    removing them after training completes.
    """
    handles = []
    n_obs = 0
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        if any(p in name for p in ("router", "log_A", "head", "embed")):
            continue

        obs = torch.quantization.MinMaxObserver(
            quant_min=-128,
            quant_max=127,
            dtype=torch.qint8,
            qscheme=torch.per_tensor_symmetric,
        )

        def _hook(_mod, _inp, out, _obs=obs):
            if isinstance(out, torch.Tensor):
                _obs(out.detach().float())

        handles.append(mod.register_forward_hook(_hook))
        n_obs += 1

    logger.info(f"QAT: registered {n_obs} activation observers")
    return handles


# ---------------------------------------------------------------------------
# ConfidenceAwareGenerator
# ---------------------------------------------------------------------------


class ConfidenceAwareGenerator:
    """
    Generate tokens with adaptive early stopping driven by confidence,
    cycling detection, and EOS awareness.

    Stopping conditions (checked after ``min_new_tokens`` are emitted):

        1. **EOS token** — unconditional stop as soon as the sampled token
           equals ``eos_token_id``, regardless of any other setting.

        2. **Sustained low uncertainty + natural break** — if the
           UncertaintyHead's output stays below ``confidence_threshold``
           for ``confidence_window`` consecutive tokens *and* the most
           recently sampled token is a "natural break" (configurable via
           ``break_token_ids``), generation stops. The intuition is that
           the model is confident *and* has reached a sentence / paragraph
           boundary, so continuing is unlikely to add useful content.

        3. **Cycling detection** — if the last ``cycle_window`` tokens
           contain a contiguous repeated n-gram of length
           ``cycle_min_len``, the generator assumes the model is stuck in
           a loop and bails out.

        4. **Max cap** — hard upper bound (``max_new_tokens``).

    Each of the above conditions tags a ``stop_reason`` string in the
    returned result so the caller can distinguish intentional EOS from
    budget exhaustion.

    The generator uses the same single-token KV-cached decode loop as
    ``UncertaintyGatedGenerator``, running at a fixed ``n_loops`` depth
    (no adaptive depth — that concern is orthogonal and can be layered
    on top).
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        n_loops: int = 8,
        eos_token_id: "int | None" = None,
        min_new_tokens: int = 1,
        confidence_threshold: float = 0.3,
        confidence_window: int = 4,
        break_token_ids: "list[int] | None" = None,
        cycle_window: int = 32,
        cycle_min_len: int = 4,
    ):
        """
        Default semantics (chosen to fail-closed rather than fail-open):

        * ``eos_token_id=None`` — EOS stopping is disabled until the caller
          explicitly passes their tokenizer's EOS id. The previous default
          of ``2`` was misleading: id 2 is just the third token in the
          vocabulary, not any real tokenizer's EOS.
        * ``break_token_ids=None`` — confidence-based stopping is disabled
          until the caller explicitly passes a list of "natural break"
          token ids (sentence-ending punctuation, newline, paragraph
          break, etc). The previous default of "empty list = any token
          counts as a break" made confidence stops trivially-firing.

        Both nullable defaults make the generator a strict no-stop
        identity by default; callers opt in to each mechanism by
        supplying ids.
        """
        self.model = model
        self.n_loops = n_loops
        self.eos_token_id = eos_token_id
        self.min_new_tokens = max(min_new_tokens, 1)
        self.confidence_threshold = confidence_threshold
        self.confidence_window = confidence_window
        # `None` → confidence stop disabled; explicit set → match against
        # the sampled token id. Coerce to set for O(1) membership check.
        self.break_token_ids: "set[int] | None" = (
            None if break_token_ids is None else set(break_token_ids)
        )
        self.cycle_window = cycle_window
        self.cycle_min_len = cycle_min_len

    # ------------------------------------------------------------------
    # Cycling detector
    # ------------------------------------------------------------------
    @staticmethod
    def _has_cycle(token_ids: list[int], window: int, min_len: int) -> bool:
        """Return True if the tail of *token_ids* contains a repeated n-gram."""
        tail = token_ids[-window:]
        n = len(tail)
        for length in range(min_len, n // 2 + 1):
            pattern = tail[n - length :]
            start = n - 2 * length
            if start < 0:
                continue
            if tail[start : start + length] == pattern:
                return True
        return False

    # ------------------------------------------------------------------
    # Generate
    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 64,
        temperature: float = 1.0,
        top_k: int = 50,
    ) -> "dict[str, object]":
        """
        Generate up to ``max_new_tokens`` new tokens, returning a dict:

            {
                "sequences":         Tensor (1, T_prompt + N),
                "stop_reason":       str — "eos" / "confidence" /
                                     "cycle" / "max_new_tokens",
                "uncertainty_trace": list[float] — per-generated-token
                                     uncertainty score from
                                     ``UncertaintyHead``. Useful for
                                     debugging threshold choices.
            }

        Single-sequence only (B=1). The stopping heuristics (especially
        the per-token natural-break check) don't generalise to batched
        rows that may want to stop at different positions; a finished-
        mask design would change the API surface considerably and is
        deferred until there's an actual batched-inference need.
        """
        assert input_ids.shape[0] == 1, (
            "ConfidenceAwareGenerator requires a single-sequence input "
            f"(got batch size {input_ids.shape[0]}). Use UncertaintyGatedGenerator "
            "or ContinuousDepthwiseBatcher for batched decode."
        )

        prompt_len = input_ids.shape[1]
        kv_cache: dict = {}

        unc_history: list[float] = []
        generated_ids: list[int] = []
        stop_reason = "max_new_tokens"

        for step in range(max_new_tokens):
            if step == 0:
                cur_ids = input_ids
                start_pos = 0
            else:
                cur_ids = input_ids[:, -1:]
                start_pos = prompt_len + step - 1

            logits, unc = self.model(
                cur_ids,
                n_loops=self.n_loops,
                kv_cache=kv_cache,
                start_pos=start_pos,
            )

            next_tok = _sample(logits[:, -1, :], temperature, top_k)
            tok_id = int(next_tok[0, 0].item())
            input_ids = torch.cat([input_ids, next_tok], dim=1)
            generated_ids.append(tok_id)

            # Uncertainty at the last position (B=1, so no batch ambiguity).
            unc_val = float(unc[0, -1].item())
            unc_history.append(unc_val)

            produced = step + 1  # 1-indexed count of tokens emitted so far

            # ── EOS: bypasses min_new_tokens (active only if id given) ──
            if self.eos_token_id is not None and tok_id == self.eos_token_id:
                stop_reason = "eos"
                break

            # ── Below this line: respect the min_new_tokens floor ──
            if produced < self.min_new_tokens:
                continue

            # ── Sustained low uncertainty + natural break ──
            # Skipped entirely when break_token_ids is None (opt-in only)
            # so a model that happens to be low-uncertainty early can't
            # stop after a single token.
            if (
                self.break_token_ids is not None
                and len(unc_history) >= self.confidence_window
            ):
                recent = unc_history[-self.confidence_window :]
                if (
                    all(u < self.confidence_threshold for u in recent)
                    and tok_id in self.break_token_ids
                ):
                    stop_reason = "confidence"
                    break

            # ── Cycling detection (literal n-gram repetition in token ids) ──
            if len(generated_ids) >= self.cycle_window:
                if self._has_cycle(
                    generated_ids, self.cycle_window, self.cycle_min_len,
                ):
                    stop_reason = "cycle"
                    break

        return {
            "sequences": input_ids,
            "stop_reason": stop_reason,
            "uncertainty_trace": unc_history,
        }


def confidence_aware_generate(
    model: nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int = 64,
    *,
    n_loops: int = 8,
    eos_token_id: "int | None" = None,
    min_new_tokens: int = 1,
    confidence_threshold: float = 0.3,
    confidence_window: int = 4,
    break_token_ids: "list[int] | None" = None,
    cycle_window: int = 32,
    cycle_min_len: int = 4,
    temperature: float = 1.0,
    top_k: int = 50,
) -> "dict[str, object]":
    """One-shot helper that constructs ConfidenceAwareGenerator and calls generate."""
    return ConfidenceAwareGenerator(
        model,
        n_loops=n_loops,
        eos_token_id=eos_token_id,
        min_new_tokens=min_new_tokens,
        confidence_threshold=confidence_threshold,
        confidence_window=confidence_window,
        break_token_ids=break_token_ids,
        cycle_window=cycle_window,
        cycle_min_len=cycle_min_len,
    ).generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
    )


# ---------------------------------------------------------------------------
# BestOfTrajectoryGenerator
# ---------------------------------------------------------------------------


class BestOfTrajectoryGenerator:
    """
    Emit the lowest-uncertainty depth across the recurrent trajectory.

    Standard decoding uses the recurrent block's ACT-weighted blend over loops.
    This generator instead asks the UncertaintyHead to score *every* loop depth
    (via ``MythOuro.forward_trajectory``) and, for each next-token position,
    emits the logits from whichever loop the head is most confident about —
    keeping the best step it saw rather than running extra loops and trying to
    undo a bad one.

    Rationale: more loops can legitimately *raise* entropy on genuinely hard
    tokens before they resolve, so "loop more while uncertain" can overshoot.
    Selecting the best-by-uncertainty step sidesteps that without needing a
    ground-truth revert signal at inference. This is the inference-side
    counterpart to the depth-extrapolation machinery already in the recurrent
    block (loop-index embedding, convergence early-exit, per-loop LoRA).

    Single-sequence only (B=1): the per-token argmin-over-loops selection is
    position-specific and doesn't generalise to batched rows that pick different
    depths. Full recompute per token (no KV cache) because each step re-scores
    the whole trajectory — an experiment / inspector path, not fast decode.

    The returned dict adds a ``chosen_loops`` trace (the loop index emitted for
    each generated token) so you can see whether best-of-trajectory actually
    diverges from always taking the deepest loop.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        n_loops: int = 8,
        eos_token_id: "int | None" = None,
        min_loops: int = 2,
        force_full_depth: bool = False,
        cycle_window: int = 32,
        cycle_min_len: int = 4,
    ):
        """
        Args:
            n_loops          -- recurrent depth scored per step (may exceed the
                                trained value for depth extrapolation).
            eos_token_id     -- stop when this id is emitted (None disables, same
                                fail-closed convention as ConfidenceAwareGenerator).
            min_loops        -- floor on the selectable depth: loops shallower
                                than this are excluded from the argmin unless the
                                trajectory is shorter (early convergence).
                                Default 2 (exclude loop 0): the per-loop
                                calibration audit (P0.5, tools/per_loop_calibration)
                                measured the UncertaintyHead as badly miscalibrated
                                at loop 0 on v2 AND v4 (ECE ~0.17–0.22, error
                                UNDERSTATED by ~0.2 — the loop-curriculum starts at
                                2, so loop 0 was never an emission loop during
                                training). Loop-0 argmin "wins" are inflated.
            force_full_depth -- suppress ACT's early-exit so every step scores
                                the full n_loops (counterfactual measurement —
                                see MythOuro.forward_trajectory).
        """
        self.model = model
        self.n_loops = n_loops
        self.eos_token_id = eos_token_id
        self.min_loops = max(min_loops, 1)
        self.force_full_depth = force_full_depth
        self.cycle_window = cycle_window
        self.cycle_min_len = cycle_min_len

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 64,
        temperature: float = 1.0,
        top_k: int = 50,
    ) -> "dict[str, object]":
        """
        Returns a dict:

            {
                "sequences":         Tensor (1, T_prompt + N),
                "stop_reason":       "eos" / "cycle" / "max_new_tokens",
                "uncertainty_trace": list[float] — uncertainty of the *emitted*
                                     loop for each generated token,
                "chosen_loops":     list[int] — which loop index was emitted.
            }
        """
        assert input_ids.shape[0] == 1, (
            "BestOfTrajectoryGenerator requires a single-sequence input "
            f"(got batch size {input_ids.shape[0]})."
        )

        generated_ids: list[int] = []
        unc_history: list[float] = []
        chosen_loops: list[int] = []
        per_loop_unc: list[list[float]] = []
        stop_reason = "max_new_tokens"

        for _ in range(max_new_tokens):
            logits_traj, unc_traj = self.model.forward_trajectory(
                input_ids, n_loops=self.n_loops,
                force_full_depth=self.force_full_depth,
            )
            # Last position only (B=1): (K, V) logits, (K,) uncertainty.
            last_logits = logits_traj[0, -1]
            last_unc = unc_traj[0, -1]
            K = int(last_unc.shape[0])
            # Full per-loop uncertainty vector for this token — lets the caller
            # distinguish "head genuinely discriminates depth" (an interior
            # minimum) from "head just penalises the last loop" (monotonic).
            per_loop_unc.append([float(u) for u in last_unc.tolist()])

            # Exclude depths below the min_loops floor from selection, unless
            # the trajectory is shorter than the floor (convergence cut it).
            floor = min(self.min_loops - 1, K - 1)
            cand_unc = last_unc.clone()
            if floor > 0:
                cand_unc[:floor] = float("inf")
            best_k = int(torch.argmin(cand_unc).item())

            chosen_loops.append(best_k)
            unc_history.append(float(last_unc[best_k].item()))

            next_tok = _sample(last_logits[best_k : best_k + 1], temperature, top_k)
            tok_id = int(next_tok[0, 0].item())
            input_ids = torch.cat([input_ids, next_tok], dim=1)
            generated_ids.append(tok_id)

            if self.eos_token_id is not None and tok_id == self.eos_token_id:
                stop_reason = "eos"
                break

            if len(generated_ids) >= self.cycle_window and (
                ConfidenceAwareGenerator._has_cycle(
                    generated_ids, self.cycle_window, self.cycle_min_len,
                )
            ):
                stop_reason = "cycle"
                break

        return {
            "sequences": input_ids,
            "stop_reason": stop_reason,
            "uncertainty_trace": unc_history,
            "chosen_loops": chosen_loops,
            "per_loop_uncertainty": per_loop_unc,
        }


def best_of_trajectory_generate(
    model: nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int = 64,
    *,
    n_loops: int = 8,
    eos_token_id: "int | None" = None,
    min_loops: int = 2,
    force_full_depth: bool = False,
    temperature: float = 1.0,
    top_k: int = 50,
) -> "dict[str, object]":
    """One-shot helper that constructs BestOfTrajectoryGenerator and calls generate.
    min_loops defaults to 2 — see BestOfTrajectoryGenerator (loop 0 measured
    miscalibrated, P0.5)."""
    return BestOfTrajectoryGenerator(
        model,
        n_loops=n_loops,
        eos_token_id=eos_token_id,
        min_loops=min_loops,
        force_full_depth=force_full_depth,
    ).generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
    )
