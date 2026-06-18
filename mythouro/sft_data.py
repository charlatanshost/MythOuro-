"""
SFT dataset for MythOuro.

The single architectural difference between SFT and pretraining/distillation
is the loss-masking contract: only assistant-response tokens contribute
gradient. Prompt tokens (system / user turns + the assistant role header)
must be masked out, otherwise the model wastes capacity learning to predict
its own prompts back, which both degrades quality and biases generations.

This module provides:

* `MixedSFTDataset` — streaming IterableDataset that interleaves three HF
  instruction corpora at fixed proportions, applies the ChatML chat
  template from `MythOuroTokenizer.apply_chat_template`, and emits
  `(input_ids, target_ids, loss_mask)` triples ready for the trainer.

* `_to_messages` adapters — per-source normalisers that convert each
  dataset's native schema into a canonical list of
  `{"role": ..., "content": ...}` dicts. This is the only piece of code
  that needs to change when adding a new SFT source.

Loss-mask semantics
-------------------
Given a packed sequence `tokens = [p_0, ..., p_{P-1}, r_0, ..., r_{R-1}]`
where `P` is prompt length and `R` is response length:

    input_ids  = tokens[:-1]      # (T,) where T = P + R - 1
    target_ids = tokens[1:]       # (T,)
    loss_mask  = [0] * (P - 1)    # predicting prompt tokens — drop
               + [1] *  R         # predicting response tokens — keep
               + [0] * (pad)      # padding — drop

The trainer's effective loss is

    L = Σ_i loss_mask[i] · CE(logits_i, target_ids[i])  /  Σ_i loss_mask[i]

so only response-token CE contributes. The terminating `<|im_end|>` and
trailing newline are included on the response side so the model learns
to halt instead of rambling.
"""

from __future__ import annotations

import json
import random
from typing import Iterator, Optional

import torch
from loguru import logger
from torch.utils.data import IterableDataset, get_worker_info


# ---------------------------------------------------------------------------
# Source mix
# ---------------------------------------------------------------------------

# Fixed proportions for SFT runs. The mix is tuned for the recurrent-MoE
# architecture's three target capabilities: instruction-following structure
# (general), procedural reasoning (math), and code generation (code).
#
# OpenHermes-2.5 history: at seq_len=512 it had ~95% rejection because
# its multi-turn conversations almost always exceeded the prompt budget
# before the response could land in the loss-bearing region. At
# seq_len=1024 (recommended for general SFT) the rejection rate drops
# to a manageable level (~30-40%), making it productive again. If you're
# running at seq_len=512, set the general ratio to 0 and rebalance —
# the iterator's diagnostic logger will warn if rejection rates spike
# anyway.
_SFT_MIX_RATIOS = {
    "general": 0.30,
    "math":    0.40,
    "code":    0.30,
}

# (key, repo, config, split, cap) — cap=None loads the full split; an int cap
# loads `train[:cap]` (non-streaming slice: cached on disk, Arrow-memory-
# mapped, bounded RAM — consistent with the anti-streaming stance below).
_SFT_DATASET_SPECS = [
    ("general", "teknium/OpenHermes-2.5",                None, "train", None),
    ("math",    "meta-math/MetaMathQA",                  None, "train", None),
    ("code",    "ise-uiuc/Magicoder-Evol-Instruct-110K", None, "train", None),
]

# ---------------------------------------------------------------------------
# CLEAN mix (default since 2026-06-11) — zero OpenAI-output provenance.
# Registry + license notes: docs/clean_sft_datasets.md. The legacy mix above
# is retained for reproducing v2/v4-era runs via mix="legacy"; new checkpoints
# must not inherit the OpenAI-ToS constraint (user decision — see roadmap
# "Licensing & data provenance").
#
# OASST note: raw OpenAssistant/oasst2 is tree-structured + multilingual;
# Tulu-3 already contains a converted, flattened OASST slice
# (source=ai2-adapt-dev/oasst1_converted), so it is ingested via Tulu.
#
# OpenMathInstruct-2 is built by augmenting GSM8K-style problems
# (problem_source=augmented_gsm8k) — the contamination filter vs the GSM8K
# test split is MANDATORY for this mix (on by default in MixedSFTDataset).
# ---------------------------------------------------------------------------

_CLEAN_MIX_RATIOS = {
    "clean_general":  0.30,
    "clean_math":     0.18,
    "clean_numina":   0.12,
    "clean_code":     0.20,
    "clean_miriad":   0.07,
    "clean_pubmedqa": 0.05,
    "clean_chem":     0.08,
}

# CHAT-HEAVY clean variant (mix="clean_chat", 2026-06-14). Same clean sources,
# but weighted toward DIVERSE conversation (Tulu) and away from low-diversity
# STRUCTURED data (math, yes/no medical QA, SMILES). Tests the hypothesis that
# the standard clean mix's ~50% structured content drives generation
# mode-collapse (worse with more SFT — observed at 6k steps), whereas diverse
# chat keeps generation varied (as v4's OpenHermes did, but with zero OpenAI
# provenance). If this avoids the collapse, it's both the diagnosis AND the fix.
_CLEAN_CHAT_MIX_RATIOS = {
    "clean_general":  0.60,   # Tulu-3: diverse chat / FLAN / converted OASST
    "clean_code":     0.20,   # moderately diverse
    "clean_math":     0.05,
    "clean_numina":   0.05,
    "clean_miriad":   0.04,
    "clean_pubmedqa": 0.02,
    "clean_chem":     0.04,
}

_CLEAN_DATASET_SPECS = [
    ("clean_general",  "allenai/tulu-3-sft-mixture", None,             "train", 300_000),
    ("clean_math",     "nvidia/OpenMathInstruct-2",  None,             "train", 250_000),
    ("clean_numina",   "AI-MO/NuminaMath-CoT",       None,             "train", 150_000),
    ("clean_code",     "nvidia/OpenCodeInstruct",    None,             "train", 200_000),
    ("clean_miriad",   "miriad/miriad-4.4M",         None,             "train", 100_000),
    ("clean_pubmedqa", "qiaojin/PubMedQA",           "pqa_artificial", "train", 100_000),
    ("clean_chem",     "AI4Chem/ChemData700K",       None,             "train", 100_000),
]


# ---------------------------------------------------------------------------
# Schema adapters — one per source
# ---------------------------------------------------------------------------


def _to_messages_openhermes(sample: dict) -> Optional[list[dict]]:
    """
    OpenHermes-2.5 stores conversations as a list under `"conversations"`
    with `{"from": "system"|"human"|"gpt", "value": ...}` entries.
    """
    convo = sample.get("conversations") or []
    if not convo:
        return None
    role_map = {"system": "system", "human": "user", "gpt": "assistant"}
    msgs = []
    for turn in convo:
        role = role_map.get(turn.get("from"))
        content = turn.get("value")
        if role is None or not content:
            continue
        msgs.append({"role": role, "content": content})
    # Need at least a user turn and an assistant turn to be useful.
    has_user = any(m["role"] == "user" for m in msgs)
    has_assistant = any(m["role"] == "assistant" for m in msgs)
    if not (has_user and has_assistant):
        return None
    return msgs


def _to_messages_magicoder(sample: dict) -> Optional[list[dict]]:
    """
    Magicoder-Evol-Instruct uses a flat `(instruction, response)` schema.
    """
    instruction = sample.get("instruction")
    response = sample.get("response")
    if not instruction or not response:
        return None
    return [
        {"role": "user",      "content": instruction},
        {"role": "assistant", "content": response},
    ]


def _to_messages_metamath(sample: dict) -> Optional[list[dict]]:
    """
    MetaMathQA uses `(query, response)`. Responses contain step-by-step
    math reasoning, which is exactly the CoT pattern we want to transfer.
    """
    query = sample.get("query")
    response = sample.get("response")
    if not query or not response:
        return None
    return [
        {"role": "user",      "content": query},
        {"role": "assistant", "content": response},
    ]


def _to_messages_passthrough(sample: dict) -> Optional[list[dict]]:
    """
    For sources shipping a ready `messages` list of {"role", "content"}
    dicts (Tulu-3 SFT mixture, NuminaMath-CoT). Validates roles/content and
    requires at least one user + one assistant turn.
    """
    msgs = []
    for m in sample.get("messages") or []:
        role = m.get("role")
        content = m.get("content")
        if role not in ("system", "user", "assistant") or not content:
            continue
        msgs.append({"role": role, "content": content})
    has_user = any(m["role"] == "user" for m in msgs)
    has_assistant = any(m["role"] == "assistant" for m in msgs)
    if not (has_user and has_assistant):
        return None
    return msgs


def _to_messages_openmath(sample: dict) -> Optional[list[dict]]:
    """OpenMathInstruct-2: (problem, generated_solution)."""
    problem = sample.get("problem")
    solution = sample.get("generated_solution")
    if not problem or not solution:
        return None
    return [
        {"role": "user",      "content": problem},
        {"role": "assistant", "content": solution},
    ]


def _to_messages_opencode(sample: dict) -> Optional[list[dict]]:
    """
    OpenCodeInstruct: (input, output) plus unit-test execution metadata.
    Where an execution status is recorded and not passing, the sample is
    dropped — the dataset's verification loop is its main quality signal.
    """
    inp = sample.get("input")
    out = sample.get("output")
    if not inp or not out:
        return None
    # tests_execution_status is a JSON-encoded LIST of per-unit-test results,
    # e.g. '["pass", "pass", "fail", ...]' (verified live, 2026-06-12). Require
    # EVERY test to pass — the dataset's execution verification is its whole
    # quality signal. (The earlier scalar comparison rejected 100% of samples
    # because it matched the whole list-string against "pass".)
    _PASS = ("pass", "passed", "success", "all_passed")
    status = sample.get("tests_execution_status")
    if status:
        try:
            parsed = json.loads(status) if isinstance(status, str) else status
        except (json.JSONDecodeError, TypeError):
            parsed = status
        if isinstance(parsed, list):
            if not all(str(s).lower() in _PASS for s in parsed):
                return None
        elif str(parsed).lower() not in _PASS:
            return None
    return [
        {"role": "user",      "content": inp},
        {"role": "assistant", "content": out},
    ]


def _to_messages_pubmedqa(sample: dict) -> Optional[list[dict]]:
    """
    PubMedQA (pqa_artificial): abstract contexts + question → long_answer
    with the final yes/no/maybe decision appended.
    """
    question = sample.get("question")
    long_answer = sample.get("long_answer")
    decision = sample.get("final_decision")
    if not question or not long_answer:
        return None
    ctx = sample.get("context") or {}
    contexts = ctx.get("contexts") if isinstance(ctx, dict) else None
    user = question
    if contexts:
        user = "\n\n".join(str(c) for c in contexts[:4]) + "\n\nQuestion: " + question
    assistant = long_answer
    if decision:
        assistant = f"{long_answer}\n\nFinal answer: {decision}"
    return [
        {"role": "user",      "content": user},
        {"role": "assistant", "content": assistant},
    ]


def _to_messages_miriad(sample: dict) -> Optional[list[dict]]:
    """
    MIRIAD-4.4M: (question, answer) medical QA grounded in S2ORC papers.
    Answers are self-contained; the source passage is omitted to keep
    prompts inside the seq budget.
    """
    question = sample.get("question")
    answer = sample.get("answer")
    if not question or not answer:
        return None
    return [
        {"role": "user",      "content": question},
        {"role": "assistant", "content": answer},
    ]


def _to_messages_chemdata(sample: dict) -> Optional[list[dict]]:
    """
    ChemData700K: alpaca-style (instruction, input, output) with an optional
    `history` of prior (q, a) pairs prepended as turns.
    """
    instruction = (sample.get("instruction") or "").strip()
    inp = (sample.get("input") or "").strip()
    out = sample.get("output")
    if instruction and inp:
        user = instruction + "\n\n" + inp
    else:
        user = instruction or inp
    if not user or not out:
        return None
    msgs = []
    for pair in sample.get("history") or []:
        try:
            q, a = pair[0], pair[1]
        except (IndexError, TypeError, KeyError):
            continue
        if q and a:
            msgs.append({"role": "user", "content": q})
            msgs.append({"role": "assistant", "content": a})
    msgs.append({"role": "user", "content": user})
    msgs.append({"role": "assistant", "content": out})
    return msgs


_ADAPTERS = {
    "general": _to_messages_openhermes,
    "code":    _to_messages_magicoder,
    "math":    _to_messages_metamath,
    # Clean mix (docs/clean_sft_datasets.md)
    "clean_general":  _to_messages_passthrough,
    "clean_math":     _to_messages_openmath,
    "clean_numina":   _to_messages_passthrough,
    "clean_code":     _to_messages_opencode,
    "clean_miriad":   _to_messages_miriad,
    "clean_pubmedqa": _to_messages_pubmedqa,
    "clean_chem":     _to_messages_chemdata,
}


# ---------------------------------------------------------------------------
# Packing
# ---------------------------------------------------------------------------


def _build_sft_example(
    messages: list[dict],
    tokenizer,
    seq_len: int,
    *,
    _reject_counter: Optional[dict] = None,
) -> Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Render `messages` through the chat template, tokenise, and build
    `(input_ids, target_ids, loss_mask)` tensors of length `seq_len`.

    Two-pass rendering to compute the loss mask precisely:
      1. Render the *prompt* (everything up to but not including the
         assistant content) with `add_generation_prompt=True`. The token
         count of this string is `P`, the prompt length.
      2. Render the *full* conversation (prompt + assistant content + EOS)
         with `add_generation_prompt=False`. Tokens `[P:]` are the
         response — exactly what the loss should target.

    Examples that don't end on an assistant turn are dropped: there's
    nothing to learn to predict.

    Examples where the prompt alone exceeds `seq_len` are dropped: there
    is no room left for a response in the loss-bearing region, so the
    gradient would be zero. Cheaper to skip than to train on noise.

    Returns None to signal "skip this sample" without raising.

    `_reject_counter`, when provided, is a dict that gets incremented
    per rejection reason. Used by the dataset iterator's diagnostic
    log path to surface *why* samples are being rejected — without it,
    a high rejection rate is invisible until training silently never
    starts.
    """
    def _bump(reason: str) -> None:
        if _reject_counter is not None:
            _reject_counter[reason] = _reject_counter.get(reason, 0) + 1

    if not messages or messages[-1].get("role") != "assistant":
        _bump("not_assistant_last")
        return None

    # Pass 1: prompt only.
    prompt_msgs = messages[:-1]
    if not prompt_msgs:
        _bump("empty_prompt_msgs")
        return None

    # Render-as-text, then tokenize. The Ouro tokenizer's
    # `apply_chat_template(..., tokenize=True)` returns a `BatchEncoding`
    # object (`len(...) == 2`, not the token count) which silently breaks
    # the length comparisons below — 100% of samples get rejected with
    # `P >= F` because P=2 and F=2. Rendering to a string and encoding
    # with `tokenizer.encode` avoids the BatchEncoding return path and
    # gives us proper `list[int]` regardless of which underlying HF
    # tokenizer is in use.
    try:
        prompt_text = tokenizer.apply_chat_template(
            prompt_msgs,
            add_generation_prompt=True,
            tokenize=False,
        )
        full_text = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=False,
            tokenize=False,
        )
        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        full_ids   = tokenizer.encode(full_text,   add_special_tokens=False)
    except Exception:                                          # noqa: BLE001
        _bump("template_error")
        return None

    P = len(prompt_ids)
    F = len(full_ids)
    if P >= seq_len:
        _bump("prompt_too_long")
        return None
    if P >= F:
        _bump("empty_or_shorter_response")
        return None

    # Truncate the full sequence (including response) to fit seq_len + 1
    # tokens — we need T+1 tokens to produce input/target pairs of length T.
    full_ids = full_ids[: seq_len + 1]
    F = len(full_ids)
    if F < 2:
        _bump("too_short_after_truncate")
        return None

    # input/target shift by one.
    input_ids  = torch.tensor(full_ids[:-1], dtype=torch.long)
    target_ids = torch.tensor(full_ids[1:],  dtype=torch.long)
    T = input_ids.shape[0]

    # Loss mask: 1 at positions where target is a response token.
    # `target_ids[i] = full_ids[i + 1]`, so i predicts a response token
    # iff (i + 1) >= P, i.e. i >= P - 1.
    loss_mask = torch.zeros(T, dtype=torch.float32)
    if P - 1 < T:
        loss_mask[P - 1 :] = 1.0

    # Pad to seq_len so every batch is the same shape. Padding contributes
    # nothing to the loss because loss_mask stays 0 in the padded region.
    if T < seq_len:
        pad_len = seq_len - T
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            # Fall back to EOS — common when pad_token_id is unset. Padding
            # ids never enter the loss thanks to the mask, so the choice
            # only matters for the attention mask (which MythOuro derives
            # from positions, not values).
            pad_id = tokenizer.eos_token_id or 0
        input_ids = torch.cat([
            input_ids, torch.full((pad_len,), pad_id, dtype=torch.long),
        ])
        target_ids = torch.cat([
            target_ids, torch.full((pad_len,), pad_id, dtype=torch.long),
        ])
        loss_mask = torch.cat([
            loss_mask, torch.zeros(pad_len, dtype=torch.float32),
        ])

    return input_ids, target_ids, loss_mask


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class MixedSFTDataset(IterableDataset):
    """
    Interleaves three streaming HF instruction datasets and yields
    SFT-ready `(input_ids, target_ids, loss_mask)` tensors.

    Sharding model
    --------------
    Same as `MixedDataset`: every `(rank, worker_id)` pair owns a disjoint
    shard of each source. No cross-process coordination needed.

    Per-step source selection
    -------------------------
    Source is drawn per yielded sample, weighted by `_SFT_MIX_RATIOS`.
    Empirical mix is the user's ratios over many steps, not exact within
    a single batch — the correct interpretation for SGD-style mixing.

    Robustness
    ----------
    Any source failing to load is logged and its weight renormalised
    across the survivors. Per-sample exceptions inside the chat-template
    render path are silently skipped (corrupt samples are individually
    cheap to discard).
    """

    def __init__(
        self,
        tokenizer,
        seq_len: int,
        rank: int = 0,
        world_size: int = 1,
        *,
        mix: str = "clean",
        mix_ratios: Optional[dict] = None,
        contamination_filter: "Optional[bool]" = None,
        seed: int = 0,
    ):
        """
        mix -- "clean" (default since 2026-06-11: zero OpenAI-output
               provenance, see docs/clean_sft_datasets.md) or "legacy"
               (the v2/v4-era OpenHermes/Magicoder/MetaMathQA mix, kept for
               reproduction; carries the OpenAI-ToS distribution constraint).
        contamination_filter -- drop samples whose assistant text contains
               verbatim 13-grams from the GSM8K/ARC eval benchmarks
               (data/contamination.py). Default: ON for the clean mix
               (OpenMathInstruct-2 is augmented FROM GSM8K-style problems),
               OFF for legacy (preserves v2/v4 reproduction byte-for-byte).
        """
        if mix not in ("clean", "clean_chat", "legacy"):
            raise ValueError(
                f"mix must be 'clean', 'clean_chat', or 'legacy'; got {mix!r}"
            )
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.rank = rank
        self.world_size = world_size
        self.mix = mix
        is_clean = mix in ("clean", "clean_chat")  # both use the clean sources
        self.specs = _CLEAN_DATASET_SPECS if is_clean else _SFT_DATASET_SPECS
        default_ratios = {
            "clean":      _CLEAN_MIX_RATIOS,
            "clean_chat": _CLEAN_CHAT_MIX_RATIOS,
            "legacy":     _SFT_MIX_RATIOS,
        }[mix]
        self.ratios = mix_ratios or default_ratios
        self.contamination_filter = (
            is_clean if contamination_filter is None else contamination_filter
        )
        self.seed = seed

    def _open_source(
        self,
        repo: str,
        config: Optional[str],
        split: str,
        total_shards: int,
        shard_index: int,
        cap: Optional[int] = None,
    ) -> Optional[Iterator]:
        """
        Open one HF dataset and return an iterator. Returns None on failure.

        Non-streaming mode (`streaming=False`)
        --------------------------------------
        SFT corpora are small (each <1 GB) and entirely fit in RAM. We
        materialise them up front so training does zero network I/O.
        This is a deliberate trade vs. the pretraining `MixedDataset`:

          * pretraining streams FineWeb-Edu / OWM / codeparrot because
            those datasets are TB-scale and impossible to materialise;
          * SFT uses tens of thousands of curated instruction examples
            that fit on disk in <2 GB total.

        Streaming was tried first and proved fragile on home internet:
        HF's range-request iterator hangs silently for hours when TCP
        connections stall, with no surfaced error. Non-streaming uses
        plain sequential downloads via huggingface_hub, which fail
        loudly and respect retries+timeouts.

        Datasets are loaded from the local HuggingFace cache. If a
        dataset isn't cached yet, this call will download it (with a
        visible progress bar). Pre-download once with
        `python -c "from datasets import load_dataset; load_dataset(<repo>, split='train')"`
        for any new corpus you add to `_SFT_DATASET_SPECS`.

        Sharding still applies when `total_shards > 1` so distributed
        runs read disjoint slices, but the single-process default path
        (total_shards == 1) skips the call entirely.
        """
        from datasets import load_dataset

        # Capped sources (the clean mix's million-scale sets) load a split
        # slice: still non-streaming (cached on disk, loud failures, Arrow
        # memory-mapped so RAM stays bounded), just bounded rows.
        effective_split = f"{split}[:{cap}]" if cap else split

        try:
            ds = load_dataset(
                repo, name=config, split=effective_split, streaming=False,
            )
        except Exception as exc:                              # noqa: BLE001
            logger.warning(
                f"MixedSFTDataset: skipping {repo!r} — failed to load "
                f"({exc}). Pre-download with `load_dataset({repo!r}, "
                f"split={split!r})` if this is the first run."
            )
            return None

        if total_shards > 1:
            try:
                ds = ds.shard(num_shards=total_shards, index=shard_index)
            except Exception as exc:                          # noqa: BLE001
                logger.warning(
                    f"MixedSFTDataset: {repo!r} doesn't support "
                    f"shard({total_shards}, {shard_index}) ({exc}); "
                    "falling back to un-sharded data"
                )

        logger.info(
            f"MixedSFTDataset: loaded {repo!r} "
            f"({len(ds):,} examples) into memory"
        )
        return iter(ds)

    def __iter__(self):
        worker = get_worker_info()
        num_workers = worker.num_workers if worker else 1
        worker_id = worker.id if worker else 0
        total_shards = self.world_size * num_workers
        shard_index = self.rank * num_workers + worker_id

        # Open each source independently.
        active: list[dict] = []
        for key, repo, config, split, cap in self.specs:
            if self.ratios.get(key, 0.0) <= 0:
                continue
            it = self._open_source(
                repo, config, split, total_shards, shard_index, cap=cap,
            )
            if it is None:
                continue
            active.append({
                "key": key, "repo": repo, "config": config, "split": split,
                "cap": cap, "iter": it, "weight": self.ratios[key],
                "adapter": _ADAPTERS[key],
            })

        if not active:
            raise RuntimeError(
                "MixedSFTDataset: no sources opened successfully. "
                "Check network access and HuggingFace dataset availability."
            )

        # Eval-benchmark contamination guard (clean mix default). Built once
        # per iterator; checks ASSISTANT-side text for verbatim 13-grams from
        # the benchmarks we eval on. Mandatory for the clean mix because
        # OpenMathInstruct-2 is augmented from GSM8K-style problems.
        contam = None
        if self.contamination_filter:
            try:
                from data.contamination import ContaminationFilter
                contam = ContaminationFilter(["gsm8k", "arc"])
                contam.build_index()
                logger.info(
                    f"MixedSFTDataset: contamination filter active "
                    f"({contam.stats['ngrams']:,} benchmark 13-grams)"
                )
            except Exception as exc:                          # noqa: BLE001
                logger.warning(
                    f"MixedSFTDataset: contamination filter unavailable "
                    f"({exc}) — proceeding WITHOUT it. Do not distribute a "
                    "checkpoint trained this way without re-checking overlap."
                )
                contam = None

        rng = random.Random(self.seed + shard_index)

        # Diagnostic counters per source. Logged once every
        # `_DIAG_EVERY` attempted samples so we can see what the
        # acceptance / rejection breakdown looks like in real time.
        # Critical when the iterator silently rejects everything (e.g.
        # all samples have prompts longer than seq_len) — without this
        # log the symptom is "training never starts" with no clue why.
        _DIAG_EVERY = 1000
        stats: dict = {}
        for s in active:
            stats[s["key"]] = {
                "attempted": 0,
                "yielded": 0,
                "no_messages": 0,
                "reject_reasons": {},
            }
        total_attempted_global = 0

        while True:
            # Weighted pick. Renormalises naturally as failed sources are
            # dropped on the fly.
            weights = [s["weight"] for s in active]
            src = rng.choices(active, weights=weights, k=1)[0]
            key = src["key"]
            stats[key]["attempted"] += 1
            total_attempted_global += 1

            try:
                sample = next(src["iter"])
            except StopIteration:
                # Re-open exhausted source from the top of its shard.
                src["iter"] = self._open_source(
                    src["repo"], src["config"], src["split"],
                    total_shards, shard_index, cap=src["cap"],
                )
                if src["iter"] is None:
                    active.remove(src)
                    if not active:
                        return
                continue
            except Exception as exc:                          # noqa: BLE001
                logger.warning(
                    f"MixedSFTDataset: {key} stream error ({exc}); "
                    "skipping batch"
                )
                continue

            messages = src["adapter"](sample)
            if messages is not None and contam is not None:
                assistant_text = "\n".join(
                    m["content"] for m in messages if m["role"] == "assistant"
                )
                if contam.is_contaminated(assistant_text):
                    reasons = stats[key]["reject_reasons"]
                    reasons["contaminated"] = reasons.get("contaminated", 0) + 1
                    messages = None
            if messages is None:
                stats[key]["no_messages"] += 1
            else:
                built = _build_sft_example(
                    messages, self.tokenizer, self.seq_len,
                    _reject_counter=stats[key]["reject_reasons"],
                )
                if built is None:
                    pass  # already counted by reason in reject_reasons
                else:
                    stats[key]["yielded"] += 1
                    yield built

            if total_attempted_global % _DIAG_EVERY == 0:
                # Build a compact summary per source so the user can SEE
                # what's flowing in real time. Includes the top rejection
                # reason for each source so we know *why* samples are
                # being dropped, not just how many.
                lines = []
                for k, st in stats.items():
                    rate = (
                        100.0 * st["yielded"] / st["attempted"]
                        if st["attempted"] > 0 else 0.0
                    )
                    reasons = st["reject_reasons"]
                    reasons_str = (
                        " ".join(f"{r}={c}" for r, c in sorted(
                            reasons.items(), key=lambda kv: -kv[1]
                        ))
                        if reasons else "-"
                    )
                    lines.append(
                        f"{k}: {st['yielded']}/{st['attempted']} "
                        f"({rate:.1f}% accept) [{reasons_str}]"
                    )
                logger.info(
                    f"MixedSFTDataset diag: {' | '.join(lines)}"
                )
