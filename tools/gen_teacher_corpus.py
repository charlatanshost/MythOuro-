"""
Teacher-corpus generator (docs/teacher_corpus_plan.md).

Batched Ouro continuation of real-corpus seeds → local JSONL shards that
`MixedDataset` streams behind `--teacher-data-ratio`. Attacks token SUPPLY
(ideas.md: teacher-generated synthetic data + sequence-level KD, one build).

Runs on whichever card is free (designed for the 5070 while the Max trains):

    python -m tools.gen_teacher_corpus --device cuda:0 --trust-remote-code \
        --target-tokens 40_000_000

Output: `<out-dir>/shard_NNNN.jsonl` rows {"text", "source", "seed_len"} and a
`MANIFEST.json` with generation params + accept/reject stats (provenance, in
the dataset_selection.md spirit). Text = real seed + teacher continuation;
`seed_len` marks the boundary. Filters are deliberately dumb/fast: min length,
distinct-1 floor, top-token-share ceiling. Spot-read before training on it.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import signal
import sys
import time
from collections import Counter
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mythouro.device as dev                  # noqa: E402
from mythouro.training_utils import (          # noqa: E402
    _DATASET_SPECS,
    _MIX_RATIOS,
    extract_field_text,
    _teacher_cache_usable,
    load_distillation_teacher,
    teacher_logits,
    teacher_logits_cached,
)

ROWS_PER_SHARD = 1000


def _sample_top_p(probs: torch.Tensor, top_p: float, on_device: bool) -> torch.Tensor:
    """
    Top-p (nucleus) categorical sample → indices into the vocab, shape (B, 1).

    `on_device=True` samples where `probs` lives via inverse-CDF
    (sort → cumsum → threshold → renormalise → searchsorted on a uniform draw)
    — every op verified segfault-free on XPU/PVC 2026-07-22, unlike
    `topk`/`multinomial`. `on_device=False` is the legacy CPU round-trip
    (`.cpu()` + `multinomial`). The two are the SAME distribution — inverse-CDF
    over the renormalised nucleus is exact categorical sampling — only the RNG
    stream differs.
    """
    if not on_device:
        probs = probs.cpu()
    sorted_p, order = probs.sort(dim=-1, descending=True)
    cum = sorted_p.cumsum(dim=-1)
    sorted_p[cum - sorted_p > top_p] = 0.0
    sorted_p = sorted_p / sorted_p.sum(dim=-1, keepdim=True)
    if on_device:
        cdf = sorted_p.cumsum(dim=-1)
        u = torch.rand(sorted_p.shape[0], 1, device=sorted_p.device)
        pick = torch.searchsorted(cdf, u).clamp(max=sorted_p.shape[-1] - 1)
    else:
        pick = torch.multinomial(sorted_p, 1)
    return order.gather(-1, pick)


def _generate_xpu_safe(teacher, input_ids: torch.Tensor, *, max_new: int,
                       temperature: float, top_p: float,
                       cpu_sampling: bool = False,
                       cache_factory=None) -> torch.Tensor:
    """
    Manual batched decode for XPU, where HF `generate()` segfaults (on-device
    topk/multinomial — workaround list, docs/max1100_field_notes.md). Mirrors
    the production `generate_rollout` pattern: KL-gated cached teacher forward
    (`teacher_logits_cached`, falls back to full recompute if the gate fails).
    Sampling runs on-device by default (removes a host↔device sync per token —
    meaningful in a launch-bound loop); `cpu_sampling=True` restores the legacy
    CPU path. Identical distribution either way (see `_sample_top_p`).
    """
    seq = input_ids
    cached = _teacher_cache_usable(teacher, seq)
    # Preallocated cache (gated upstream): fresh instance per generation call;
    # sized for prompt + max_new so update() can never overflow.
    past = cache_factory() if (cache_factory and cached) else None
    inv_t = 1.0 / max(temperature, 1e-5)
    with torch.no_grad():
        for i in range(max_new):
            if cached:
                cur = seq if i == 0 else seq[:, -1:]
                start = 0 if i == 0 else seq.shape[1] - 1
                logits, past = teacher_logits_cached(teacher, cur, past, start)
            else:
                logits = teacher_logits(teacher, seq)
            probs = torch.softmax(logits[:, -1, :].float() * inv_t, dim=-1)
            nxt = _sample_top_p(probs, top_p, on_device=not cpu_sampling)
            seq = torch.cat([seq, nxt.to(seq.device)], dim=1)
    return seq


def _seed_streams(tok, seed_len: int, rng: "random.Random",
                  *, stream_seed: "int | None" = None,
                  stream_buffer: int = 1000):
    """
    Per-corpus generators yielding fixed-length token seeds from a RANDOM
    WINDOW of each document.

    **Cross-session traversal (fixed 2026-07-23).** `load_dataset(streaming=True)`
    iterates from document #1 every time the *process* starts, and `rng` only
    picks the window *within* a document — so every session re-harvested the
    same source documents in the same order. Measured on the first 4.28M of v2:
    **6,038 rows came from only 3,886 distinct seed documents** (1,588 prefixes
    repeated, every repeat spanning different shards = different sessions;
    564 documents were used three times). Continuations differ (temperature
    0.9) so there were no duplicate *texts*, which is why row-level dedup never
    caught it — the redundancy is one level up, in the source material.

    `stream_seed` fixes it via `.shuffle()`, which reorders the **shards** (all
    three corpora are many-file) as well as buffering, so a new session begins
    on a different shard instead of replaying document #1. The seed is bumped
    per epoch so a stream that exhausts and restarts does not repeat itself
    either. `stream_seed=None` disables shuffling (rollback path).

    Seeding from the document HEAD (the v1 behaviour) is systematically
    biased toward boilerplate: source files open with license headers and
    imports, scraped math pages with nav cruft. Measured on the first
    5.84M-token harvest (2026-07-21): **57% of code samples were the teacher
    faithfully continuing an Apache/copyright header** — ~600k tokens of
    legalese — while math/general were only ~0.5% affected. A random offset
    lands mid-document where the actual content lives.
    """
    from datasets import load_dataset

    def stream(repo, config, split, field):
        epoch = 0
        while True:
            ds = load_dataset(repo, name=config, split=split, streaming=True)
            if stream_seed is not None:
                ds = ds.shuffle(seed=stream_seed + epoch,
                                buffer_size=stream_buffer)
            epoch += 1
            skipped = 0
            for sample in ds:
                # extract_field_text, NOT sample.get(field): `field` is a TUPLE
                # for the instruct corpora (("problem","generated_solution")),
                # and a dict .get() with a tuple key ALWAYS misses. That bare
                # .get returned "" on every row of math_instruct and
                # code_instruct — 26.5% of the mix — so those streams spun
                # forever in this `while True` and the harvest's first batch
                # never completed. 6 hours at 87% CPU, zero rows.
                text = extract_field_text(sample, field)
                if not text:
                    skipped += 1
                    if skipped % 50_000 == 0:
                        # A stream that yields nothing used to be SILENT. Now it
                        # says so, because the failure mode is an invisible hang.
                        print(f"  [seed-stream] {repo}: {skipped:,} rows with no "
                              f"usable text (field={field!r}) — check the field spec",
                              flush=True)
                    continue
                # Cheap pre-filter BEFORE tokenizing: a `seed_len`-token document
                # needs roughly 3 characters per token, so anything far shorter
                # cannot possibly qualify. Skips the majority of rejects without
                # paying for a 2048-token tokenize, which is what made a raised
                # --prompt-len so expensive.
                if len(text) < seed_len * 3:
                    continue
                # Tokenize a generous prefix so there is room to pick a window.
                ids = tok(text, truncation=True, max_length=2048)["input_ids"]
                if len(ids) < seed_len:
                    continue
                hi = len(ids) - seed_len
                start = rng.randint(0, hi) if hi > 0 else 0
                yield ids[start:start + seed_len]

    return {
        key: stream(repo, config, split, field)
        for key, repo, config, split, field in _DATASET_SPECS
    }


_BOILERPLATE = re.compile(
    r"licen[sc]e|copyright|Apache License|permission is hereby|"
    r"WITHOUT WARRANTIES|redistribut", re.I)


# ---------------------------------------------------------------------------
# Instruction framing (--chat-template)
# ---------------------------------------------------------------------------
#
# WHY THIS EXISTS. Until 2026-08-11 every teacher row was a CONTINUATION: 48
# tokens of real text, teacher writes the next 768. That means the student has
# never once seen an instruction -> response pair from its teacher. Its whole
# training history is raw web text plus teacher continuations of raw web text —
# and the ONLY instruction-shaped data it ever met was SFT, applied all at once,
# in a ChatML format it had never seen, to a model with zero instruction prior.
# SFT then collapsed it at 3k, 36.2k and at 32x batch (docs/generation_probe_
# tracker.md 2026-08-10). Routing instruction-following through the TEACHER
# instead puts it on the one channel that demonstrably works on this model:
# on-policy distillation took code L3+ from 51.2% to 75.0%.
#
# GROUNDED, NOT INVENTED. The instruction wraps a REAL corpus snippet, so the
# teacher supplies the FORM and the corpus supplies the CONTENT. This also
# attacks a known defect: free continuation is what produced the medical
# harvest's fabrications ("the PAWL study", 30-mg ibuprofen — real dosing is
# 200-400mg). Summarising a real abstract is anchored in a way continuing from a
# 48-token seed is not.
#
# FIXED-LENGTH PROMPTS on purpose. Variable-length prompts would need left
# padding and attention masks through _generate_xpu_safe, which is the fragile
# hand-rolled XPU path. Truncating the snippet so every prompt is exactly
# --prompt-len tokens keeps the existing fixed-size batching working unchanged.
_SYSTEM = "You are a helpful assistant."

_INSTRUCTION_TEMPLATES = {
    "general": [
        "Explain the following passage in your own words.\n\n",
        "What is the main idea of this text? Answer in a few sentences.\n\n",
        "Summarise the following passage.\n\n",
    ],
    "medical": [
        "Summarise the key clinical findings in this abstract.\n\n",
        "Explain the following medical passage in plain language.\n\n",
        "What condition and treatment does this passage describe?\n\n",
    ],
    "math": [
        "Explain the mathematics in the following passage, step by step.\n\n",
        "What problem is being solved here, and how? Show the reasoning.\n\n",
    ],
    "math_instruct": [
        "Solve the following problem. Show your reasoning, then give the answer.\n\n",
        "Work through this problem step by step.\n\n",
    ],
    "code": [
        "Explain what the following code does.\n\n",
        "Describe this code, then write a corrected or completed version.\n\n",
    ],
    "code_instruct": [
        "Write the code this describes, and explain it briefly.\n\n",
        "Complete the following task. Give the code, then a short explanation.\n\n",
    ],
}


def _chat_prompt(tok, snippet_ids: list[int], source: str, prompt_len: int,
                 rng) -> "list[int]":
    """
    One fixed-length ChatML prompt of EXACTLY `prompt_len` tokens.

    Layout:  <|im_start|>system ... <|im_end|>
             <|im_start|>user {instruction}\n\n{snippet}<|im_end|>
             <|im_start|>assistant\n

    The snippet is truncated to whatever budget remains after the framing, so
    the total is always `prompt_len` and the caller's batching is unchanged.
    Falls back to the `general` templates for any source without its own.
    """
    templates = _INSTRUCTION_TEMPLATES.get(source) or _INSTRUCTION_TEMPLATES["general"]
    instruction = templates[rng.randrange(len(templates))]
    prefix = tok.encode(
        f"<|im_start|>system\n{_SYSTEM}<|im_end|>\n<|im_start|>user\n{instruction}"
    )
    suffix = tok.encode("<|im_end|>\n<|im_start|>assistant\n")
    budget = prompt_len - len(prefix) - len(suffix)
    if budget < 16:
        raise ValueError(
            f"--prompt-len {prompt_len} leaves only {budget} tokens for the "
            f"snippet after {len(prefix)}+{len(suffix)} of ChatML framing. "
            f"Raise it; a prompt with no content teaches nothing."
        )
    if len(snippet_ids) < budget:
        # NEVER pad. An earlier version padded to `budget` with pad/eot ids,
        # which put up to 161 <|endoftext|> tokens between the snippet and the
        # `assistant` cue — the teacher would be asked to answer a question
        # whose context trails off into end-of-document markers. The caller
        # draws seeds of exactly --prompt-len tokens, so budget (prompt_len
        # minus framing) is ALWAYS satisfied; this is an invariant check, not a
        # fallback.
        raise ValueError(
            f"snippet has {len(snippet_ids)} tokens but the prompt budget is "
            f"{budget}. Seeds must be drawn at --prompt-len; padding here would "
            f"put end-of-document markers between the question and the answer."
        )
    return prefix + snippet_ids[:budget] + suffix


def _reject_reason(cont_ids: list[int], min_new: int, min_distinct1: float,
                   max_top_share: float, text: str = "") -> "str | None":
    """None = passes; else which filter rejected (for the tuning telemetry)."""
    if len(cont_ids) < min_new:
        return "too_short"
    if text and len(_BOILERPLATE.findall(text[:800])) >= 2:
        return "boilerplate"
    counts = Counter(cont_ids)
    if len(counts) / len(cont_ids) < min_distinct1:
        return "low_distinct1"
    if counts.most_common(1)[0][1] / len(cont_ids) > max_top_share:
        return "high_top_share"
    return None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--teacher-id", default="ByteDance/Ouro-2.6B-Thinking")
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--out-dir", default="data_teacher")
    p.add_argument("--target-tokens", type=int, default=5_000_000,
                   help="Stop after this many ACCEPTED continuation tokens.")
    p.add_argument("--batch", type=int, default=12)
    p.add_argument("--seed-len", type=int, default=48)
    p.add_argument("--chat-template", action="store_true",
                   help="Harvest INSTRUCTION -> RESPONSE pairs instead of raw "
                        "continuations. Wraps a real corpus snippet in a ChatML "
                        "instruction and keeps the teacher's answer, so the "
                        "corpus supplies the CONTENT and the teacher supplies "
                        "the FORM. Until 2026-08-11 every teacher row was a "
                        "continuation, so the student had NEVER seen an "
                        "instruction/response pair from its teacher — the only "
                        "instruction-shaped data it ever met was SFT, which "
                        "collapsed it at every dose and batch size. This routes "
                        "instruction-following through on-policy distillation "
                        "instead, the one channel proven on this model.")
    p.add_argument("--prompt-len", type=int, default=256,
                   help="Fixed ChatML prompt length in tokens for "
                        "--chat-template. Every prompt is truncated/padded to "
                        "exactly this, which keeps the existing fixed-size "
                        "batching (and the fragile hand-rolled XPU generate "
                        "path) working with no left-padding or attention masks. "
                        "~30 tokens go to framing; the rest is the snippet.")
    p.add_argument("--max-new", type=int, default=768)
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--min-new", type=int, default=128,
                   help="Reject continuations shorter than this after EOS trim.")
    p.add_argument("--min-distinct1", type=float, default=0.20,
                   help="Floor on unique/total tokens of the continuation. "
                        "Calibrated against REAL corpus text at ~768 tok "
                        "(2026-07-19: general p10=0.38, math p10=0.26, code "
                        "p10=0.23 — distinct-1 falls with length and is "
                        "naturally low for code). 0.20 sits under all three; "
                        "top-share is the actual degeneracy guard.")
    p.add_argument("--max-top-share", type=float, default=0.50)
    p.add_argument("--prealloc-cache", action="store_true",
                   help="Preallocate the teacher KV cache (no cat-doubling -> "
                        "bigger batch fits). Runs a KL equivalence gate at "
                        "startup and falls back to the stock cache on failure.")
    p.add_argument("--cpu-sampling", action="store_true",
                   help="Restore the legacy CPU top-p path (.cpu()+multinomial) "
                        "instead of on-device inverse-CDF sampling. Identical "
                        "distribution; keep as rollback only.")
    p.add_argument("--seed-mix", default=None,
                   help="Override the SEED-draw mix, e.g. "
                        "'general=0.32,math=0.42,code=0.26'. Defaults to "
                        "_MIX_RATIOS (40/40/20). Needed because acceptance and "
                        "mean length differ per source, so drawing 40/40/20 "
                        "does NOT yield a 40/40/20 ACCEPTED corpus (measured "
                        "2026-07-23 on v2: 45.3/38.0/16.7 by token). This flag "
                        "is harvest-local on purpose — _MIX_RATIOS is shared "
                        "with the training MixedDataset and must not move.")
    p.add_argument("--telemetry", action="store_true",
                   help="Log per-sample filter stats (source, length, "
                        "distinct-1 at 256 tokens and final, top-share, reject "
                        "reason) to <out-dir>/telemetry_<start>.jsonl for EVERY "
                        "sample, accepted or rejected. Rejects never reach the "
                        "shards, so this is the only way to measure early-abort "
                        "separability (harvest_speedup_plan.md lever 1) or to "
                        "predict a filter change's acceptance impact. Cheap: "
                        "~30 short lines per ~143 s generation cycle.")
    p.add_argument("--stream-seed", type=int, default=None,
                   help="Seed for shuffling the seed-corpus SHARD ORDER. "
                        "Default: a fresh random value per session, recorded in "
                        "the manifest so the run stays reproducible after the "
                        "fact. Without this every session re-read the corpora "
                        "from document #1 — measured 2026-07-23: 6,038 v2 rows "
                        "came from only 3,886 distinct seed documents.")
    p.add_argument("--stream-buffer", type=int, default=1000,
                   help="Reservoir size for streaming shuffle. Shard-order "
                        "shuffling is what prevents cross-session repeats; this "
                        "adds local mixing. Larger = more mixing but a slower "
                        "first batch (the buffer must fill before the first "
                        "yield).")
    p.add_argument("--no-stream-shuffle", action="store_true",
                   help="Disable seed-stream shuffling entirely (rollback to "
                        "the pre-2026-07-23 sequential traversal).")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    # Resolved before any expensive work so it can be printed and recorded.
    stream_seed = None
    if not args.no_stream_shuffle:
        stream_seed = (args.stream_seed if args.stream_seed is not None
                       else random.randrange(2 ** 31))
        print(f"seed-stream shuffle ON (stream_seed={stream_seed}, "
              f"buffer={args.stream_buffer})", flush=True)
    else:
        print("seed-stream shuffle OFF — sequential traversal; this session "
              "will re-read the same documents as any other unshuffled run",
              flush=True)

    # Resolve the seed mix BEFORE the teacher load — a typo here should cost a
    # second, not a two-minute model load on an unattended overnight launch.
    # Valid seed sources are everything in _DATASET_SPECS — NOT just the keys of
    # _MIX_RATIOS. Some specs are harvest-only (e.g. "medical"), deliberately
    # absent from _MIX_RATIOS so MixedDataset skips them and the training mix is
    # untouched; they must still be selectable here.
    _spec_keys = {k for k, *_ in _DATASET_SPECS}
    seed_mix = dict(_MIX_RATIOS)
    if args.seed_mix:
        override = {}
        for part in args.seed_mix.split(","):
            k, sep, v = part.partition("=")
            k = k.strip()
            if not sep:
                raise SystemExit(
                    f"--seed-mix: expected 'source=weight', got {part!r}")
            if k not in _spec_keys:
                raise SystemExit(
                    f"--seed-mix: unknown source {k!r}; "
                    f"expected one of {sorted(_spec_keys)}")
            try:
                override[k] = float(v)
            except ValueError:
                raise SystemExit(
                    f"--seed-mix: weight for {k!r} is not a number: {v!r}")
            if override[k] < 0:
                raise SystemExit(f"--seed-mix: negative weight for {k!r}")
        # --seed-mix FULLY specifies the draw distribution: whatever is named is
        # what gets sampled (weights are normalised below), and anything omitted
        # is simply not drawn. A subset is therefore legal and meaningful —
        # `--seed-mix medical=1.0` is a pure medical harvest. (It used to require
        # naming every _MIX_RATIOS key, which made harvest-only sources like
        # "medical" unusable.)
        if not override:
            raise SystemExit("--seed-mix named no sources")
        total = sum(override.values())
        if total <= 0:
            raise SystemExit("--seed-mix weights must sum to > 0")
        seed_mix = {k: v / total for k, v in override.items()}
        print(f"seed mix overridden -> {seed_mix} (targets the ACCEPTED mix; "
              f"shared _MIX_RATIOS is untouched)", flush=True)

    torch.manual_seed(args.seed)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(
        args.teacher_id, trust_remote_code=args.trust_remote_code)
    teacher = load_distillation_teacher(
        args.teacher_id, student_vocab_size=tok.vocab_size,
        device=args.device, dtype=torch.bfloat16,
        trust_remote_code=args.trust_remote_code)
    if teacher is None:
        raise SystemExit("teacher failed to load")
    eot = tok.convert_tokens_to_ids("<|endoftext|>")
    # In chat mode the response terminates at <|im_end|>, NOT <|endoftext|>.
    # Truncating on the wrong token would keep the teacher's next turn —
    # a second <|im_start|>user block — inside the "response", teaching the
    # student to write both halves of the conversation.
    stop_id = tok.convert_tokens_to_ids("<|im_end|>") if args.chat_template else eot

    cache_factory = None
    if args.prealloc_cache:
        from tools.prealloc_ut_cache import (
            make_prealloc_cache, validate_cache_equivalence)
        total = args.seed_len + args.max_new + 8
        probe_ids = torch.randint(
            0, tok.vocab_size, (1, 12), device=args.device)
        if validate_cache_equivalence(teacher, probe_ids, max_len=total):
            cache_factory = lambda: make_prealloc_cache(teacher, max_len=total)  # noqa: E731
        else:
            print("prealloc-cache gate FAILED -> using stock dynamic cache")

    out = Path(args.out_dir)
    out.mkdir(exist_ok=True)
    existing = sorted(out.glob("shard_*.jsonl"))
    shard_idx = int(existing[-1].stem.split("_")[1]) + 1 if existing else 0

    # Chat mode needs a snippet long enough to fill the prompt budget;
    # _chat_prompt truncates down to whatever the framing leaves.
    _stream_len = args.prompt_len if args.chat_template else args.seed_len
    streams = _seed_streams(tok, _stream_len, random.Random(args.seed),
                            stream_seed=stream_seed,
                            stream_buffer=args.stream_buffer)
    keys = list(seed_mix)
    weights = [seed_mix[k] for k in keys]
    rng = torch.Generator().manual_seed(args.seed)
    _tmpl_rng = random.Random(args.seed + 1)   # template choice, independent of seeds
    _prompt_tokens = args.prompt_len if args.chat_template else args.seed_len

    accepted_tok = accepted_n = rejected_n = 0
    reject_reasons: Counter = Counter()
    rows: list[dict] = []
    t0 = time.time()
    # Sessions-aware manifest (2026-07-23): counters used to be per-session and
    # each relaunch OVERWROTE them, under-reporting multi-session corpora (v2
    # read "2.13M" while 2.84M sat on disk). Now each session appends a record
    # and top-level totals sum across sessions. Pre-fix manifests are wrapped
    # as a single "legacy" session (its numbers may cover only the LAST old
    # session — rows on disk stay the ground truth for old corpora).
    manifest_path = out / "MANIFEST.json"
    manifest = {}
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
        except Exception:                                       # noqa: BLE001
            manifest = {}
    if "sessions" not in manifest:
        legacy = {k: manifest[k] for k in
                  ("accepted", "rejected", "accepted_tokens", "reject_reasons")
                  if k in manifest}
        manifest = {"sessions": ([{"legacy": True, **legacy}] if legacy else [])}
    session = {
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "teacher_id": args.teacher_id, "seed_len": args.seed_len,
        "max_new": args.max_new, "temperature": args.temperature,
        "top_p": args.top_p, "batch": args.batch,
        "prealloc_cache": bool(args.prealloc_cache),
        "cpu_sampling": bool(args.cpu_sampling),
        "filters": {
            "min_new": args.min_new, "min_distinct1": args.min_distinct1,
            "max_top_share": args.max_top_share},
        # The mix actually DRAWN this session (may differ from _MIX_RATIOS when
        # --seed-mix compensates for per-source acceptance/length differences).
        "mix": seed_mix,
        "seed_mix_overridden": bool(args.seed_mix),
        # Records WHICH documents this session traversed. Sessions sharing a
        # stream_seed re-read the same source material (see _seed_streams).
        "stream_seed": stream_seed,
        "stream_buffer": args.stream_buffer if stream_seed is not None else None,
    }
    manifest["sessions"].append(session)

    tele_path = out / f"telemetry_{session['started'].replace(':', '').replace(' ', '_')}.jsonl"
    tele = tele_path.open("a") if args.telemetry else None
    if tele is not None:
        print(f"telemetry -> {tele_path}", flush=True)

    def flush():
        nonlocal rows, shard_idx
        if not rows:
            return
        path = out / f"shard_{shard_idx:04d}.jsonl"
        with path.open("a") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        rows = []
        shard_idx += 1
        session.update(accepted=accepted_n, rejected=rejected_n,
                       reject_reasons=dict(reject_reasons),
                       accepted_tokens=accepted_tok,
                       updated=time.strftime("%Y-%m-%d %H:%M:%S"))
        manifest["total_accepted"] = sum(
            x.get("accepted", 0) for x in manifest["sessions"])
        manifest["total_accepted_tokens"] = sum(
            x.get("accepted_tokens", 0) for x in manifest["sessions"])
        manifest_path.write_text(json.dumps(manifest, indent=2))

    # Graceful stop: flush the in-memory rows before exiting on Ctrl-C / SIGTERM.
    # Without this, interrupting mid-shard silently discarded up to
    # ROWS_PER_SHARD-1 accepted rows — ~700k tokens (≈1.7 h of GPU work) were
    # about to be lost stopping the 2026-07-29 medical harvest, and the same gap
    # cost rows in the power outage. Harvests are meant to be stopped whenever
    # the card is wanted, so stopping must never destroy completed work.
    # We only set a flag here: flushing from inside a signal handler could
    # interleave with the writer in `flush()`. The loop checks it and exits
    # cleanly at the next batch boundary.
    stop_requested = {"v": False}

    def _request_stop(signum, _frame):
        if stop_requested["v"]:            # second Ctrl-C = impatient user
            print("\nsecond signal — exiting immediately (buffer LOST)", flush=True)
            raise SystemExit(130)
        stop_requested["v"] = True
        print(f"\nsignal {signum} — finishing this batch, then flushing "
              f"{len(rows)} buffered rows. Ctrl-C again to abandon them.",
              flush=True)

    for _sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(_sig, _request_stop)

    while accepted_tok < args.target_tokens and not stop_requested["v"]:
        sources = [keys[torch.multinomial(
            torch.tensor(weights, dtype=torch.float), 1, generator=rng).item()]
            for _ in range(args.batch)]
        seeds = [next(streams[s]) for s in sources]
        if args.chat_template:
            seeds = [_chat_prompt(tok, sd, src, args.prompt_len, _tmpl_rng)
                     for sd, src in zip(seeds, sources)]
        input_ids = torch.tensor(seeds, device=args.device)
        if dev.backend(args.device) == "xpu":
            # HF generate() segfaults on XPU (on-device topk/multinomial);
            # use the manual cached-teacher + CPU-sampling path.
            gen = _generate_xpu_safe(
                teacher, input_ids, max_new=args.max_new,
                temperature=args.temperature, top_p=args.top_p,
                cpu_sampling=args.cpu_sampling,
                cache_factory=cache_factory)
        else:
            with torch.no_grad():
                gen = teacher.generate(
                    input_ids, max_new_tokens=args.max_new, do_sample=True,
                    temperature=args.temperature, top_p=args.top_p,
                    pad_token_id=tok.pad_token_id or eot or 0)
        for row, src in zip(gen.tolist(), sources):
            cont = row[_prompt_tokens:]
            if stop_id is not None and stop_id in cont:
                cont = cont[:cont.index(stop_id)]
            reason = _reject_reason(cont, args.min_new, args.min_distinct1,
                                    args.max_top_share,
                                    tok.decode(cont[:300]))
            if tele is not None and cont:
                counts = Counter(cont)
                rec = {
                    "source": src, "len": len(cont), "reason": reason,
                    "d1_final": round(len(counts) / len(cont), 4),
                    "top_share": round(
                        counts.most_common(1)[0][1] / len(cont), 4),
                }
                # distinct-1 at candidate early-abort points: the separability
                # signal for continuous-batching lane eviction.
                for n in (128, 256, 384):
                    rec[f"d1_{n}"] = (
                        round(len(Counter(cont[:n])) / n, 4)
                        if len(cont) >= n else None)
                tele.write(json.dumps(rec) + "\n")
            if reason is not None:
                rejected_n += 1
                reject_reasons[reason] += 1
                continue
            accepted_n += 1
            accepted_tok += len(cont)
            if args.chat_template:
                # Keep the FULL ChatML exchange in `text`, terminator included:
                # the student has to learn the format, not just the prose. The
                # <|im_end|> is re-appended because it was stripped above to
                # measure the response.
                # skip_special_tokens=False is LOAD-BEARING: the default
                # decode strips <|im_start|>/<|im_end|>, which would write rows
                # with the prose but NO ChatML markers — the student would learn
                # the content and never the format, which is the entire purpose
                # of this mode.
                rows.append({
                    "text": tok.decode(row[:_prompt_tokens] + cont + [stop_id],
                                       skip_special_tokens=False),
                    "source": src, "seed_len": _prompt_tokens, "chat": True,
                })
            else:
                rows.append({
                    "text": tok.decode(row[:args.seed_len] + cont),
                    "source": src, "seed_len": args.seed_len,
                })
        if len(rows) >= ROWS_PER_SHARD:
            flush()
        if tele is not None:
            tele.flush()   # per batch: an outage costs one cycle, not the run
        el = time.time() - t0
        rj = " ".join(f"{k}={v}" for k, v in reject_reasons.most_common())
        print(f"accepted {accepted_n} ({accepted_tok/1e6:.2f}M tok) "
              f"rejected {rejected_n} [{rj}] | {accepted_tok/max(el,1):.0f} tok/s",
              flush=True)

    flush()                      # catches the buffer on BOTH exit paths
    if tele is not None:
        tele.close()
    if stop_requested["v"]:
        print(f"stopped by signal — flushed cleanly, nothing lost. "
              f"Re-run to resume (the launcher's cumulative target picks up "
              f"from the manifest).", flush=True)
    print(f"done: {accepted_tok/1e6:.2f}M accepted tokens in "
          f"{(time.time()-t0)/3600:.2f} h → {out}")


if __name__ == "__main__":
    main()
