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


def _seed_streams(tok, seed_len: int, rng: "random.Random"):
    """
    Per-corpus generators yielding fixed-length token seeds from a RANDOM
    WINDOW of each document.

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
        while True:
            ds = load_dataset(repo, name=config, split=split, streaming=True)
            for sample in ds:
                text = sample.get(field, "")
                if not text:
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
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    # Resolve the seed mix BEFORE the teacher load — a typo here should cost a
    # second, not a two-minute model load on an unattended overnight launch.
    seed_mix = dict(_MIX_RATIOS)
    if args.seed_mix:
        override = {}
        for part in args.seed_mix.split(","):
            k, sep, v = part.partition("=")
            k = k.strip()
            if not sep:
                raise SystemExit(
                    f"--seed-mix: expected 'source=weight', got {part!r}")
            if k not in _MIX_RATIOS:
                raise SystemExit(
                    f"--seed-mix: unknown source {k!r}; "
                    f"expected one of {sorted(_MIX_RATIOS)}")
            try:
                override[k] = float(v)
            except ValueError:
                raise SystemExit(
                    f"--seed-mix: weight for {k!r} is not a number: {v!r}")
            if override[k] < 0:
                raise SystemExit(f"--seed-mix: negative weight for {k!r}")
        if set(override) != set(_MIX_RATIOS):
            missing = sorted(set(_MIX_RATIOS) - set(override))
            raise SystemExit(
                f"--seed-mix must name every source {sorted(_MIX_RATIOS)}; "
                f"missing {missing}")
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

    streams = _seed_streams(tok, args.seed_len, random.Random(args.seed))
    keys = list(seed_mix)
    weights = [seed_mix[k] for k in keys]
    rng = torch.Generator().manual_seed(args.seed)

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

    while accepted_tok < args.target_tokens:
        sources = [keys[torch.multinomial(
            torch.tensor(weights, dtype=torch.float), 1, generator=rng).item()]
            for _ in range(args.batch)]
        seeds = [next(streams[s]) for s in sources]
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
            cont = row[args.seed_len:]
            if eot is not None and eot in cont:
                cont = cont[:cont.index(eot)]
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

    flush()
    if tele is not None:
        tele.close()
    print(f"done: {accepted_tok/1e6:.2f}M accepted tokens in "
          f"{(time.time()-t0)/3600:.2f} h → {out}")


if __name__ == "__main__":
    main()
