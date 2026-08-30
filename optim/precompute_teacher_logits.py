#!/usr/bin/env python3
"""
OPT-B step 1 — precompute the teacher's top-K logits for a FIXED offline corpus.

WHY. `_StepProfiler` measures the teacher forward at 78.4% of every training
step. OPT-A removes the on-policy share (cached across rollout reuse); this
removes the offline share by answering the teacher's questions once, offline,
and reading them back from disk during training.

WHAT IT WRITES, per shard:
    tokens.npy     int32   (N, L+1)   the exact windows training will consume
    topk_idx.npy   uint16  (N, L, K)  teacher's top-K vocab ids  (vocab < 65536)
    topk_val.npy   float16 (N, L, K)  RAW teacher logits at T=1 (rescalable)
    tail_lse.npy   float32 (N, L)     logsumexp(remaining_logits / temperature)
plus manifest.json recording teacher id, seq_len, K, temperature and vocab.

⚠ THE CACHE *IS* THE DATASET. Tokens are stored alongside the logits rather than
recomputed at train time. This is deliberate: any independent re-tokenisation or
re-shuffle would silently misalign the cache with the batch and distil against
the wrong tokens. Alignment by construction is the only safe design here.

⚠ TEMPERATURE IS BAKED INTO `tail_lse`. logsumexp(z/T) over the tail cannot be
recovered from logsumexp(z), so a cache built at T=2.0 is only valid at T=2.0.
The manifest records it and the training loader refuses a mismatch. Top-K values
are stored raw, so only the tail term is temperature-locked.

⚠ THIS IS A GPU JOB and it is not free. It runs the teacher over the whole
corpus once. Budget it against what it saves: the teacher does ~1,820 tok/s
inside training, and considerably more here (forward only, no student, no
backward, large batch). See the report it prints at the end.

⚠ IT ALSO FIXES THE OFFLINE CORPUS. Streaming fineweb-edu is effectively
infinite; a cache is finite. Size it for the steps you plan, or the offline path
starts re-reading — which is the token-dilution half of the v5 post-mortem.
`--report-only` prints the sizing and captured-mass numbers WITHOUT writing.

USAGE
    python -u -m tools.precompute_teacher_logits \
      --files 'data_teacher_code/shard_*.jsonl,data_teacher_math/shard_*.jsonl' \
      --teacher-id ByteDance/Ouro-2.6B-Thinking --device xpu:0 \
      --seq-len 1024 --top-k 32 --temperature 2.0 \
      --out data_teacher_logits_k32 --max-tokens 30000000
"""
from __future__ import annotations

import argparse, glob, json, os, time
import numpy as np
import torch
from loguru import logger


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--files", required=True,
                   help="comma-separated globs of teacher-corpus JSONL")
    p.add_argument("--teacher-id", required=True)
    p.add_argument("--tokenizer-id", default=None,
                   help="defaults to --teacher-id")
    p.add_argument("--device", default="xpu:0")
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--top-k", type=int, default=32)
    p.add_argument("--temperature", type=float, default=2.0,
                   help="MUST match the training --temperature (default 2.0). "
                        "Baked into tail_lse; the loader asserts it.")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--out", required=True)
    p.add_argument("--rows-per-shard", type=int, default=2000)
    p.add_argument("--max-tokens", type=int, default=0,
                   help="stop after roughly this many cached tokens (0 = all)")
    p.add_argument("--report-only", action="store_true",
                   help="measure captured mass + sizing on a sample, write nothing")
    p.add_argument("--sample-rows", type=int, default=64,
                   help="rows used for the --report-only measurement")
    p.add_argument("--trust-remote-code", action="store_true")
    return p.parse_args()


def iter_windows(files, tok, seq_len, limit_tokens):
    """Yield (L+1)-token windows, packed across documents."""
    import json as _json
    buf, produced = [], 0
    for path in files:
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    text = _json.loads(line).get("text", "")
                except Exception:
                    continue
                if not text:
                    continue
                buf.extend(tok(text, add_special_tokens=False)["input_ids"])
                while len(buf) >= seq_len + 1:
                    yield buf[: seq_len + 1]
                    buf = buf[seq_len:]
                    produced += seq_len
                    if limit_tokens and produced >= limit_tokens:
                        return


def main():
    a = parse_args()
    from transformers import AutoTokenizer, AutoModelForCausalLM

    files = sorted(f for p in a.files.split(",") for f in glob.glob(p.strip()))
    if not files:
        raise SystemExit(f"no files matched {a.files!r}")
    logger.info(f"precompute: {len(files)} corpus files")

    tok = AutoTokenizer.from_pretrained(a.tokenizer_id or a.teacher_id,
                                        trust_remote_code=a.trust_remote_code)
    logger.info(f"precompute: loading teacher {a.teacher_id}")
    teacher = AutoModelForCausalLM.from_pretrained(
        a.teacher_id, dtype=torch.bfloat16,
        trust_remote_code=a.trust_remote_code,
    ).to(a.device).eval()
    for q in teacher.parameters():
        q.requires_grad_(False)

    os.makedirs(a.out, exist_ok=True)
    K, L, T = a.top_k, a.seq_len, a.temperature

    shard, n_shard = {"tokens": [], "topk_idx": [], "topk_val": [], "tail_lse": []}, 0
    batch, n_rows, n_tokens = [], 0, 0
    mass_sum, mass_n = 0.0, 0
    t0 = time.perf_counter()
    limit = a.sample_rows * (L + 1) if a.report_only else a.max_tokens

    def flush_batch():
        nonlocal batch, n_rows, n_tokens, mass_sum, mass_n
        if not batch:
            return
        ids = torch.tensor(batch, dtype=torch.long, device=a.device)   # (b, L+1)
        with torch.no_grad():
            out = teacher(ids[:, :-1], use_cache=False, past_key_values=None)
            lg = (out.logits if hasattr(out, "logits") else out).float()  # (b,L,V)
            val, idx = lg.topk(K, dim=-1)
            # captured probability mass at T=1 — the number that justifies K
            full_lse = torch.logsumexp(lg, dim=-1)
            mass = (torch.logsumexp(val, dim=-1) - full_lse).exp()
            mass_sum += float(mass.sum()); mass_n += mass.numel()
            # tail logsumexp AT THE TRAINING TEMPERATURE
            scaled = lg / T
            masked = scaled.scatter(-1, idx, torch.full_like(val, -1e30))
            tail = torch.logsumexp(masked, dim=-1)                      # (b,L)
        if not a.report_only:
            shard["tokens"].append(ids.cpu().numpy().astype(np.int32))
            shard["topk_idx"].append(idx.cpu().numpy().astype(np.uint16))
            shard["topk_val"].append(val.cpu().numpy().astype(np.float16))
            shard["tail_lse"].append(tail.cpu().numpy().astype(np.float32))
        n_rows += len(batch); n_tokens += len(batch) * L
        batch = []

    def flush_shard():
        nonlocal shard, n_shard
        if not shard["tokens"]:
            return
        d = os.path.join(a.out, f"shard_{n_shard:05d}")
        os.makedirs(d, exist_ok=True)
        for k, v in shard.items():
            np.save(os.path.join(d, f"{k}.npy"), np.concatenate(v, axis=0))
        n_shard += 1
        shard = {k: [] for k in shard}

    vocab = None
    for win in iter_windows(files, tok, L, limit):
        batch.append(win)
        if len(batch) >= a.batch:
            flush_batch()
            if vocab is None:
                vocab = int(teacher.config.vocab_size)
            if not a.report_only and sum(x.shape[0] for x in shard["tokens"]) >= a.rows_per_shard:
                flush_shard()
        if a.report_only and n_rows >= a.sample_rows:
            break
        if a.max_tokens and n_tokens >= a.max_tokens and not a.report_only:
            break
    flush_batch()
    if not a.report_only:
        flush_shard()

    dt = time.perf_counter() - t0
    bytes_per_tok = 4 * K + 4          # uint16 idx + fp16 val + fp32 tail
    logger.info("=" * 68)
    logger.info(f"  rows {n_rows:,}  tokens {n_tokens:,}  in {dt/60:.1f} min "
                f"({n_tokens/max(dt,1e-9):,.0f} tok/s)")
    if mass_n:
        logger.info(f"  MEAN TOP-{K} PROBABILITY MASS CAPTURED: "
                    f"{mass_sum/mass_n*100:.3f}%")
        logger.info("    ^ this is the number that justifies K. Below ~99% the "
                    "lumped tail is carrying real signal — raise K.")
    logger.info(f"  storage {bytes_per_tok} B/token → "
                f"{n_tokens*bytes_per_tok/1e9:.2f} GB for what was just measured")
    if a.report_only:
        logger.warning("  --report-only: NOTHING WAS WRITTEN.")
    else:
        json.dump({
            "teacher_id": a.teacher_id, "seq_len": L, "top_k": K,
            "temperature": T, "vocab_size": vocab, "rows": n_rows,
            "tokens": n_tokens, "shards": n_shard,
            "mean_topk_mass": (mass_sum / mass_n) if mass_n else None,
        }, open(os.path.join(a.out, "manifest.json"), "w"), indent=2)
        logger.success(f"  wrote {n_shard} shards → {a.out}")
    logger.info("=" * 68)


if __name__ == "__main__":
    main()
