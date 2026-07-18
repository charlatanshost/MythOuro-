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
import sys
import time
from collections import Counter
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mythouro.training_utils import (          # noqa: E402
    _DATASET_SPECS,
    _MIX_RATIOS,
    load_distillation_teacher,
)

ROWS_PER_SHARD = 1000


def _seed_streams(tok, seed_len: int):
    """Per-corpus generators yielding fixed-length token seeds."""
    from datasets import load_dataset

    def stream(repo, config, split, field):
        while True:
            ds = load_dataset(repo, name=config, split=split, streaming=True)
            for sample in ds:
                text = sample.get(field, "")
                if not text:
                    continue
                ids = tok(text, truncation=True, max_length=seed_len + 8)["input_ids"]
                if len(ids) >= seed_len:
                    yield ids[:seed_len]

    return {
        key: stream(repo, config, split, field)
        for key, repo, config, split, field in _DATASET_SPECS
    }


def _passes(cont_ids: list[int], min_new: int, min_distinct1: float,
            max_top_share: float) -> bool:
    if len(cont_ids) < min_new:
        return False
    counts = Counter(cont_ids)
    if len(counts) / len(cont_ids) < min_distinct1:
        return False
    if counts.most_common(1)[0][1] / len(cont_ids) > max_top_share:
        return False
    return True


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
    p.add_argument("--min-distinct1", type=float, default=0.30)
    p.add_argument("--max-top-share", type=float, default=0.50)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

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

    out = Path(args.out_dir)
    out.mkdir(exist_ok=True)
    existing = sorted(out.glob("shard_*.jsonl"))
    shard_idx = int(existing[-1].stem.split("_")[1]) + 1 if existing else 0

    streams = _seed_streams(tok, args.seed_len)
    keys = list(_MIX_RATIOS)
    weights = [_MIX_RATIOS[k] for k in keys]
    rng = torch.Generator().manual_seed(args.seed)

    accepted_tok = accepted_n = rejected_n = 0
    rows: list[dict] = []
    t0 = time.time()
    manifest = {
        "teacher_id": args.teacher_id, "seed_len": args.seed_len,
        "max_new": args.max_new, "temperature": args.temperature,
        "top_p": args.top_p, "filters": {
            "min_new": args.min_new, "min_distinct1": args.min_distinct1,
            "max_top_share": args.max_top_share},
        "mix": _MIX_RATIOS, "started": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

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
        manifest.update(accepted=accepted_n, rejected=rejected_n,
                        accepted_tokens=accepted_tok,
                        updated=time.strftime("%Y-%m-%d %H:%M:%S"))
        (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))

    while accepted_tok < args.target_tokens:
        sources = [keys[torch.multinomial(
            torch.tensor(weights, dtype=torch.float), 1, generator=rng).item()]
            for _ in range(args.batch)]
        seeds = [next(streams[s]) for s in sources]
        input_ids = torch.tensor(seeds, device=args.device)
        with torch.no_grad():
            gen = teacher.generate(
                input_ids, max_new_tokens=args.max_new, do_sample=True,
                temperature=args.temperature, top_p=args.top_p,
                pad_token_id=tok.pad_token_id or eot or 0)
        for row, src in zip(gen.tolist(), sources):
            cont = row[args.seed_len:]
            if eot is not None and eot in cont:
                cont = cont[:cont.index(eot)]
            if not _passes(cont, args.min_new, args.min_distinct1,
                           args.max_top_share):
                rejected_n += 1
                continue
            accepted_n += 1
            accepted_tok += len(cont)
            rows.append({
                "text": tok.decode(row[:args.seed_len] + cont),
                "source": src, "seed_len": args.seed_len,
            })
        if len(rows) >= ROWS_PER_SHARD:
            flush()
        el = time.time() - t0
        print(f"accepted {accepted_n} ({accepted_tok/1e6:.2f}M tok) "
              f"rejected {rejected_n} | {accepted_tok/max(el,1):.0f} tok/s",
              flush=True)

    flush()
    print(f"done: {accepted_tok/1e6:.2f}M accepted tokens in "
          f"{(time.time()-t0)/3600:.2f} h → {out}")


if __name__ == "__main__":
    main()
