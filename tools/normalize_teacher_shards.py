#!/usr/bin/env python3
"""
Normalise teacher-corpus JSONL shards to ONE schema.

WHY. `MixedDataset._open_source` passes every shard to
`load_dataset("json", data_files=[...])`, which locks its schema to the FIRST
file (sorted order) and then refuses to cast the rest:

    Couldn't cast  text/source/seed_len -> {'text': ..., 'source': ...}
    because column names don't match

Measured 2026-08-31: `data_teacher_code` and `data_teacher_math` carry
{source, text}; `data_teacher_med` and `data_teacher_v2` also carry `seed_len`.
`code` sorts first, so med and v2 were dropped ENTIRELY — 410 skipped batches in
50 minutes, and a "broadened corpus" run that was silently code+math only.

The failure is a WARNING, not an error: the stream re-opens and keeps going, so
a run looks healthy while quietly training on the wrong mix. That is the worst
kind of bug this project has, and it is the second one this week whose signature
was "silently trains the wrong thing" (the other: growth_metadata dropped on
save).

The trainer reads only `text` (`extract_field_text`), so extra keys are dead
weight and dropping them is lossless for training.

    python -u -m tools.normalize_teacher_shards --dry-run     # inspect
    python -u -m tools.normalize_teacher_shards               # rewrite
"""
from __future__ import annotations
import argparse, glob, json, os, shutil, sys
from collections import Counter

KEEP = ("text", "source")
DEFAULT_DIRS = ("data_teacher_code", "data_teacher_math",
                "data_teacher_med", "data_teacher_v2")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dirs", nargs="*", default=list(DEFAULT_DIRS))
    ap.add_argument("--keep", nargs="*", default=list(KEEP))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    keep = tuple(a.keep)

    print(f"  canonical schema: {sorted(keep)}\n")
    plan = []
    for d in a.dirs:
        for f in sorted(glob.glob(os.path.join(d, "shard_*.jsonl"))):
            keys, rows, extra = Counter(), 0, Counter()
            with open(f) as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        o = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    rows += 1
                    keys[tuple(sorted(o))] += 1
                    for k in o:
                        if k not in keep:
                            extra[k] += 1
            needs = bool(extra) or any(set(k) != set(keep) for k in keys)
            plan.append((f, rows, dict(extra), needs))
    todo = [p for p in plan if p[3]]
    for f, rows, extra, needs in plan:
        mark = "REWRITE" if needs else "ok"
        print(f"  {mark:8s} {f:44s} {rows:>7,} rows"
              + (f"   drops: {sorted(extra)}" if extra else ""))
    print(f"\n  {len(todo)} of {len(plan)} shards need rewriting")
    if a.dry_run:
        print("  --dry-run: nothing written."); return
    if not todo:
        print("  nothing to do."); return

    for f, rows, extra, _ in todo:
        bak = f + ".preschema"
        if not os.path.exists(bak):
            shutil.copy2(f, bak)          # pristine-only backup
        tmp = f + ".tmp"
        n_out = 0
        with open(f) as src, open(tmp, "w") as dst:
            for line in src:
                line = line.strip()
                if not line:
                    continue
                try:
                    o = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not o.get("text"):
                    continue              # a row with no text trains nothing
                dst.write(json.dumps({k: o.get(k, "") for k in keep}) + "\n")
                n_out += 1
        os.replace(tmp, f)
        print(f"  rewrote {f}  {rows:,} -> {n_out:,} rows")
    print("\n  done. Backups at *.preschema (restore with:"
          " for b in */shard_*.preschema; do mv \"$b\" \"${b%.preschema}\"; done)")


if __name__ == "__main__":
    main()
