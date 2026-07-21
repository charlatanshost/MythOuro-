"""
Strip boilerplate rows from a harvested teacher corpus (docs/teacher_corpus_plan.md).

The v1 harvest seeded from each document's HEAD, which is systematically
boilerplate — source files open with license headers, scraped pages with nav
cruft. Measured on the first 5.84M-token corpus (2026-07-21): **57% of CODE
samples were the teacher faithfully continuing an Apache/copyright header**
(math 0.7%, general 0.5%). `gen_teacher_corpus` now seeds from a random window
and rejects boilerplate at write time; this tool retro-fits an existing corpus
so it doesn't have to be thrown away.

Non-destructive: reads `--in-dir`, writes cleaned shards to `--out-dir`, never
mutates the input (which a training run may be streaming from live).

    python -m tools.clean_teacher_corpus --in-dir data_teacher \
        --out-dir data_teacher_clean
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.gen_teacher_corpus import _BOILERPLATE     # noqa: E402

ROWS_PER_SHARD = 1000


def is_boilerplate(text: str, window: int, min_hits: int) -> bool:
    """Same rule as the generator's write-time filter."""
    return len(_BOILERPLATE.findall(text[:window])) >= min_hits


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--in-dir", default="data_teacher")
    p.add_argument("--out-dir", default="data_teacher_clean")
    p.add_argument("--window", type=int, default=800,
                   help="Chars from the start of each row to scan.")
    p.add_argument("--min-hits", type=int, default=2,
                   help="License-ish matches within the window to reject.")
    p.add_argument("--dry-run", action="store_true",
                   help="Report what would be dropped; write nothing.")
    args = p.parse_args()

    src = Path(args.in_dir)
    shards = sorted(src.glob("shard_*.jsonl"))
    if not shards:
        raise SystemExit(f"no shards in {src}")

    kept, dropped = [], Counter()
    total = Counter()
    kept_tok = dropped_tok = 0
    for f in shards:
        for line in f.open():
            r = json.loads(line)
            total[r["source"]] += 1
            # Rough token estimate is fine for reporting (chars/4).
            approx = max(1, len(r["text"]) // 4)
            if is_boilerplate(r["text"], args.window, args.min_hits):
                dropped[r["source"]] += 1
                dropped_tok += approx
            else:
                kept.append(r)
                kept_tok += approx

    print(f"{'source':10} {'total':>8} {'dropped':>9} {'kept':>8}")
    for s in sorted(total):
        print(f"{s:10} {total[s]:8} {dropped[s]:9} {total[s]-dropped[s]:8} "
              f"({100*dropped[s]/total[s]:.1f}% dropped)")
    print(f"\n{'TOTAL':10} {sum(total.values()):8} {sum(dropped.values()):9} "
          f"{len(kept):8}")
    print(f"~tokens: kept {kept_tok/1e6:.2f}M, dropped {dropped_tok/1e6:.2f}M")

    if args.dry_run:
        print("\n--dry-run: nothing written")
        return

    out = Path(args.out_dir)
    out.mkdir(exist_ok=True)
    for i in range(0, len(kept), ROWS_PER_SHARD):
        chunk = kept[i:i + ROWS_PER_SHARD]
        with (out / f"shard_{i // ROWS_PER_SHARD:04d}.jsonl").open("w") as fh:
            for r in chunk:
                fh.write(json.dumps(r) + "\n")

    manifest = {}
    src_manifest = src / "MANIFEST.json"
    if src_manifest.exists():
        manifest = json.loads(src_manifest.read_text())
    manifest["cleaned_from"] = str(src)
    manifest["cleaned_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    manifest["clean_rule"] = {
        "filter": "boilerplate (license/copyright)",
        "window_chars": args.window, "min_hits": args.min_hits,
    }
    manifest["clean_dropped"] = dict(dropped)
    manifest["clean_kept_rows"] = len(kept)
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nwrote {len(kept)} rows to {out}/ (+ MANIFEST.json)")


if __name__ == "__main__":
    main()
