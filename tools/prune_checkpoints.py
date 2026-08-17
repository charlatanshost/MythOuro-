"""
Reclaim checkpoint space — DRY RUN BY DEFAULT, nothing is deleted without --yes.

WHY. 2026-08-17: 35 checkpoint directories, 168 files, 552 GB, against 91 GB
free. The pour to step 140,000 needs ~67 GB of that. Most of the 552 GB is
concluded experiments whose RESULT lives in docs/generation_probe_tracker.md —
the weights add nothing the write-up does not already record.

WHAT IT KEEPS, per directory:
  * the newest checkpoint, always (so every line stays resumable / re-evaluable)
  * `--keep N` newest instead, if you want more
  * anything matching `--keep-steps` (exact step numbers, comma separated)

WHAT IT REFUSES TO TOUCH. `_PROTECTED` holds the directories the project is
actually built on. These are skipped entirely unless named explicitly with
`--include`, and even then the newest is kept:
  * checkpoints_newmix  — THE LIVE LINE. step_0108471.pt is the base every
    branch descends from and every baseline was measured on.
  * checkpoints_reuse8  — its ancestor; holds step_0070500.pt, the reference for
    the 67.5% code L3+ measurement.

⚠ THIS DELETES FILES. The design is: print the full list, print the total, and
exit — unless `--yes` is passed. Read the list first. There is no undo, and
checkpoints are not in git (they are gitignored, correctly — they are 3.35 GB
each).

Usage
-----
    python -m tools.prune_checkpoints                    # dry run, everything
    python -m tools.prune_checkpoints --only checkpoints_lambda04 --yes
    python -m tools.prune_checkpoints --keep 2 --yes     # keep 2 newest per dir
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from pathlib import Path

_PROTECTED = {
    "checkpoints_newmix": "THE LIVE LINE — step_0108471.pt is the base for everything",
    "checkpoints_reuse8": "ancestor of the live line — holds the 70,500 reference",
}


def _step(p: str) -> int:
    m = re.search(r"step_0*(\d+)\.pt$", p)
    return int(m.group(1)) if m else -1


def plan(keep: int, keep_steps: set, only: list, include: list) -> tuple:
    dirs = sorted(d for d in glob.glob("checkpoints_*") if os.path.isdir(d))
    if only:
        dirs = [d for d in dirs if d in only]
    rows, doomed, freed = [], [], 0
    for d in dirs:
        prot = d in _PROTECTED and d not in include
        files = sorted(glob.glob(f"{d}/step_*.pt"), key=_step)
        if not files:
            continue
        size = sum(os.path.getsize(f) for f in files)
        if prot:
            rows.append((d, len(files), size, 0, 0, f"PROTECTED — {_PROTECTED[d]}"))
            continue
        keepset = set(files[-keep:])
        keepset |= {f for f in files if _step(f) in keep_steps}
        drop = [f for f in files if f not in keepset]
        dsz = sum(os.path.getsize(f) for f in drop)
        rows.append((d, len(files), size, len(drop), dsz, ""))
        doomed += drop
        freed += dsz
    return rows, doomed, freed


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--keep", type=int, default=1,
                   help="Newest checkpoints to keep per directory (default 1).")
    p.add_argument("--keep-steps", default="",
                   help="Comma-separated step numbers to keep everywhere, e.g. "
                        "70500,108471.")
    p.add_argument("--only", nargs="+", default=[],
                   help="Restrict to these directories.")
    p.add_argument("--include", nargs="+", default=[],
                   help="Prune a PROTECTED directory anyway (newest still kept).")
    p.add_argument("--yes", action="store_true",
                   help="Actually delete. Without this it is a dry run.")
    args = p.parse_args()

    keep_steps = {int(s) for s in args.keep_steps.split(",") if s.strip()}
    rows, doomed, freed = plan(args.keep, keep_steps, args.only, args.include)

    print(f"\n  {'directory':<32}{'files':>6}{'size':>9}{'delete':>8}{'frees':>9}  note")
    for d, n, size, dn, dsz, note in rows:
        print(f"  {d:<32}{n:>6}{size/1e9:>8.0f}G{dn:>8}{dsz/1e9:>8.0f}G  {note}")
    print(f"\n  would delete {len(doomed)} files, freeing {freed/1e9:.0f} GB")
    print(f"  keeping the {args.keep} newest per directory"
          + (f" plus steps {sorted(keep_steps)}" if keep_steps else ""))

    if not doomed:
        print("  nothing to do."); return
    if not args.yes:
        print("\n  DRY RUN — nothing deleted. Read the list above, then re-run")
        print("  with --yes. Checkpoints are not in git and there is no undo.")
        return

    print()
    gone = 0
    for f in doomed:
        try:
            sz = os.path.getsize(f); os.remove(f); gone += sz
        except OSError as e:
            print(f"  FAILED {f}: {e}")
    print(f"  deleted {len(doomed)} files, freed {gone/1e9:.0f} GB")
    # prove the protected line survived
    for d in _PROTECTED:
        if os.path.isdir(d):
            n = len(glob.glob(f"{d}/step_*.pt"))
            print(f"  {d}: {n} checkpoints still present")


if __name__ == "__main__":
    main()
