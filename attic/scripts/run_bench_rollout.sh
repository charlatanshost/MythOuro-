#!/usr/bin/env bash
# ROLLOUT COST SWEEP — measure it, do not project it.
#
#   bash run_bench_rollout.sh          # ~15 min
#
# WHY A DRIVER AND NOT ONE PROCESS. rollout-len 128 at batch 32 aborted the whole
# run with a GPU page fault (NotPresent, PDE level 1, drm_neo.cpp) after 64 had
# completed cleanly — so a single-process sweep loses every config after the first
# crash. Each config runs in its own subprocess here; a crash costs that cell and
# nothing else, and the surviving cells still answer the question.
#
# WHY IT MATTERS THAT THIS IS MEASURED. The only real number so far is len 64 =
# 11.74 s (174 tok/s, 2.4 h of an 8.5 h leg). Everything said about 128 and 256 is
# an O(n^2) PROJECTION, and projections have been wrong three times this week —
# a "20 minute" eval that ran 2 h, a "4x" cost that was 16x, and a "budget
# artifact" that was not. Twice the surprise was in our favour. So measure.
#
# THE SWEEP ALSO ISOLATES THE CRASH. Batch varies as well as length, which
# separates "128 tokens is the problem" from "32 x 128 lanes is the problem".
# If batch 8 survives len 256, the cost question is answered AND the page fault
# is characterised as memory-driven rather than length-driven.
set -uo pipefail
trap 'pkill -INT -P $$ 2>/dev/null; true' INT TERM
cd "$(dirname "$0")"
source ../venv-xpu/bin/activate
export SYCL_CACHE_PERSISTENT=1 PYTORCH_ALLOC_CONF=expandable_segments:True TRITON_DEFAULT_BACKEND=intel
mkdir -p logs reports
if pgrep -f "training[.](distill|sft)" >/dev/null; then echo "a trainer is running"; exit 1; fi

CKPT=checkpoints_codemix/step_0149500.pt
OUT=reports/bench_rollout.tsv
LOG="logs/bench_rollout_$(date +%Y%m%d_%H%M).log"
: > "$OUT"

echo "len	batch	sec	tok_s" >> "$OUT"
for B in 8 16 32; do
  for L in 64 128 256; do
    printf "  len %-4s batch %-3s ... " "$L" "$B"
    if out=$(timeout 900 python -u -m tools.bench_rollout -c "$CKPT" --device xpu:0 \
               --one --lens "$L" --batch "$B" --reps 2 2>>"$LOG" | grep '^RESULT'); then
      echo "$out" | cut -f2- >> "$OUT"
      echo "$out" | awk -F'\t' '{printf "%7.2fs  %6.0f tok/s\n",$4,$5}'
    else
      echo "CRASHED or timed out"
      echo "$L	$B	CRASH	CRASH" >> "$OUT"
    fi
  done
done

echo
echo "=== measured rollout cost, and what it means per 6,000-step leg ==="
python3 - <<'EOFPY'
import csv
rows=[r for r in csv.DictReader(open("reports/bench_rollout.tsv"), delimiter="\t")]
print(f"  {'len':>5}{'batch':>7}{'sec':>9}{'tok/s':>9}{'leg hours':>11}   (750 regenerations)")
for r in rows:
    if r["sec"]=="CRASH":
        print(f"  {r['len']:>5}{r['batch']:>7}{'CRASH':>9}"); continue
    s=float(r["sec"])
    print(f"  {r['len']:>5}{r['batch']:>7}{s:>9.2f}{float(r['tok_s']):>9.0f}{750*s/3600:>11.1f}")
print("\n  The current recipe is len 64 / batch 32 = 2.4 h, 29% of an 8.5 h leg.")
print("  A config under ~4 h of rollout time is a workable night.")
EOFPY
