#!/usr/bin/env bash
# Code eval at n=320, with the venv activated. USE THIS INSTEAD OF A BARE python LINE.
#
#   bash run_eval.sh checkpoints_balancer_off/step_0006000.pt [tag]
#
# WHY THIS SCRIPT EXISTS. `python` does not exist outside the venv on this box —
# only `python3` — so every hand-written `python -u -m tools.code_eval ...` line
# dies with "Command 'python' not found". That has now happened FOUR times,
# including once where it silently skipped a corpus-normalisation step and cost
# a restart. The run_*.sh convention exists precisely so the activation cannot
# be forgotten.
#
# Settings are pinned to match every archived n=320 baseline exactly:
#   T=0.4  pen=1.15  seed=1234  max_new=96  framing=bare  chat=False
# Do not change them ad hoc — the comparisons stop working.
#
# ⚠ tools.code_eval takes ONE checkpoint (`-c` is required=True, no nargs).
# Pass a single path. A glob will expand and die on unrecognised arguments.
set -uo pipefail
cd "$(dirname "$0")"
source ../venv-xpu/bin/activate

CKPT="${1:?usage: bash run_eval.sh <checkpoint.pt> [tag]}"
[ -f "$CKPT" ] || { echo "no such checkpoint: $CKPT"; exit 1; }
TAG="${2:-$(basename "$(dirname "$CKPT")")_$(basename "$CKPT" .pt | sed 's/step_0*//')}"
OUT="reports/code_${TAG}.json"
mkdir -p reports logs

if pgrep -f "python -u -m training\.(distill|sft)" >/dev/null; then
  echo "a trainer is running — the eval will contend for the card. Stop it first."; exit 1; fi

echo "=== eval: $CKPT  ->  $OUT ==="
python -u -m tools.code_eval -c "$CKPT" --device xpu:0 \
  --samples 32 --temperature 0.4 --seed 1234 --repetition-penalty 1.15 \
  --json "$OUT" 2>&1 | tee -a "logs/eval_${TAG}.log"

python - "$OUT" <<'PY'
import json,sys,math,os
def read(p):
    d=json.load(open(p)); n=sum(len(t["samples"]) for t in d["tasks"])
    l4=sum(1 for t in d["tasks"] for s in t["samples"] if s.get("rung",0)>=4)
    return n,d["diagnostics"]["per_sample_l3plus"]*100,l4
n,l3,l4=read(sys.argv[1]); ci=1.96*math.sqrt((l3/100)*(1-l3/100)/n)*100
print(f"\n  {os.path.basename(sys.argv[1]):34s} L3+ {l3:5.1f}% ±{ci:.1f}   L4 {l4}/{n}")
ctl="reports/code_g2_control.json"
if os.path.exists(ctl):
    cn,cl3,cl4=read(ctl)
    print(f"  {'code_g2_control.json (278M base)':34s} L3+ {cl3:5.1f}% ±{1.96*math.sqrt((cl3/100)*(1-cl3/100)/cn)*100:.1f}   L4 {cl4}/{cn}")
    print(f"\n  delta  L3+ {l3-cl3:+.1f} pp   L4 {l4-cl4:+d}")
print("\n  ⚠ ONE checkpoint is ONE DRAW. L3+ has ~13.6 pp checkpoint-to-checkpoint")
print("    sd (2026-08-31) — 5x the sampling CI. Do not call a delta this size a")
print("    result. Means over >=3 checkpoints per condition, or nothing.")
PY
