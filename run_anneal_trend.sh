#!/usr/bin/env bash
# IS 82.5% A TREND OR A SPIKE? — a test, ~8 min, no training.
#
#   bash run_anneal_trend.sh
#
# The α=0.45 leg ended at RAW code L3+ 82.5%, against 65.0% at its 140,000 start
# and a previous project record of 75.0% (step 108,471). That is a big claim off
# ONE endpoint at n=80 (±10pp CI), and this project has misread single points
# repeatedly — the 2026-08-17 U-shape (75.0/41.2/53.8/60.0/66.2/68.8) and the
# 2026-08-18 "phase transition" both came from reading a curve at one end.
#
# Two intermediate milestones settle it. RAW only: chat was flat (33.8 -> 31.2,
# inside noise) so it is not the informative axis here.
#
#   rising 65 -> ~72 -> ~78 -> 82.5  -> real, the anneal is compounding. Go to 0.40.
#   flat then a jump at the end      -> 82.5 is a spike. Re-measure at a second
#                                       seed before believing it.
set -uo pipefail
trap 'pkill -INT -P $$ 2>/dev/null; true' INT TERM
cd "$(dirname "$0")"
source ../venv-xpu/bin/activate
export SYCL_CACHE_PERSISTENT=1 PYTORCH_ALLOC_CONF=expandable_segments:True TRITON_DEFAULT_BACKEND=intel
mkdir -p logs reports
if pgrep -f "training[.](distill|sft)" >/dev/null; then echo "a trainer is running"; exit 1; fi

LOG="logs/anneal_trend_$(date +%Y%m%d_%H%M).log"
{
for S in 141500 142500; do
  C="checkpoints_anneal045/step_0$S.pt"
  [ -f "$C" ] || { echo "MISSING $C"; continue; }
  echo; echo "--- $S RAW ---"
  python -u -m tools.code_eval -c "$C" --device xpu:0 \
    --samples 8 --temperature 0.4 --seed 1234 --repetition-penalty 1.15 \
    --json "reports/code_anneal045_${S}_pen115.json"
done
} 2>&1 | tee "$LOG"

echo
echo "=== RAW code L3+ across the anneal leg ==="
python3 - <<'PY'
import json,os
pts=[(140000,65.0)]
for s in (141500,142500,143500):
    p=f"reports/code_anneal045_{s}_pen115.json"
    if os.path.exists(p):
        pts.append((s,round(100*json.load(open(p))["diagnostics"]["per_sample_l3plus"],1)))
for s,v in pts: print(f"  {s:>8}  {v:5.1f}%")
print("\n  previous project record: 75.0% @108,471 (α=0.5)")
if len(pts)>2:
    rising=all(pts[i][1]<=pts[i+1][1]+3 for i in range(len(pts)-1))
    print("  -> monotone-ish RISE: the anneal is compounding. Anneal to 0.40." if rising
          else "  -> NOT monotone: 82.5 may be a spike. Confirm at a second seed first.")
PY
