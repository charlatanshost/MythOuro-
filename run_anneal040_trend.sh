#!/usr/bin/env bash
# IS 58.8% A DECLINE, A DIP, OR NOISE? — test, ~12 min, no training.
#
#   bash run_anneal040_trend.sh
#
# The α=0.40 leg ended at RAW 58.8% against 82.5% where it started, while CHAT
# went 31.2% → 50.0%, an all-time high. Both frames are single endpoints from a
# 6,000-step leg. The 0.45 leg was measured at three points and was cleanly
# monotone; this one has only its end, and every misread this week came from
# exactly that.
#
# Three shapes, three different decisions:
#   MONOTONE FALL (82.5 -> ~75 -> ~66 -> 58.8)  -> α=0.40 genuinely trades raw for
#       chat. Real result: pick the α that matches the goal, do not "fix" it.
#   ROSE THEN FELL (peak mid-leg above 82.5)    -> 0.40 helps briefly then
#       over-corrects. The right stop is a mid-leg checkpoint, not 149,500.
#   DIP AND RECOVERING                          -> same U-shape as 116k-120k; more
#       steps at 0.40 may finish the recovery.
#
# BOTH FRAMES at each point, because reading either alone inverts the verdict —
# that is the standing 2026-08-18 warning and this leg is its clearest example yet.
set -uo pipefail
trap 'pkill -INT -P $$ 2>/dev/null; true' INT TERM
cd "$(dirname "$0")"
source ../venv-xpu/bin/activate
export SYCL_CACHE_PERSISTENT=1 PYTORCH_ALLOC_CONF=expandable_segments:True TRITON_DEFAULT_BACKEND=intel
mkdir -p logs reports
if pgrep -f "training[.](distill|sft)" >/dev/null; then echo "a trainer is running"; exit 1; fi

LOG="logs/anneal040_trend_$(date +%Y%m%d_%H%M).log"
{
for S in 145500 147500; do
  C="checkpoints_anneal040/step_0$S.pt"
  [ -f "$C" ] || { echo "MISSING $C"; continue; }
  echo; echo "--- $S RAW ---"
  python -u -m tools.code_eval -c "$C" --device xpu:0 --samples 8 --temperature 0.4 \
    --seed 1234 --repetition-penalty 1.15 --json "reports/code_anneal040_${S}_pen115.json"
  echo; echo "--- $S CHAT ---"
  python -u -m tools.code_eval -c "$C" --device xpu:0 --samples 8 --temperature 0.4 \
    --seed 1234 --repetition-penalty 1.15 --chat-template --extract \
    --json "reports/code_anneal040_${S}_CHAT.json"
done
} 2>&1 | tee "$LOG"

echo
echo "=== BOTH FRAMES ACROSS BOTH ANNEAL LEGS ==="
python3 - <<'PY'
import json,os
def v(p):
    return round(100*json.load(open(p))["diagnostics"]["per_sample_l3plus"],1) if os.path.exists(p) else None
rows=[(140000,"0.50",65.0,33.8),(143500,"0.45",82.5,31.2)]
for s in (145500,147500,149500):
    rows.append((s,"0.40",v(f"reports/code_anneal040_{s}_pen115.json"),v(f"reports/code_anneal040_{s}_CHAT.json")))
print(f"  {'step':>8}{'α':>7}{'RAW':>8}{'CHAT':>8}")
for s,a,r,c in rows:
    print(f"  {s:>8}{a:>7}{(f'{r:.1f}' if r else '—'):>8}{(f'{c:.1f}' if c else '—'):>8}")
PY
