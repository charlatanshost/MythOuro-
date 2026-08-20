#!/usr/bin/env bash
# WAS THE 116k-120k "EVENT" REAL, OR JUST THE BOTTOM OF A U?
#
#   bash run_ce_ushape.sh          # ~10 min, no training, watch it
#
# THE SITUATION. best_exit CE across the pour read 0.348 / 0.280 / 0.286 / 0.494 /
# 0.561 (108k -> 125,181) and was called a step change at 116k-120k. But raw-framed
# code has since RECOVERED — 75.0 -> 60.0 -> 55.0 -> 65.0 at 140,000 — so the
# disruption was transient. The CE series simply STOPS at 125,181, which is the
# bottom of the dip. If CE has come back down at 140,000 there was no event at all,
# only a U-shape read while it was still descending. That would be the third
# U-shape misread this week and it would retire the last standing piece of the
# 2026-08-18 narrative.
#
# ⚠ WHY 116,000 IS RE-MEASURED. best_exit streams its eval text from remote HF, so
# its ABSOLUTE numbers only compare WITHIN one sitting (proved 2026-08-18: a repeat
# on the same checkpoint reproduced to +0.0012, while step 108,000 read 0.348 here
# against an archived 0.2036 from weeks earlier). Comparing tonight's 140,000
# against Tuesday's 116,000 would repeat exactly the mistake that produced the
# false "degraded 1.7-3.5x" alarm. Both points must be measured NOW, together.
#
# WHAT IT MEANS:
#   140,000 back near 116,000's level -> NO EVENT. It was a dip, everything
#       recovered, and the pour simply worked. Correct the tracker.
#   140,000 still elevated            -> the break is real and persists 20k steps
#       later, which makes it worth hunting properly with the ce log we now keep.
set -uo pipefail
trap 'pkill -INT -P $$ 2>/dev/null; true' INT TERM
cd "$(dirname "$0")"
source ../venv-xpu/bin/activate
export SYCL_CACHE_PERSISTENT=1 PYTORCH_ALLOC_CONF=expandable_segments:True TRITON_DEFAULT_BACKEND=intel

LOG="logs/ce_ushape_$(date +%Y%m%d_%H%M).log"
mkdir -p logs reports

if pgrep -f "training[.](distill|sft|train_depth_policy)" >/dev/null; then
  echo "a trainer is running — it needs the card; stop it first"; exit 1
fi
if pgrep -f "tools[.](code_eval|math_eval|best_exit_probe|onpolicy_rollout_probe|kd_exhaustion)" >/dev/null; then
  echo "another probe is running; let it finish"; exit 1
fi

{
for S in 116000 140000; do
  C="checkpoints_newmix/step_0$S.pt"
  [ -f "$C" ] || { echo "MISSING $C"; continue; }
  echo; echo "--- $S ---"
  python -u -m tools.best_exit_probe -c "$C" --device xpu:0 \
    --n-loops 8 --trained-loops 4 --by-domain \
    --json "reports/best_exit_${S}_tonight.json"
done
} 2>&1 | tee "$LOG"

echo
echo "=== SAME-SESSION COMPARISON (the only valid kind for this tool) ==="
python3 - <<'PY'
import json
def m(p):
    d=json.load(open(p))["by_domain"]
    return {e["domain"]: e["ce_trained_depth"] for e in d}
try: a,b = m("reports/best_exit_116000_tonight.json"), m("reports/best_exit_140000_tonight.json")
except FileNotFoundError as e: raise SystemExit(f"  missing: {e}")
print(f"  {'domain':14}{'116,000':>10}{'140,000':>10}{'diff':>9}")
for k in a: print(f"  {k:14}{a[k]:10.4f}{b[k]:10.4f}{b[k]-a[k]:+9.4f}")
ma,mb=sum(a.values())/len(a),sum(b.values())/len(b)
print(f"  {'MEAN':14}{ma:10.4f}{mb:10.4f}{mb-ma:+9.4f}")
print()
if mb-ma < 0.05: print("  RECOVERED -> there was no event. A U-shape read mid-dip. Correct the record.")
else: print("  STILL ELEVATED -> the break is real and persists. Worth hunting in the ce log.")
PY
