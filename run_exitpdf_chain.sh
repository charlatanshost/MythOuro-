#!/usr/bin/env bash
# EXIT_PDF CHAIN — verify the pilot did not cost capability, then run the leg.
#
#   bash run_exitpdf_chain.sh          # ~8h total, unattended
#
# THE PILOT ALREADY PASSED HALF THE GATE (2026-09-02, 1,200 steps):
#   halt BEFORE  [0.249, 0.252, 0.256, 0.244]  mean depth 2.49/4  KL 0.0001
#   halt AFTER   [0.012, 0.167, 0.350, 0.471]  mean depth 3.28/4  KL 0.3132
# The halt distribution moved OFF uniform and toward DEEPER loops — the
# pre-registered PASS branch, and the opposite of the loop-0 collapse that would
# have vindicated Silent Thinking's final-only position.
#
# The other half is "L4/L3+ hold or rise". Loss falling 0.93 -> 0.74 is not
# capability. This script measures it, and only continues if nothing collapsed.
#
# ⚠ THE ABORT THRESHOLD IS CATASTROPHIC-ONLY, ON PURPOSE. L3+ has 13.6 pp
# checkpoint-to-checkpoint sd (2026-08-31), so a few points of movement is noise
# and must NOT stop the run. What this catches is a rung-3-style collapse — that
# one went 0.980 -> 0.075. Abort at L3+ < 50% (1.5 sd below the pruned24 mean of
# 71.1%) or L0 > 45% (against pruned24's ~12%).
set -uo pipefail
cd "$(dirname "$0")"
source ../venv-xpu/bin/activate

CK=checkpoints_exitpdf/step_0001200.pt
[ -f "$CK" ] || { echo "missing $CK — run STEPS=1200 bash run_exitpdf.sh first"; exit 1; }
if pgrep -f "python -u -m training\.(distill|sft)" >/dev/null; then
  echo "a trainer is running"; exit 1; fi

echo "=== STAGE 1: does the pilot hold capability? ==="
bash run_eval.sh "$CK" exitpdf_1200

python - <<'PY' || exit 1
import json, sys, os
p="reports/code_exitpdf_1200.json"
if not os.path.exists(p):
    print("  eval produced no json — refusing to continue"); sys.exit(1)
d=json.load(open(p)); S=[s for t in d["tasks"] for s in t["samples"]]
n=len(S); l3=d["diagnostics"]["per_sample_l3plus"]*100
l0=sum(1 for s in S if s.get("rung",0)==0)*100/n
l4=sum(1 for s in S if s.get("rung",0)>=4)
print(f"\n  exit_pdf @1,200: L3+ {l3:.1f}%   L0 {l0:.0f}%   L4 {l4}/{n}")
print(f"  pruned24 baseline: L3+ 71.1%   L0 ~12%")
if l3 < 50.0 or l0 > 45.0:
    print("\n  ⛔ COLLAPSE — aborting the chain. This is the rung-3 signature.")
    print("     Revert: --depth-reg-coeff back to 0.3, or --loop-loss-weighting off.")
    sys.exit(1)
print("\n  ✅ capability held. Both halves of the gate pass — continuing to the full leg.")
PY

echo
echo "=== STAGE 2: the full leg, 3,000 more steps (~7.4h at the measured 8.9 s/step) ==="
bash run_exitpdf.sh

echo
echo "=== STAGE 3: read BOTH instruments over >=3 checkpoints ==="
LAST=$(ls -t checkpoints_exitpdf/step_0*.pt | head -1)
echo "  bash run_eval.sh $LAST exitpdf_final"
echo "  bash run_prose_readout.sh $(ls -t checkpoints_exitpdf/step_0*.pt | head -3 | tr '\n' ' ')"
echo "  baselines, all measured in one session:"
echo "    pruned24  prose 0.104 / 0.571   code L3+ 71.1%"
echo "    mathcode (regressed)  prose 0.159 / 0.482"
