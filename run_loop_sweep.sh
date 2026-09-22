#!/usr/bin/env bash
# LOOP SWEEP — does inference depth move CONTENT accuracy, or only syntax?
#
#   bash run_loop_sweep.sh [checkpoint]      # ~45 min, 3 loop counts, no training
#
# THE HYPOTHESIS (2026-09-22). arXiv 2607.14427 measures, on a 135M model, that
# convergence depth is ordered by token type — whitespace shallowest, CONTENT
# WORDS DEEPEST — with the median token converging at loop 6 and ~10% still
# updating at 8. We train and infer at K=4. If K=4 truncates content tokens
# mid-convergence while syntax has long since settled, that is exactly our
# failure shape: fluent, grammatical, wrong.
#
# ⚠ THE PAPER DOES NOT CLAIM MORE LOOPS = MORE CAPABILITY. Its own finding is
# that validation loss is FLAT past 8 loops; the result is an efficiency one
# (early-exit at 4.94 average loops for the same quality). So this sweep tests a
# mechanism the paper supplies, not a claim it makes.
#
# ⚠ A SWEEP LIKE THIS ALREADY RAN AND FOUND FLAT — 2026-07-31, loops 4/6/8,
# "the depth sweep did NOT lift the wall". Two things differ now and neither is
# a reason to expect a different answer, only a reason the old one does not
# settle it: (a) that base was far weaker (math L3+ 5.0%, L4 0.0%); (b) the
# halt gates were shaped only by the depth regulariser, whereas exit_pdf has
# since trained them against the task loss (verified 2026-09-22: `halt` carries
# gradient to recurrent.act.halt.weight/bias). Extrapolating past TRAINED depth
# is the thing the 07-31 entry correctly says nothing is known about.
#
# WHAT WOULD COUNT
#   L4 rises with K, L3+ flat      -> content improves, syntax already settled.
#                                     The hypothesis survives and rung 5
#                                     (grow_depth, TRAIN at K>4) is the move.
#   everything flat                -> dies here, for ~45 minutes. The 07-31
#                                     result reproduces on a stronger base and
#                                     loop count is closed as an inference lever.
#   everything falls               -> K>4 is out-of-distribution for a model
#                                     trained at 4, which is the expected null
#                                     and what 07-31 saw.
#
# ⚠ ONE CHECKPOINT, ONE DRAW. L3+ has 13.6pp checkpoint sd and L4 relative sd
# 0.77. This sweep varies K on the SAME weights with the SAME seed, so the
# comparison is paired and far tighter than cross-checkpoint — but a 2pp L4
# move is still noise. Look for a monotone trend across three values, not a
# single gap.
set -uo pipefail
cd "$(dirname "$0")"
source ../venv-xpu/bin/activate
export TRITON_DEFAULT_BACKEND=intel
unset SYCL_CACHE_PERSISTENT PYTORCH_ALLOC_CONF
export PYTHONFAULTHANDLER=1 PYTHONUNBUFFERED=1
export SYCL_QUEUE_THREAD_POOL_SIZE=1
export ZE_SERIALIZE=2
CK="${1:-checkpoints_anneal_lr/step_0003000.pt}"
[ -f "$CK" ] || { echo "missing $CK"; exit 1; }
if pgrep -f "[p]ython -u -m (training|tools)\." >/dev/null; then echo "card busy"; exit 1; fi
TAG=$(basename "$(dirname "$CK")" | sed 's/^checkpoints_//')_$(basename "$CK" .pt | sed 's/step_0*//')
mkdir -p reports logs

for K in 4 6 8; do
  OUT="reports/code_loops${K}_${TAG}.json"
  [ -f "$OUT" ] && { echo "=== K=$K already measured ==="; continue; }
  echo; echo "=== K=$K loops ==="
  python -u -m tools.code_eval -c "$CK" --device xpu:0 --n-loops "$K" \
    --samples 32 --temperature 0.4 --seed 1234 --repetition-penalty 1.15 \
    --max-new 96 --json "$OUT" 2>&1 | tee -a "logs/loops${K}_${TAG}.log" | grep -E "per-sample L3|COMMITTED" || true
done

echo
echo "=================================================================="
echo "  INFERENCE DEPTH vs ACCURACY — same weights, same seed, K varied"
echo "=================================================================="
python3 - "$TAG" <<'PY'
import json, os, sys
tag=sys.argv[1]
print(f"  {'K':>3} {'L0':>7} {'L3+':>7} {'L4':>7} {'commit':>8}")
for K in (4,6,8):
    p=f"reports/code_loops{K}_{tag}.json"
    if not os.path.exists(p): print(f"  {K:>3}   (missing)"); continue
    d=json.load(open(p)); S=[s for t in d["tasks"] for s in t["samples"]]; n=len(S)
    r=lambda f: sum(1 for s in S if f(int(s.get("rung",0))))/n
    com=sum(1 for s in S if str(s.get("committed"))=="True")/n
    print(f"  {K:>3} {r(lambda x:x==0):6.1%} {r(lambda x:x>=3):6.1%} {r(lambda x:x>=4):6.1%} {com:7.1%}")
print()
print("  L4 rising with K and L3+ flat = content improves where syntax had settled.")
print("  All flat = the 2026-07-31 result reproduces on a stronger base; loop")
print("  count is not an inference lever and rung 5 stays shelved.")
print("  ⚠ one checkpoint. Read the TREND across three K, not any single gap.")
PY
