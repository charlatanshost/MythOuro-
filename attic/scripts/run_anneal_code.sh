#!/usr/bin/env bash
# DID THE ANNEAL COST CODE CAPABILITY? — a test, not a training run. ~8 min.
#
#   bash run_anneal_code.sh
#
# WHY THIS IS THE DECIDING NUMBER. The α 0.5→0.45 anneal split by domain at
# α=0.0: prose improved a lot (top_share 0.139 -> 0.092 over four seeds, and the
# text agrees — the letter-salad mode is gone), while code and math got WORSE
# (fibonacci 0.246 -> 0.370, quadratic 0.125 -> 0.180). The six-seed MEAN is
# 0.154 -> 0.153 — flat — because the two effects cancel. The mean is useless here.
#
# But the rollout probe samples at T=1.0 / top_k 50 / NO repetition penalty, which
# is far harsher than deployment. code_eval at pen 1.15 showed 0/80 degenerate at
# 140,000 on the same weights. So the fibonacci result is worst-case sampling, and
# whether it MATTERS depends entirely on this measurement.
#
# BASELINES @140,000 (α=0.5), same seed and settings:
#   RAW  65.0% L3+ | looped 0/80 | char-degen 0/80
#   CHAT 33.8% L3+ | looped 0/80 | char-degen 0/80
#
# DECIDE:
#   both hold   -> the anneal bought prose coherence for free. Anneal again (0.40).
#   raw drops   -> it was bought with the capability the product needs. Stop at
#                  0.45 or revert; step_0140000.pt is intact, as is checkpoints_newmix.
set -uo pipefail
trap 'pkill -INT -P $$ 2>/dev/null; true' INT TERM
cd "$(dirname "$0")"
source ../venv-xpu/bin/activate
export SYCL_CACHE_PERSISTENT=1 PYTORCH_ALLOC_CONF=expandable_segments:True TRITON_DEFAULT_BACKEND=intel
mkdir -p logs reports

if pgrep -f "training[.](distill|sft)" >/dev/null; then echo "a trainer is running"; exit 1; fi

DIR="${DIR:-checkpoints_anneal045}"   # override: DIR=checkpoints_anneal040 bash ...
C=$(ls -t "$DIR"/step_*.pt | head -1)
TAG=$(basename "$DIR" | sed "s/^checkpoints_//")   # report prefix follows DIR
S=$(basename "$C" | sed 's/step_0*//; s/\.pt//')
LOG="logs/anneal_code_$(date +%Y%m%d_%H%M).log"

{
echo "=== $TAG @ $S — code capability vs 140,000 (65.0 raw / 33.8 chat) ==="
echo; echo "--- 1/2 RAW ---"
python -u -m tools.code_eval -c "$C" --device xpu:0 \
  --samples 8 --temperature 0.4 --seed 1234 --repetition-penalty 1.15 \
  --json "reports/code_${TAG}_${S}_pen115.json"
echo; echo "--- 2/2 CHAT ---"
python -u -m tools.code_eval -c "$C" --device xpu:0 \
  --samples 8 --temperature 0.4 --seed 1234 --repetition-penalty 1.15 \
  --chat-template --extract --json "reports/code_${TAG}_${S}_CHAT.json"
} 2>&1 | tee "$LOG"

echo
python3 - "$S" "$TAG" <<'EOFPY'
import json,sys
S,TAG=sys.argv[1],sys.argv[2]
def g(p):
    d=json.load(open(p))["diagnostics"]
    return 100*d["per_sample_l3plus"], d["looped"], d["char_degenerate"]
r=g(f"reports/code_{TAG}_{S}_pen115.json"); c=g(f"reports/code_{TAG}_{S}_CHAT.json")
print(f"  {'frame':6}{'@140,000':>10}{'@'+S:>10}{'diff':>8}   looped  degen")
print(f"  {'RAW':6}{65.0:10.1f}{r[0]:10.1f}{r[0]-65.0:+8.1f}   {r[1]:>4}/80 {r[2]:>4}/80")
print(f"  {'CHAT':6}{33.8:10.1f}{c[0]:10.1f}{c[0]-33.8:+8.1f}   {c[1]:>4}/80 {c[2]:>4}/80")
print("\n  n=80 -> +-10pp CI. Report BOTH frames; either alone inverts the verdict.")
EOFPY
