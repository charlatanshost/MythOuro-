#!/usr/bin/env bash
# G2 SWEEP — is the L3+ drop a MONOTONE decline or a U-SHAPED transient?
#
#   bash run_g2_sweep.sh
#
# WHY. The G2 readout measured, within one session and at identical settings:
#     step 0 (= bit-exact 278M base)   L3+ 68.8%   L4  9/320
#     step 8,696 (397M, code+math)     L3+ 48.4%   L4 12/320
# The L3+ drop is real (z=-5.22, p<0.0001). The L4 "gain" is NOT (z=+0.67,
# p=0.51) — +3/320 is sampling noise. By the pre-registered G2 bar that is a
# FAIL.
#
# BUT THIS PROJECT HAS SEEN THIS EXACT SHAPE BEFORE AND IT RECOVERED. At 278M
# the code corpus first SOLD L3+ (76.6 -> 54.1) and more steps BOUGHT IT BACK
# (54.7 -> 77.5) — the roadmap calls it "a recovering transient (U-shape)".
# 48.4% at 8,696 steps could be the trough rather than the destination.
#
# Two endpoints cannot tell those apart. The intermediate checkpoints can, and
# they already exist — NO NEW TRAINING. That is the whole point of this script.
#
#   monotone decline -> the leg is genuinely damaging L3+; code+math is the
#                       wrong corpus at this size. Switch to run_grown48_broadmix.sh.
#   dip and rise     -> U-shape; it needs STEPS, not a redesign. Keep pouring.
#
# Guessing wrong costs nights in either direction, which is what makes ~50
# minutes of eval on checkpoints you already have the cheap move.
#
# ⚠ Interpretation limit, unchanged from the G2 pre-registration: this leg moved
# THREE variables at once — size (278M->397M), corpus (code+math only), and
# --rollout-batch (32->8). And the capacity is NOMINAL: the new experts measure
# cos 0.91 to their parents at 0.40 output magnitude, having seen 11.9M tokens
# each against the 430M that shaped the originals. A bad result here does not
# refute the capacity hypothesis; it was never cleanly tested.
set -uo pipefail
cd "$(dirname "$0")"
source ../venv-xpu/bin/activate

DIR=checkpoints_grown48
STEPS="0002000 0004000 0006000"     # endpoints 0 and 8696 already measured
EV="--device xpu:0 --samples 32 --temperature 0.4 --seed 1234 --repetition-penalty 1.15"
LOG="logs/g2_sweep_$(date +%Y%m%d_%H%M).log"
mkdir -p reports logs

if pgrep -f "python -u -m (training\.(distill|sft)|tools\.code_eval)" >/dev/null; then
  echo "an eval or trainer is already using the card — stop it first."; exit 1; fi
for s in $STEPS; do
  [ -f "$DIR/step_$s.pt" ] || { echo "missing $DIR/step_$s.pt"; exit 1; }
done
echo "=== G2 sweep: $STEPS  (~17 min each, ~50 min total) ==="
echo "=== log: $LOG ==="

for s in $STEPS; do
  out=reports/code_g2_sweep_$s.json
  if [ -f "$out" ]; then echo "  $s already done — skipping"; continue; fi
  echo "=== step $s ==="
  python -u -m tools.code_eval -c "$DIR/step_$s.pt" $EV --json "$out" 2>&1 | tee -a "$LOG"
done

python3 - <<'PY' 2>&1 | tee -a "$LOG"
import json, os, math
def read(p):
    d=json.load(open(p)); n=sum(len(t["samples"]) for t in d["tasks"])
    l4=sum(1 for t in d["tasks"] for s in t["samples"] if s.get("rung",0)>=4)
    return n, d["diagnostics"]["per_sample_l3plus"], l4
pts=[(0,"reports/code_g2_control.json")]
for s in ("0002000","0004000","0006000"):
    pts.append((int(s), f"reports/code_g2_sweep_{s}.json"))
pts.append((8696,"reports/code_g2_final.json"))
rows=[]
for step,p in pts:
    if os.path.exists(p):
        n,l3,l4=read(p); rows.append((step,l3*100,l4,n))
print("\n"+"="*62)
print("  G2 SWEEP — L3+ trajectory across the leg".center(62))
print("="*62)
print(f"  {'step':>6} {'L3+':>8} {'±95%':>6} {'L4':>9}   ")
for step,l3,l4,n in rows:
    ci=1.96*math.sqrt((l3/100)*(1-l3/100)/n)*100
    bar="#"*int(l3/2)
    print(f"  {step:>6} {l3:7.1f}% {ci:5.1f}  {l4:>4}/{n:<4} {bar}")
if len(rows)>=4:
    ls=[r[1] for r in rows]
    lo=min(range(len(ls)), key=lambda i: ls[i])
    print()
    if lo in (0,len(ls)-1):
        print("  SHAPE: MONOTONE — no interior trough.")
        if ls[-1]<ls[0]:
            print("  => the leg is genuinely degrading L3+. code+math is the wrong")
            print("     corpus at this size. Next: run_grown48_broadmix.sh.")
    else:
        rise=ls[-1]-ls[lo]
        print(f"  SHAPE: U — trough at step {rows[lo][0]} ({ls[lo]:.1f}%), "
              f"recovered {rise:+.1f} pp since.")
        print("  => consistent with the documented recovering transient.")
        print("     Needs STEPS, not a redesign. Keep pouring and re-read.")
    print("\n  ⚠ three variables moved this leg (size, corpus, rollout-batch) and")
    print("    the new experts are 91% twins — this sweep describes the LEG,")
    print("    not the capacity hypothesis.")
print("="*62)
PY
