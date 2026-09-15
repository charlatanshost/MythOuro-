#!/usr/bin/env bash
# THE TOKEN CURVE — main-thread #2, open since June, the roadmap's go/no-go.
#
#   bash run_token_curve.sh          # ONE leg: 3,000 steps = 49.2M tokens ≈ 6.0 h
#
#   THE WIDENED MODEL (2026-09-15 onward — the 278M curve is flat, see pt3):
#   VARIANT=mythouro_distill_wide DIR=checkpoints_wide SRC=checkpoints_wide/step_0000000.pt \
#     bash run_token_curve.sh
#   Same recipe, same legs, same readouts. SRC is the Net2Wider promotion of
#   curve@9000 (436M / 240M activated), so this curve's point 0 IS the 278M
#   curve's point 3 — function-preserving, verified to 1.5e-5. Compare every
#   point against pt3 (distinct1 0.556, halt 2.81, L3+ 75.0%).
#   ...read it out...
#   bash run_token_curve.sh          # next night: the next leg, resumes itself
#
# WHAT IT ANSWERS. docs/roadmap.md: "Push 5-10x tokens, inspect every ~50M, on a
# 24-expert model, with means over >=3 checkpoints and BOTH instruments. This
# curve is the go/no-go for capital." Growth un-parked itself once without
# meeting its own condition — "the token-curve shows we've reached
# compute-optimal at current size" — and failed. This is that condition,
# finally being measured. NOTHING about a bigger student is decided until it
# reports.
#
# THE RECIPE, every piece of it gated:
#   objective    exit_pdf, depth-reg 0.1     the durable win (halt 2.00 -> 2.88)
#   rollouts     --teacher-mix-alpha 0       gated 2026-09-13: 1,500 steps, no
#                                            collapse, degeneracy IMPROVED, 1.70x
#   batching     mb2/ga8                     as every reference number was made;
#                                            mb4/ga4 changes load-balance stats
#                                            and has NOT been gated
#   corpus       code+math+v2+med            no chat data; that axis is closed
#   teacher path dense                       OPT-B is applied but no cache is
#                                            passed, so the objective is the
#                                            proven one, not the coarsened one
#
# SEED: checkpoints_alpha0/step_0001500.pt. The α=0 gate leg IS point 0 of this
# curve — its readout already exists (prose_alpha0_*, code_alpha0_1500) and every
# later point is compared to it, not only to the exit_pdf seed before it.
#
# ⚠ THE FIRST LEG HAS AN EXTRA JOB. The α=0 gate ran 1,500 steps, which catches
# COLLAPSE (fast) and cannot see DRIFT (slow). It left two soft signals: halt
# 2.74 (0.04 above the 2.70 stop) and L3+ 60.6% / committed 20.9% both down
# inside noise. If leg 1's readout shows halt below 2.70, or L3+ still falling,
# or committed still falling, that is drift and α=0 is NOT safe for the curve —
# stop, and re-run leg 1 at α=0.45 (10.5 h) before continuing.
#
# DISK. 3.12 GB per checkpoint. Unpruned, 10 legs x 3 milestones = 93 GB and it
# grew to 98% full once already. This script prunes AT THE START of each leg:
# it keeps every leg-boundary checkpoint (step % 3000 == 0 — the curve's
# backbone) and everything from the leg just finished (so its readout can still
# be run), and deletes the rest. Steady state ~20 GB. KEEP_ALL=1 disables it.
set -uo pipefail
STOP=0
trap 'STOP=1; pkill -INT -P $$ 2>/dev/null; true' INT TERM
cd "$(dirname "$0")"
source ../venv-xpu/bin/activate
export TRITON_DEFAULT_BACKEND=intel
unset SYCL_CACHE_PERSISTENT PYTORCH_ALLOC_CONF
export PYTHONFAULTHANDLER=1 PYTHONUNBUFFERED=1
export SYCL_QUEUE_THREAD_POOL_SIZE=1
export ZE_SERIALIZE=2

VARIANT="${VARIANT:-mythouro_distill_tiny}"
DIR="${DIR:-checkpoints_curve}"
SRC="${SRC:-checkpoints_alpha0/step_0001500.pt}"
TEACHER=ByteDance/Ouro-2.6B-Thinking
FILES="data_teacher_code/shard_*.jsonl,data_teacher_math/shard_*.jsonl,data_teacher_v2/shard_*.jsonl,data_teacher_med/shard_*.jsonl"
LEG="${LEG:-3000}"                 # steps per leg
TOK_PER_STEP=$((2*8*1024))         # mb2 x ga8 x seq1024 = 16,384
mkdir -p logs reports "$DIR"

if pgrep -f "python -u -m training\.(distill|sft)" >/dev/null; then
  echo "a trainer is already running"; exit 1; fi
[ -f "$SRC" ] || { echo "missing $SRC"; exit 1; }
FREE=$(df --output=avail -BG . | tail -1 | tr -dc '0-9')
[ "$FREE" -lt 30 ] && { echo "only ${FREE}G free — need ~30G for a leg plus its readout"; exit 1; }

# ---- seed once ----
if [ ! -f "$DIR/step_0000000.pt" ]; then
  cp "$SRC" "$DIR/step_0000000.pt"
  python - "$DIR/step_0000000.pt" <<'PY'
import torch, os, sys
p=sys.argv[1]
ck=torch.load(p,map_location="cpu",weights_only=False); ck["step"]=0
torch.save(ck,p+".tmp"); os.replace(p+".tmp",p); print(f"  seeded {p} at step 0")
PY
fi

# ---- where are we ----
at=$(basename "$(ls -t $DIR/step_*.pt | head -1)" | sed 's/step_0*//; s/\.pt//'); at=${at:-0}
if [ $((at % LEG)) -ne 0 ]; then
  echo "⚠ resuming MID-LEG at step $at (a previous leg was interrupted). Finishing it."
fi
leg_end=$(( (at / LEG + 1) * LEG ))
leg_no=$(( leg_end / LEG ))

# ---- prune: keep backbone + the leg just finished ----
if [ "${KEEP_ALL:-0}" != "1" ] && [ "$at" -ge "$LEG" ]; then
  prev_start=$(( at - LEG ))
  n=0
  for f in $DIR/step_*.pt; do
    s=$(basename "$f" .pt | sed 's/step_0*//'); s=${s:-0}
    if [ $((s % LEG)) -eq 0 ]; then continue; fi          # backbone
    if [ "$s" -gt "$prev_start" ]; then continue; fi      # last leg, still readable
    rm -f "$f"; n=$((n+1))
  done
  [ $n -gt 0 ] && echo "=== pruned $n intermediate checkpoints from earlier legs ==="
fi

LOG="logs/curve_${DIR#checkpoints_}_leg${leg_no}_$(date +%Y%m%d_%H%M).log"
echo "=== TOKEN CURVE [$VARIANT, $DIR] leg $leg_no: step $at -> $leg_end ==="
echo "=== cumulative tokens on this curve after this leg: $(( leg_end * TOK_PER_STEP / 1000000 ))M"
echo "=== (plus the 24.6M of the α=0 gate leg it was seeded from) ==="
echo "=== expect ~7.25 s/step, ~6.0 h.   log: $LOG ==="

python -u -m training.distill \
  --student-variant "$VARIANT" \
  --student-device xpu:0 --teacher-device xpu:0 --teacher-id "$TEACHER" \
  --seq-len 1024 --micro-batch 2 --grad-accum 8 \
  --warmup-steps 500 --lr 1e-4 --min-lr 3e-5 --start-loops 4 \
  --loop-loss-weighting exit_pdf --depth-reg-coeff 0.1 \
  --divergence rev_kl \
  --use-sandwich-norm --use-depth-aware-init \
  --teacher-mix-alpha 0.0 --rollout-len 64 --rollout-batch 8 --rollout-reuse 8 \
  --teacher-data-ratio 0.2 --teacher-data-files "$FILES" \
  --onpolicy-lambda 0.7 \
  --ckpt-dir "$DIR" \
  --ckpt-every-mins 20 --ckpt-milestone-every 1000 --keep-last 3 \
  --num-workers 0 --trust-remote-code --log-every 10 \
  --total-steps "$leg_end" \
  > >(tee -a "$LOG") 2> >(tee -a "$LOG.err" >&2)

a=$(( leg_end - 2000 )); b=$(( leg_end - 1000 )); c=$leg_end
P() { printf "%s/step_%07d.pt" "$DIR" "$1"; }
echo
echo "=== LEG $leg_no DONE. READ IT OUT BEFORE THE NEXT ONE — the next launch prunes. ==="
echo "  1. prose + halt, three checkpoints:"
echo "     bash run_prose_readout.sh $(P $a) $(P $b) $(P $c)"
echo "  2. code:"
echo "     bash run_eval.sh $(P $c) curve_leg${leg_no}"
echo "  3. medical:"
echo "     python3 tools/medical_readout.py reports/prose_curve_*.json"
echo
echo "  COMPARE AGAINST POINT 0 (the α=0 gate), not only the exit_pdf seed:"
echo "    prose  top_share 0.109  distinct1 0.545  LOOPING 2/90  stutter 0/90  halt 2.74"
echo "    code   L0 2.2%   L3+ 60.6%   L4 3.8%   committed 20.9%"
if [ "$leg_no" -eq 1 ]; then
  echo
  echo "  ⚠ LEG 1 IS ALSO THE DRIFT CHECK FOR α=0:"
  echo "    halt below 2.70, or L3+ / committed still falling  ->  drift. STOP."
  echo "    Re-run this leg at α=0.45 before continuing. Do not let it become an"
  echo "    assumption because the 1,500-step gate passed."
fi
echo
echo "  THE CURVE QUESTION, every leg: is it still gaining? Plot the means over"
echo "  >=3 checkpoints per point. Flat across 2-3 points = compute-optimal at"
echo "  this size = the un-park condition for growth is MET. Still rising = keep"
echo "  pouring; a bigger student is not yet justified."
