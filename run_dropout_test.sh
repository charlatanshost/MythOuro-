#!/usr/bin/env bash
# EXPERT-DROPOUT TEST — was DILUTION the mechanism, or was it --rollout-batch?
#
#   bash run_dropout_test.sh          # ~8.3h, 3,000 steps
#
# THE QUESTION. Two legs from effectively the same parent, same code+math corpus,
# measured in one session at α=0.0:
#
#   base step_0157000                          top_share 0.137   distinct1 0.520
#   mathcode leg, 24 experts                   top_share 0.159   distinct1 0.482  REGRESSED
#   grown48 -> masked -> pruned24, 48 experts  top_share 0.104   distinct1 0.571  IMPROVED
#
# Growth broke the regression wall — but NOT through capacity. Activated params
# never moved off 180,726,115, the added experts never differentiated (~90% twins,
# asymptotic), and MASKING them made the model better. So the benefit was at
# TRAINING time: half of every token's routing went to experts contributing at
# ~0.40 amplitude, so the originals effectively saw ~2 slots instead of 4.
#
# The owner's sharper account — twins as a reference copy holding what the
# originals shed — was tested and REFUTED: cos(twin@8696, original@0) = 0.9424
# against cos(original@8696, original@0) = 0.9497. The twins drifted the same
# distance. No anchor.
#
# ⚠ WHAT IS STILL UNSEPARATED. Those two legs also differed in --rollout-batch
# (mathcode 32, grown48 8). Less diverse on-policy sampling is a live alternative
# explanation for the whole effect.
#
# THIS RUN ISOLATES DILUTION. 24 experts, code+math, --expert-dropout 0.5 (which
# drops effective top-k 4 -> 2, matching the growth case), and EVERYTHING ELSE
# MATCHED TO THE MATHCODE LEG — including --rollout-batch 32. One variable.
#
#   prose beats the mathcode leg (0.159/0.482)  -> DILUTION confirmed. The effect
#       is available with no extra parameters and no promotion, and it should be
#       folded into the pour.
#   prose regresses like mathcode               -> ROLLOUT-BATCH was the variable
#       all along, which is a cheaper lever than either and reframes the growth
#       result entirely.
#
# ⚠ NOT loop-weighted, NOT exit_pdf, depth-reg back to 0.3. This is a clean
# replication of the mathcode leg with ONE change. Do not fold exit_pdf in — that
# would recreate the multi-variable mistake the whole 08-27..09-01 arc was.
set -uo pipefail
STOP=0
trap 'STOP=1; pkill -INT -P $$ 2>/dev/null; true' INT TERM
cd "$(dirname "$0")"
source ../venv-xpu/bin/activate
export TRITON_DEFAULT_BACKEND=intel
unset SYCL_CACHE_PERSISTENT PYTORCH_ALLOC_CONF
export PYTHONFAULTHANDLER=1 PYTHONUNBUFFERED=1

DIR=checkpoints_dropout
SRC=checkpoints_base/step_0157000.pt          # the SAME parent both legs used
TEACHER=ByteDance/Ouro-2.6B-Thinking
FILES='data_teacher_code/shard_*.jsonl,data_teacher_math/shard_*.jsonl'
STEPS="${STEPS:-3000}"
LOG="logs/dropout_$(date +%Y%m%d_%H%M).log"
mkdir -p logs reports "$DIR"

if pgrep -f "python -u -m training\.(distill|sft)" >/dev/null; then
  echo "a trainer is already running"; exit 1; fi
[ -f "$SRC" ] || { echo "missing $SRC"; exit 1; }
[ -f "$DIR/step_0000000.pt" ] || { cp "$SRC" "$DIR/step_0000000.pt"; \
  python - <<'PY'
import torch
p="checkpoints_dropout/step_0000000.pt"
ck=torch.load(p,map_location="cpu",weights_only=False)
ck["step"]=0          # fresh lineage — see the 2026-09-01 zero-step failure
torch.save(ck,p+".tmp"); import os; os.replace(p+".tmp",p)
print("  seeded at step 0")
PY
}

at=$(basename "$(ls -t $DIR/step_*.pt | head -1)" | sed 's/step_0*//; s/\.pt//'); at=${at:-0}
echo "=== dropout test: $at -> $((at+STEPS))  (24 experts, expert-dropout 0.5) ==="
echo "=== matched to the mathcode leg EXCEPT --expert-dropout ==="
echo "=== log: $LOG ==="

python -u -m training.distill \
  --student-variant mythouro_distill_tiny \
  --student-device xpu:0 --teacher-device xpu:0 --teacher-id "$TEACHER" \
  --seq-len 1024 --micro-batch 8 --grad-accum 2 \
  --warmup-steps 500 --lr 1e-4 --min-lr 3e-5 --start-loops 4 \
  --expert-dropout 0.5 \
  --depth-reg-coeff 0.3 --divergence rev_kl \
  --use-sandwich-norm --use-depth-aware-init \
  --teacher-mix-alpha 0.45 --rollout-len 64 --rollout-batch 32 --rollout-reuse 8 \
  --teacher-data-ratio 0.2 --teacher-data-files "$FILES" \
  --onpolicy-lambda 0.7 \
  --ckpt-dir "$DIR" \
  --ckpt-every-mins 15 --ckpt-milestone-every 500 --keep-last 8 \
  --num-workers 0 --trust-remote-code --log-every 50 \
  --total-steps $((at+STEPS)) \
  > >(tee -a "$LOG") 2> >(tee -a "$LOG.err" >&2)

echo
echo "=== READ THE PROSE — that is the axis the wall was on ==="
echo "  bash run_prose_readout.sh $(ls -t $DIR/step_0*.pt 2>/dev/null | head -3 | tr '\n' ' ')"
echo "  bars, same session:  mathcode 0.159/0.482   base 0.137/0.520   grown48 0.104/0.571"
