#!/usr/bin/env bash
# ROLLOUT-REGIME TEST — the last unexplained variable in the growth result.
#
#   bash run_rollout_test.sh          # ~8.3h, 3,000 steps
#
# THE OPEN QUESTION. Two legs from the same parent on the same code+math corpus
# produced opposite prose outcomes:
#
#   mathcode 157,238 -> 163,238   top_share 0.137 -> 0.159   REGRESSED
#   grown48  157,000 -> 8,696     top_share 0.137 -> 0.104   IMPROVED
#
# Growth was NOT the mechanism: activated params never moved off 180,726,115, the
# new experts never differentiated (~90% twins, asymptotic, initialisation and
# load balancing both ruled out), and masking them made the model BETTER.
# Routing dilution was NOT the mechanism either: --expert-dropout 0.5 matched to
# the mathcode leg landed on the base (0.140/0.500) rather than on grown48
# (0.104/0.571), and reproduced the salad mode in the text.
#
# What is left is the ROLLOUT REGIME, and it is not one flag:
#
#   draws per fill = rollout_reuse * (rollout_batch // micro_batch)
#     mathcode  8 * (32//8) = 32 draws   32 distinct on-policy sequences
#     grown48   8 * (8//4)  = 16 draws    8 distinct on-policy sequences
#
# rollout_batch sets HOW MANY distinct sequences exist; micro_batch modulates how
# STALE they get before refresh. Matching both requires moving both, so this run
# treats the rollout regime as ONE variable and adopts grown48's wholesale:
# micro-batch 4, grad-accum 4, rollout-batch 8. Tokens/step is IDENTICAL at
# 16,384 either way, so the optimiser sees the same batch size.
#
# ⚠ WHAT A POSITIVE RESULT WOULD AND WOULD NOT ESTABLISH. It would show the
# rollout regime explains the growth leg — which is the answer worth having, and
# the one that transfers to anyone else running on-policy distillation. It would
# NOT decompose WHICH half (fewer sequences, or fresher ones). That is a
# follow-up: rollout-batch 16 at micro-batch 8 matches grown48's staleness while
# keeping 16 sequences, and separates them.
#
# THE GATE:
#   prose reaches ~0.104/0.571      -> the rollout regime IS the mechanism.
#                                      Growth was incidental all along, and the
#                                      finding is a config lever, not an
#                                      architecture one.
#   prose stays ~0.137-0.159        -> NONE of growth, dilution or rollout regime
#                                      explains it. That would be genuinely
#                                      strange and worth reporting as an open
#                                      anomaly rather than quietly dropping.
set -uo pipefail
STOP=0
trap 'STOP=1; pkill -INT -P $$ 2>/dev/null; true' INT TERM
cd "$(dirname "$0")"
source ../venv-xpu/bin/activate
export TRITON_DEFAULT_BACKEND=intel
unset SYCL_CACHE_PERSISTENT PYTORCH_ALLOC_CONF
export PYTHONFAULTHANDLER=1 PYTHONUNBUFFERED=1

DIR=checkpoints_rollout
SRC=checkpoints_base/step_0157000.pt      # the SAME parent both legs used
TEACHER=ByteDance/Ouro-2.6B-Thinking
FILES='data_teacher_code/shard_*.jsonl,data_teacher_math/shard_*.jsonl'
STEPS="${STEPS:-3000}"
LOG="logs/rollout_$(date +%Y%m%d_%H%M).log"
mkdir -p logs reports "$DIR"

if pgrep -f "python -u -m training\.(distill|sft)" >/dev/null; then
  echo "a trainer is already running"; exit 1; fi
[ -f "$SRC" ] || { echo "missing $SRC"; exit 1; }
if [ ! -f "$DIR/step_0000000.pt" ]; then
  cp "$SRC" "$DIR/step_0000000.pt"
  python - <<'PY'
import torch, os
p="checkpoints_rollout/step_0000000.pt"
ck=torch.load(p,map_location="cpu",weights_only=False)
ck["step"]=0                    # fresh lineage — see the 09-01 zero-step failure
torch.save(ck,p+".tmp"); os.replace(p+".tmp",p); print("  seeded at step 0")
PY
fi

at=$(basename "$(ls -t $DIR/step_*.pt | head -1)" | sed 's/step_0*//; s/\.pt//'); at=${at:-0}
echo "=== rollout-regime test: $at -> $((at+STEPS)) ==="
echo "=== mathcode replication with grown48's rollout regime (mb4/ga4, rb8) ==="
echo "=== log: $LOG ==="

python -u -m training.distill \
  --student-variant mythouro_distill_tiny \
  --student-device xpu:0 --teacher-device xpu:0 --teacher-id "$TEACHER" \
  --seq-len 1024 --micro-batch 4 --grad-accum 4 \
  --warmup-steps 500 --lr 1e-4 --min-lr 3e-5 --start-loops 4 \
  --depth-reg-coeff 0.3 --divergence rev_kl \
  --use-sandwich-norm --use-depth-aware-init \
  --teacher-mix-alpha 0.45 --rollout-len 64 --rollout-batch 8 --rollout-reuse 8 \
  --teacher-data-ratio 0.2 --teacher-data-files "$FILES" \
  --onpolicy-lambda 0.7 \
  --ckpt-dir "$DIR" \
  --ckpt-every-mins 15 --ckpt-milestone-every 500 --keep-last 8 \
  --num-workers 0 --trust-remote-code --log-every 50 \
  --total-steps $((at+STEPS)) \
  > >(tee -a "$LOG") 2> >(tee -a "$LOG.err" >&2)

echo
echo "=== READ ON PROSE, >=3 checkpoints ==="
echo "  bash run_prose_readout.sh $(ls -t $DIR/step_0*.pt 2>/dev/null | head -3 | tr '\n' ' ')"
echo "  bars (same session, same parent, same corpus):"
echo "    mathcode  0.159/0.482  looping 5/30   <- rollout regime 32/32 draws"
echo "    DROPOUT   0.140/0.500  looping 3/30   <- dilution, refuted"
echo "    base      0.137/0.520  looping 1/30"
echo "    grown48   0.104/0.571  looping 1/30   <- the target"
