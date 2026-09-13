#!/usr/bin/env bash
# α=0 GATE — can this model make its own rollouts without the teacher?
#
#   bash run_alpha0_gate.sh          # 1,500 steps, ~2.5 h at the probed pace
#
# THE STAKE. `--teacher-mix-alpha 0` measured **1.85x** (12,492 -> 6,751 ms/step,
# 2026-09-12 probe) — the largest single throughput lever on the board, larger
# than OPT-B's 1.40x. The token curve is ~10 nights; at 1.85x it is ~5. That is
# the whole reason to spend one evening here first.
#
# WHY IT IS NOT FREE. α is the un-collapse lever from docs/onpolicy_plan.md:
# the rollout sampling distribution is
#     α·softmax(teacher/T) + (1−α)·softmax(student/T)
# and α "drags a collapsed student's rollouts back toward the teacher's support
# — the un-collapse lever that makes on-policy distillation viable from a
# degenerate checkpoint." It was set to 0.45 when this model WAS collapsing.
#
# ⚠ α IS A SWITCH, NOT A DIAL. mythouro/training_utils.py:
#   use_teacher = teacher is not None and teacher_mix_alpha > 0.0
# Any α > 0 runs the teacher once per generated token and pays the full 5.7 s.
# There is no α=0.2 that buys half the saving. So this is a binary decision and
# it deserves a real gate rather than a taper.
#
# THE ARGUMENTS, neither of which is evidence:
#   FOR  the model is no longer degenerate — code L0 2-7% against a 4-31%
#        historical band, prose LOOPING 1/90, halt 2.88 converged. And on-policy
#        distillation is *meant* to train on the student's own distribution;
#        teacher-mixing is a crutch, so removing it is arguably more correct.
#   AGAINST  collapse is exactly what α was added to prevent, and this model has
#        collapsed before. "It looks fine now" is how the 48-expert growth leg
#        was justified, and that cost a week.
#
# ⚠ ALSO CHANGED: --micro-batch 4 --grad-accum 4 (was 2/8). Tokens/step is
# IDENTICAL at 16,384 and the optimisation is mathematically identical, so this
# is not a confound for quality — but it is a second change, so if throughput
# comes out oddly, that is where to look. Memory ~9.6 GB against the ~19.3 GB
# that page-faulted at mb8 (run_exitpdf.sh:44).
#
# THE GATE, pre-registered against the exit_pdf seed this branches from:
#   collapse — prose LOOPING >= 5/90, or stutter up sharply, or salad returns
#       -> α=0 is not survivable. Keep α=0.45 and run the curve at ~10 nights.
#       This is the outcome α exists to prevent and the one to watch hardest.
#   code regresses — L3+ below ~55% (seed 65.0%, checkpoint sd 13.6pp) or L0
#       above ~12% (seed 4.7%)
#       -> the rollouts got worse even without visible degeneracy. Same verdict.
#   halt falls below ~2.70 (seed 2.88, sd 0.001)
#       -> depth is being spent. Same verdict.
#   everything holds
#       -> take α=0. The curve halves, and the recipe is also more honestly
#          on-policy than it was.
#
# ⚠ 1,500 steps is chosen to catch COLLAPSE, which appears fast, not slow
# capability drift, which does not. A clean gate here does NOT license α=0 for a
# 30,000-step curve without re-reading at the first curve milestone.
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

DIR=checkpoints_alpha0
SRC=checkpoints_exitpdf/step_0007200.pt   # the seed every reference number below is from
TEACHER=ByteDance/Ouro-2.6B-Thinking
FILES="data_teacher_code/shard_*.jsonl,data_teacher_math/shard_*.jsonl,data_teacher_v2/shard_*.jsonl,data_teacher_med/shard_*.jsonl"
STEPS="${STEPS:-1500}"
LOG="logs/alpha0_$(date +%Y%m%d_%H%M).log"
mkdir -p logs reports "$DIR"

if pgrep -f "python -u -m training\.(distill|sft)" >/dev/null; then
  echo "a trainer is already running"; exit 1; fi
[ -f "$SRC" ] || { echo "missing $SRC"; exit 1; }
FREE=$(df --output=avail -BG . | tail -1 | tr -dc '0-9')
[ "$FREE" -lt 20 ] && { echo "only ${FREE}G free — need ~20G for 4 milestones + rolling"; exit 1; }

if [ ! -f "$DIR/step_0000000.pt" ]; then
  cp "$SRC" "$DIR/step_0000000.pt"
  python - <<'PY'
import torch, os
p="checkpoints_alpha0/step_0000000.pt"
ck=torch.load(p,map_location="cpu",weights_only=False)
ck["step"]=0                      # fresh lineage — see the 09-01 zero-step failure
torch.save(ck,p+".tmp"); os.replace(p+".tmp",p); print("  seeded at step 0")
PY
fi

at=$(basename "$(ls -t $DIR/step_*.pt | head -1)" | sed 's/step_0*//; s/\.pt//'); at=${at:-0}
echo "=== α=0 gate: $at -> $((at+STEPS))   (pure-student rollouts, mb4/ga4) ==="
echo "=== EXPECT ~6.8 s/step if the probe holds; 12.5 would mean α did not take ==="
echo "=== log: $LOG ==="

python -u -m training.distill \
  --student-variant mythouro_distill_tiny \
  --student-device xpu:0 --teacher-device xpu:0 --teacher-id "$TEACHER" \
  --seq-len 1024 --micro-batch 4 --grad-accum 4 \
  --warmup-steps 500 --lr 1e-4 --min-lr 3e-5 --start-loops 4 \
  --loop-loss-weighting exit_pdf --depth-reg-coeff 0.1 \
  --divergence rev_kl \
  --use-sandwich-norm --use-depth-aware-init \
  --teacher-mix-alpha 0.0 --rollout-len 64 --rollout-batch 8 --rollout-reuse 8 \
  --teacher-data-ratio 0.2 --teacher-data-files "$FILES" \
  --onpolicy-lambda 0.7 \
  --ckpt-dir "$DIR" \
  --ckpt-every-mins 20 --ckpt-milestone-every 500 --keep-last 4 \
  --num-workers 0 --trust-remote-code --log-every 50 \
  --total-steps $((at+STEPS)) \
  > >(tee -a "$LOG") 2> >(tee -a "$LOG.err" >&2)

echo
echo "=== READ IN THIS ORDER ==="
echo "  1. COLLAPSE is the thing α prevents — prose first, 3 checkpoints:"
echo "     bash run_prose_readout.sh $DIR/step_000{0500,1000,1500}.pt"
echo "     gate: LOOPING >= 5/90, stutter up sharply, or salad returning = STOP"
echo "           halt below ~2.70 = depth spent = STOP"
echo "  2. Code, the capability guard rail:"
echo "     bash run_eval.sh $DIR/step_0001500.pt alpha0_1500"
echo "     gate: L3+ below ~55% or L0 above ~12% = STOP"
echo "  3. Medical, same instrument as every other leg:"
echo "     python3 tools/medical_readout.py reports/prose_alpha0_*.json"
echo "  4. READ THE TEXTS. Counts pointed the wrong way three times this month."
echo
echo "  SEED REFERENCES (exit_pdf, what this branches from):"
echo "    code   L0 4.7%   L3+ 65.0%   L4 4.4%"
echo "    prose  top_share 0.106  distinct1 0.527  LOOPING 4/90  stutter 2/90"
echo "    halt   2.88/4 (sd 0.001)"
echo "    speed  12.34 s/step at mb2/ga8 α=0.45; this run should be ~6.8"
