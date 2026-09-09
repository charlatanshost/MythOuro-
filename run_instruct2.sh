#!/usr/bin/env bash
# INSTRUCTION DOSE-RESPONSE, LEG 2 — 16x oversample (~0.98 epochs).
#
#   bash run_instruct2.sh          # ~8.3h, 3,000 steps
#
# WHY. 2026-09-09 settled what breaks this model at length, with three controls
# on ONE checkpoint so only framing and sampling varied:
#
#   bare  top_p=0     22.2% degenerate   64.1% terminate   L3+ 66.6%
#   chat  top_p=0     69.1% degenerate    ~10% terminate
#   chat  top_p=0.92  66.6% degenerate
#
#   * It is NOT length. Bare framing at 512 scores exactly as at 96 and
#     terminates cleanly 64% of the time. 128/320 samples are both L3+ and
#     self-terminating.
#   * It is NOT decoding. Nucleus sampling moved degeneracy 2.5pp and made
#     closure WORSE. The attractor is in the weights.
#   * It IS the framing. Same weights, same budget, 22.2% vs 69.1%.
#
# The model has seen 0.27 epochs of a 1.0M-token chat corpus and nothing else in
# ChatML. Being out-of-distribution there is the expected consequence of never
# having trained on it. This leg raises the dose.
#
# ⚠ THE DOSE WINDOW IS THE WHOLE EXPERIMENT, AND IT IS GENUINELY OPEN.
# Rung 8's limits — 10.3 epochs cost −25pp raw code L3+, 1.35 epochs cost −6.2pp
# — were ALL measured on `data_teacher_chat`, the 26,130-row harvest later shown
# to be "1.22M UNUSABLE tokens that passed every structural check". On the CLEAN
# corpus there is exactly ONE datapoint: 0.27 epochs, which cost nothing
# (code L3+ 70.0% vs 65.0% at the seed, halt 2.78 vs 2.88, LOOPING 1/90 vs 4/90).
# The clean-corpus dose-response above 0.27 epochs has never been measured.
#
# DOSE: 16x oversample = 40,720/407,852 rows = 9.98% of the mix = ~0.98 epochs
# over 3,000 steps. Measured, not estimated. That is 3.6x leg 1 and still below
# the 1.35 epochs that cost 6.2pp on the BAD corpus.
#
# ⚠ SEEDED FROM THE SAME CHECKPOINT AS LEG 1, deliberately. Chaining from
# instruct@3,000 would confound dose with step count — the mistake that made the
# grown48 result unreadable for a week (2026-09-05: "match STEPS before
# attributing a difference to mechanism"). Same start, same steps, only the dose
# differs.
#
# THE GATE, pre-registered:
#   chat degeneracy drops well below 69.1%  -> dose is the lever. Instruction
#       data fixes ChatML out-of-distribution, and the axis rung 8 closed is
#       open. Continue up the dose curve.
#   chat degeneracy unchanged               -> 1 epoch is still not enough, OR
#       1.0M tokens is simply too small a corpus to move it. Harvest more before
#       spending another night on dose.
#   bare code L3+ falls below ~55%          -> STOP. That is the rung-8 capacity
#       loss reappearing on the clean corpus, and it means the dose window is
#       narrower than the corpus is useful. 65-70% is the operating band.
#   halt depth falls below ~2.70            -> depth is being spent (2.88 is the
#       exit_pdf baseline, 2.00 the pre-exit_pdf floor).
#
# ⚠ Medical was checked before building this (2026-09-09) and leg 1 did NOT harm
# it: stutter 0/30 and looping 0/30 on the medical seeds across all three
# instruct checkpoints, the cleanest in the lineage. Factual recall was FLAT —
# diabetes symptoms 20%, same as base and exit_pdf@6,000; the seed's 0% was a bad
# draw, so leg 1's apparent "recovery" was regression to the mean, not a win.
# Re-check medical the same way after this leg; a higher dose is where format
# training would start to cost content if it is going to.
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

DIR=checkpoints_instruct2
SRC=checkpoints_exitpdf/step_0007200.pt      # SAME start as leg 1 — only dose differs
TEACHER=ByteDance/Ouro-2.6B-Thinking
CC='data_teacher_chat_clean/shard_*.jsonl'
FILES="data_teacher_code/shard_*.jsonl,data_teacher_math/shard_*.jsonl,data_teacher_v2/shard_*.jsonl,data_teacher_med/shard_*.jsonl"
for _ in $(seq 16); do FILES="$FILES,$CC"; done   # 16x oversample
STEPS="${STEPS:-3000}"
LOG="logs/instruct2_$(date +%Y%m%d_%H%M).log"
mkdir -p logs reports "$DIR"

if pgrep -f "python -u -m training\.(distill|sft)" >/dev/null; then
  echo "a trainer is already running"; exit 1; fi
[ -f "$SRC" ] || { echo "missing $SRC"; exit 1; }

# ⚠ DISK. 3.35GB per checkpoint. Leg 1 left 56G free at 94% used, and this leg
# writes 7 permanent milestones plus the rolling set. keep-last is 4 here, not
# 8, which saves ~13GB against leg 1's retention.
FREE=$(df --output=avail -BG . | tail -1 | tr -dc '0-9')
if [ "$FREE" -lt 40 ]; then
  echo "only ${FREE}G free — need ~40G. Clear non-milestone checkpoints first:"
  echo "  ls checkpoints_instruct/step_0*.pt | grep -vE '(000|500)\.pt$'"
  exit 1; fi

if [ ! -f "$DIR/step_0000000.pt" ]; then
  cp "$SRC" "$DIR/step_0000000.pt"
  python - <<'PY'
import torch, os
p="checkpoints_instruct2/step_0000000.pt"
ck=torch.load(p,map_location="cpu",weights_only=False)
ck["step"]=0                      # fresh lineage — see the 09-01 zero-step failure
torch.save(ck,p+".tmp"); os.replace(p+".tmp",p); print("  seeded at step 0")
PY
fi

at=$(basename "$(ls -t $DIR/step_*.pt | head -1)" | sed 's/step_0*//; s/\.pt//'); at=${at:-0}
echo "=== instruction leg 2: $at -> $((at+STEPS))  (chat_clean 16x = 9.98%, ~0.98 epochs) ==="
echo "=== EXPECT 5 corpus directories below, chat_clean with 48 shards (3 x 16). ==="
echo "=== log: $LOG ==="

python -u -m training.distill \
  --student-variant mythouro_distill_tiny \
  --student-device xpu:0 --teacher-device xpu:0 --teacher-id "$TEACHER" \
  --seq-len 1024 --micro-batch 2 --grad-accum 8 \
  --warmup-steps 500 --lr 1e-4 --min-lr 3e-5 --start-loops 4 \
  --loop-loss-weighting exit_pdf --depth-reg-coeff 0.1 \
  --divergence rev_kl \
  --use-sandwich-norm --use-depth-aware-init \
  --teacher-mix-alpha 0.45 --rollout-len 64 --rollout-batch 8 --rollout-reuse 8 \
  --teacher-data-ratio 0.2 --teacher-data-files "$FILES" \
  --onpolicy-lambda 0.7 \
  --ckpt-dir "$DIR" \
  --ckpt-every-mins 15 --ckpt-milestone-every 500 --keep-last 4 \
  --num-workers 0 --trust-remote-code --log-every 50 \
  --total-steps $((at+STEPS)) \
  > >(tee -a "$LOG") 2> >(tee -a "$LOG.err" >&2)

echo
echo "=== READ IN THIS ORDER ==="
echo "  1. THE GATE — chat degeneracy at 512 (baseline 69.1% on the seed):"
echo "     bash run_chat_eval.sh $DIR/step_0003000.pt instruct2_3000"
echo "  2. GUARD RAIL — bare code must stay in the 65-70% band:"
echo "     bash run_eval.sh $DIR/step_0003000.pt instruct2_3000"
echo "  3. Prose, halt depth and the medical seeds, >=3 checkpoints:"
echo "     bash run_prose_readout.sh $DIR/step_000{2000,2500,3000}.pt"
echo "  4. Medical specifically — stutter/looping and diabetes symptoms:"
echo "     python3 tools/medical_readout.py reports/prose_instruct2_*.json"
echo
echo "  bars: seed chat-deg 69.1% | bare L3+ 65.0% | halt 2.88 | leg1 L3+ 70.0%"
