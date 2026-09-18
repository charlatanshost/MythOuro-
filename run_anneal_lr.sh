#!/usr/bin/env bash
# LR ANNEAL — the first time this lineage's learning rate goes below 3e-5.
#
#   bash run_anneal_lr.sh          # 3,000 steps from wide@9000, ~6.4 h
#
# WHY. The ceiling is measured (2026-09-18): the teacher scores 97.8% L3+ /
# 70.6% L4 / 0.615 distinct1 on the student's own instruments, against the
# student's 68.2% / 4.5% / 0.559. Two curves went flat far below it while
# soft-KL to the teacher kept falling. Tokens, FFN width and the teacher are
# ruled out. The cheapest thing left is the schedule — and the owner's record
# says the schedule has moved this model dramatically before.
#
# WHAT THE CURVE ACTUALLY RAN. `_cosine_lr` is computed against the ABSOLUTE
# step and `--total-steps`. Each leg resumes on a cosine that reaches
# --min-lr 3e-5 exactly at the leg boundary; the next leg's larger --total-steps
# puts the LR back up mid-cosine (leg 3 of the wide curve ran 4.9e-5 -> 3.0e-5).
# So: a jump every 3,000 steps, and **the LR has never been below 3e-5 on any
# lineage in this project.** 3e-5 is not small for a 240M model on Adam. A
# final anneal toward zero is where a great deal of quality conventionally
# lands, and it has never been run here.
#
# THIS LEG. Seeded from wide@9000 with step reset to 0 so the schedule is
# fresh, --warmup-steps 0 so it starts AT 3e-5 (exactly where leg 3 ended — no
# jump), and --min-lr 1e-6, a 30x decay over 3,000 steps:
#     step    0: 3.00e-05      step 1500: 1.55e-05      step 3000: 1.00e-06
# Adam state carries over (the wide checkpoint has it; nothing was dropped).
# ONE variable changes: the LR floor. Everything else is the curve's recipe.
#
# ⚠ THE 08-21 CORRECTION APPLIES. The August α-anneal took `committed` 12% ->
# 50% and was recorded as "the first thing all year to move content" — then
# corrected: it moved PARSEABILITY, and L3+ is not a capability metric. So this
# leg is read on L4 and distinct1 with 3-checkpoint means, not on L3+ alone.
#
# THE GATE, against wide pt3 (3-ckpt code, 3-ckpt prose):
#   distinct1 0.559  halt 2.75  |  L0 7.7%  L3+ 68.2%  L4 4.5%
#   distinct1 or L4 clearly up (>~2 sd)   -> the schedule was binding. Take the
#       annealed checkpoint as the base and rethink the curve's LR floor.
#   flat                                  -> the plateau is structural, not a
#       schedule artefact. Next is alpha (0 vs 0.45 on capability, matched),
#       then depth.
#   down                                  -> the low LR let something drift
#       (routing? halt?). Read the text; the answer is in which thing moved.
#
# ⚠ AN ANNEALED CHECKPOINT IS AN ENDPOINT, NOT A SEED. At 1e-6 the optimizer
# has effectively stopped. Resuming training from it means warming up again —
# fine, but know that is what you are doing.
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

DIR=checkpoints_anneal_lr
SRC=checkpoints_wide/step_0009000.pt
VARIANT=mythouro_distill_wide
TEACHER=ByteDance/Ouro-2.6B-Thinking
FILES="data_teacher_code/shard_*.jsonl,data_teacher_math/shard_*.jsonl,data_teacher_v2/shard_*.jsonl,data_teacher_med/shard_*.jsonl"
STEPS="${STEPS:-3000}"
LOG="logs/anneal_lr_$(date +%Y%m%d_%H%M).log"
mkdir -p logs reports "$DIR"

if pgrep -f "[p]ython -u -m training\.(distill|sft)" >/dev/null; then
  echo "a trainer is already running"; exit 1; fi
[ -f "$SRC" ] || { echo "missing $SRC"; exit 1; }
FREE=$(df --output=avail -BG . | tail -1 | tr -dc '0-9')
[ "$FREE" -lt 25 ] && { echo "only ${FREE}G free"; exit 1; }

if [ ! -f "$DIR/step_0000000.pt" ]; then
  cp "$SRC" "$DIR/step_0000000.pt"
  python - "$DIR/step_0000000.pt" <<'PY'
import torch, os, sys
p=sys.argv[1]
ck=torch.load(p,map_location="cpu",weights_only=False)
assert ck.get("optimizer"), "seed has no optimizer state — the anneal needs a warm Adam"
ck["step"]=0                       # fresh schedule: 0 -> 3000, no warmup
torch.save(ck,p+".tmp"); os.replace(p+".tmp",p); print("  seeded at step 0, Adam state kept")
PY
fi

at=$(basename "$(ls -t $DIR/step_*.pt | head -1)" | sed 's/step_0*//; s/\.pt//'); at=${at:-0}
echo "=== LR ANNEAL: $at -> $STEPS   3e-5 -> 1e-6, no warmup, from wide@9000 ==="
echo "=== first step line should show lr ~3.0e-05, NOT 0 and NOT climbing ==="
echo "=== log: $LOG ==="

python -u -m training.distill \
  --student-variant "$VARIANT" \
  --student-device xpu:0 --teacher-device xpu:0 --teacher-id "$TEACHER" \
  --seq-len 1024 --micro-batch 2 --grad-accum 8 \
  --warmup-steps 0 --lr 3e-5 --min-lr 1e-6 --start-loops 4 \
  --loop-loss-weighting exit_pdf --depth-reg-coeff 0.1 \
  --divergence rev_kl \
  --use-sandwich-norm --use-depth-aware-init \
  --teacher-mix-alpha 0.0 --rollout-len 64 --rollout-batch 8 --rollout-reuse 8 \
  --teacher-data-ratio 0.2 --teacher-data-files "$FILES" \
  --onpolicy-lambda 0.7 \
  --ckpt-dir "$DIR" \
  --ckpt-every-mins 20 --ckpt-milestone-every 1000 --keep-last 3 \
  --num-workers 0 --trust-remote-code --log-every 10 \
  --total-steps "$STEPS" \
  > >(tee -a "$LOG") 2> >(tee -a "$LOG.err" >&2)

echo
echo "=== READ IT — 3 checkpoints on BOTH instruments, per the 09-17 protocol ==="
echo "  bash run_prose_readout.sh $DIR/step_000{1000,2000,3000}.pt"
echo "  for s in 1000 2000 3000; do bash run_eval.sh $DIR/step_000\$s.pt anneal_lr_\$s; done"
echo "  python3 tools/medical_readout.py reports/prose_anneal_lr_*.json"
echo
echo "  AGAINST wide pt3:  distinct1 0.559  halt 2.75  |  L0 7.7%  L3+ 68.2%  L4 4.5%"
echo "  AND the ceiling:   distinct1 0.615             |  L0 0.6%  L3+ 97.8%  L4 70.6%"
