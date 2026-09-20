#!/usr/bin/env bash
# RUNG 6 — on-policy instruction answering. The student generates the ANSWER,
# the teacher scores it. The one thing this model has never practised.
#
#   PROFILE=1 bash run_rung6.sh      # ~5 min: measure the step time FIRST
#   bash run_rung6.sh                # 3,000 steps, ~? h — the profile says
#
# WHY. 2026-09-18: the teacher scores 97.8% L3+ / 70.6% L4 / 0.615 distinct1 on
# the student's own instruments; the student sits at 68% / 4.5% / 0.559 and two
# token curves went flat there while soft-KL to the teacher kept falling.
# Tokens, FFN width, α (fully annealed) and the LR floor are all measured and
# none of them is what holds L4 at 4%. The roadmap wrote the answer in August:
# "more offline tokens only sharpen the attractor. The cure is a different
# objective, not more data." Every cheap test since has reproduced that.
#
# WHAT CHANGES. Until now every on-policy rollout was seeded with the first 16
# tokens of a corpus row — the student practised CONTINUING documents, for
# three months, and got good at it. --onpolicy-instruct wraps the row in the
# trained ChatML instead:
#     <|im_start|>user {instruction}\n\n{snippet}<|im_end|><|im_start|>assistant
# and the student generates the ANSWER, the teacher scores every token of it,
# and the loss is masked to the generated span so the prompt is not trained
# on. Buffer, teacher scoring, divergence, exit_pdf weighting — unchanged.
#
# α=0.5, NOT 0 — a design input from 2026-09-20, not a reversal. The probe on
# instruction prompts showed the student on-domain 15/15 for code at α=0 but
# only 7/15 for medical (asked about ibuprofen, it wrote a Python function). At
# α=0.5 medical is 13/15 and the rollouts are real answer attempts. So this is
# the START of a new α ladder for a new regime, to be annealed down as the
# medical on-domain rate at α=0 rises — exactly how corpus continuation went
# 0.6 → 0. The corpus-continuation anneal itself stands.
#
# COST. α>0 runs the teacher inside generation (a switch, not a dial), and the
# prompt is 48 tokens instead of 16 with O(L²) decode. Expect the pre-α=0 pace
# or worse: ~12-20 s/step is the arithmetic, and arithmetic on step times has
# been wrong five times this month. PROFILE=1 measures it. Do not plan a night
# on a number that has not been measured.
#
# SEED: anneal_lr@3000 — best three-checkpoint L0/L3+ on record. It sits at
# LR 1e-5, so this leg warms back up: a new objective wants real LR.
#
# THE GATE, pre-registered (3-checkpoint means, per the 09-17 protocol):
#   capability   code L4 (4.5% at seed; the teacher is 70.6%) and the medical
#                diabetes-symptom recall (3-4/30 at seed). These are the two
#                numbers that have NEVER moved. Either moving clearly is the
#                result this rung exists to produce.
#   guard rail   prose distinct1 >= ~0.53 and LOOPING < 5/90; bare L3+ not
#                collapsing below ~60%. This is the "did instruction-shaped
#                training damage the base" check that every chat leg failed.
#   anneal signal  run_instruct_probe.sh on the endpoint: the medical on-domain
#                rate at α=0. Above ~80% -> the next leg can drop α to 0.25.
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

DIR=checkpoints_rung6
SRC=checkpoints_anneal_lr/step_0003000.pt
VARIANT=mythouro_distill_wide
TEACHER=ByteDance/Ouro-2.6B-Thinking
FILES="data_teacher_code/shard_*.jsonl,data_teacher_math/shard_*.jsonl,data_teacher_v2/shard_*.jsonl,data_teacher_med/shard_*.jsonl"
ALPHA="${ALPHA:-0.5}"
PLEN="${PLEN:-48}"; RLEN="${RLEN:-48}"
STEPS="${STEPS:-3000}"
mkdir -p logs reports

if pgrep -f "[p]ython -u -m training\.(distill|sft)" >/dev/null; then echo "a trainer is running"; exit 1; fi
[ -f "$SRC" ] || { echo "missing $SRC"; exit 1; }
FREE=$(df --output=avail -BG . | tail -1 | tr -dc '0-9'); [ "$FREE" -lt 25 ] && { echo "only ${FREE}G free"; exit 1; }

if [ "${PROFILE:-0}" = "1" ]; then
  DIR=checkpoints_rung6_profile; rm -rf "$DIR"
  PROF="--profile-steps 10 --profile-warmup 5"; TOTAL=100000
  LOG="logs/rung6_profile_$(date +%Y%m%d_%H%M).log"
else
  PROF=""; LOG="logs/rung6_$(date +%Y%m%d_%H%M).log"
fi
mkdir -p "$DIR"
if [ ! -f "$DIR/step_0000000.pt" ]; then
  cp "$SRC" "$DIR/step_0000000.pt"
  python - "$DIR/step_0000000.pt" <<'PY'
import torch, os, sys
p=sys.argv[1]; ck=torch.load(p,map_location="cpu",weights_only=False); ck["step"]=0
torch.save(ck,p+".tmp"); os.replace(p+".tmp",p); print("  seeded at step 0")
PY
fi
at=$(basename "$(ls -t $DIR/step_*.pt | head -1)" | sed 's/step_0*//; s/\.pt//'); at=${at:-0}
[ "${PROFILE:-0}" = "1" ] || TOTAL=$STEPS

# stall watchdog — new seeding code on the XPU path gets one (field notes rule)
STALE_SEC="${STALE_SEC:-600}"
run_watched() { local log="$1"; shift; : > "$log"; "$@" > >(tee -a "$log") 2>&1 & local pid=$!
  while kill -0 "$pid" 2>/dev/null; do sleep 15
    local age=$(( $(date +%s) - $(stat -c %Y "$log") ))
    if [ "$age" -gt "$STALE_SEC" ]; then echo; echo "!!! WATCHDOG: silent ${STALE_SEC}s — killing"; tail -3 "$log"
      kill -TERM "$pid" 2>/dev/null; sleep 3; kill -KILL "$pid" 2>/dev/null; pkill -KILL -P "$pid" 2>/dev/null; return 124; fi
  done; wait "$pid"; }

echo "=== RUNG 6 ${PROFILE:+PROFILE }: $at -> $TOTAL   alpha=$ALPHA  prompt $PLEN + rollout $RLEN  from anneal@3000 ==="
echo "=== the setup log must show 'RUNG 6 on-policy INSTRUCT seeds' — if not, the flag did not take ==="
echo "=== log: $LOG ==="
run_watched "$LOG" \
  python -u -m training.distill \
    --student-variant "$VARIANT" \
    --student-device xpu:0 --teacher-device xpu:0 --teacher-id "$TEACHER" \
    --seq-len 1024 --micro-batch 2 --grad-accum 8 \
    --warmup-steps 500 --lr 1e-4 --min-lr 3e-5 --start-loops 4 \
    --loop-loss-weighting exit_pdf --depth-reg-coeff 0.1 \
    --divergence rev_kl \
    --use-sandwich-norm --use-depth-aware-init \
    --onpolicy-instruct --instruct-prompt-len "$PLEN" \
    --teacher-mix-alpha "$ALPHA" --rollout-len "$RLEN" --rollout-batch 8 --rollout-reuse 8 \
    --teacher-data-ratio 0.2 --teacher-data-files "$FILES" \
    --onpolicy-lambda 0.7 \
    --ckpt-dir "$DIR" --ckpt-every-mins 20 --ckpt-milestone-every 1000 --keep-last 3 \
    --num-workers 0 --trust-remote-code --log-every 10 \
    $PROF --total-steps "$TOTAL"
rc=$?

if [ "${PROFILE:-0}" = "1" ]; then
  rm -rf "$DIR"; echo
  python3 - "$LOG" <<'PY'
import re, sys
t=open(sys.argv[1],errors="ignore").read()
m=re.search(r"STEP PROFILE over \d+ steps \(([\d.]+) ms/step", t)
if not m: print("  no profile — read the log"); sys.exit(1)
ms=float(m.group(1))
print(f"  profiled {ms:,.0f} ms/step   (278M α=0 corpus-seed: 6,751;  wide α=0: 7,153)")
for r in ("rollout","teacher_fwd","backward","student_fwd"):
    mm=re.search(rf"{r}\s+([\d.]+)\s+([\d.]+)%", t)
    if mm: print(f"    {r:12s} {float(mm.group(1)):8.0f} ms  {float(mm.group(2)):5.1f}%")
real=ms/7153*7.69
print(f"\n  projected real: ~{real:.1f} s/step  ->  ~{real*3000/3600:.1f} h per 3,000-step leg   (projection: ratio x the wide real 7.69)")
print("  > ~12.5 h does not fit a night. Options: RLEN=32, or --rollout-reuse 16, or split the leg.")
PY
  exit 0
fi

echo
echo "=== READ IT — 3 checkpoints on BOTH instruments, plus the anneal signal ==="
echo "  bash run_prose_readout.sh $DIR/step_000{1000,2000,3000}.pt"
echo "  for s in 1000 2000 3000; do bash run_eval.sh $DIR/step_000\$s.pt rung6_\$s; done"
echo "  python3 tools/medical_readout.py reports/prose_rung6_*.json"
echo "  bash run_instruct_probe.sh $DIR/step_0003000.pt        # medical on-domain at α=0 -> next leg's α"
echo
echo "  SEED (anneal@3000):  L4 3.6%  diabetes 3/30  |  distinct1 0.546  LOOP 4/90  L3+ 81.7%"
echo "  CEILING (teacher):   L4 70.6%               |  distinct1 0.615"
