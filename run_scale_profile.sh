#!/usr/bin/env bash
# SCALE PROFILE — what does a step cost at 460M activated? Measured, not projected.
#
#   VARIANT=mythouro_distill_wide SEED=checkpoints_wide/step_0000000.pt bash run_scale_profile.sh
#                                        # the widened 240M-activated model — do this one
#   bash run_scale_profile.sh            # mythouro_distill_mid from scratch (already run: 1.22x)
#   MB=1 GA=16 bash run_scale_profile.sh # if mb2 OOMs (same 16,384 tokens/step)
#
# WHY. The token curve went flat across three points at 278M (2026-09-15) — the
# un-park condition for growth, met on the instruments it named. The target is
# `mythouro_distill_mid`: dim 1792, prelude/coda 4, top-k 6, the SAME 24 experts.
# 633M total / 460M activated = 2.55x the working model. Expert count is left
# alone because the 24->48 leg proved it does not move activated params.
#
# WHAT IS NOT KNOWN: the step time. At 278M with alpha=0 the step is 7.2 s and
# the teacher is out of rollout, so nearly all of it — rollout decode, backward,
# student forward — scales with the student. Arithmetic says ~18 s/step and
# ~15 h per 3,000-step leg. Arithmetic on step times has been wrong three
# times this month (3.5x -> 1.40x; ~41% -> 69%; 6.8 -> 7.25). This measures it.
#
# THE READING
#   <= ~15 s/step   a leg fits a night (~12.5 h). Grow.
#   ~15-20 s/step   a leg is ~15 h — two nights per point. Still viable, but
#                   the curve at this size is 2x slower to read. Grow, and take
#                   mb4/ga4 (1.16x, needs its own gate) seriously.
#   > ~20 s/step    a leg is 17+ h. Re-examine the shape before committing —
#                   dim 1536 (245M activated, 1.36x) may be the honest step.
#
# ⚠ MEMORY. First time a 633M student trains beside the 4.8 GB teacher on this
# card. Weights+Adam+grads ~10.1 GB, plus activations at 1.4x the width and 2x
# the prelude/coda depth. Should fit at mb2. If it page-faults ("NotPresent /
# PDE / Write" — the OOM signature, field notes), rerun with MB=1 GA=16: same
# tokens/step, and the profile is still valid because tokens/step is what the
# s/step figure is normalised to.
#
# ⚠ FRESH INIT. There is no 460M checkpoint; this profiles random weights.
# Step time does not depend on the weights, so that is fine for THIS purpose.
# It is NOT a training run — it exits after 5 warmup + 10 profiled steps
# without saving, and the ckpt dir is wiped first so it cannot resume anything.
set -uo pipefail
trap 'pkill -INT -P $$ 2>/dev/null; true' INT TERM
cd "$(dirname "$0")"
source ../venv-xpu/bin/activate
export TRITON_DEFAULT_BACKEND=intel
unset SYCL_CACHE_PERSISTENT PYTORCH_ALLOC_CONF
export PYTHONFAULTHANDLER=1 PYTHONUNBUFFERED=1
export SYCL_QUEUE_THREAD_POOL_SIZE=1
export ZE_SERIALIZE=2

TEACHER=ByteDance/Ouro-2.6B-Thinking
FILES="data_teacher_code/shard_*.jsonl,data_teacher_math/shard_*.jsonl,data_teacher_v2/shard_*.jsonl,data_teacher_med/shard_*.jsonl"
DIR=checkpoints_scale_profile
VARIANT="${VARIANT:-mythouro_distill_mid}"
SEED="${SEED:-}"                       # empty = fresh init; a path = profile THAT checkpoint
MB="${MB:-2}"; GA="${GA:-8}"
ALPHA="${ALPHA:-0.0}"    # teacher-mix in rollouts; >0 puts the teacher back in generation
LOG="logs/scale_profile_${VARIANT#mythouro_distill_}_mb${MB}_a${ALPHA}_$(date +%Y%m%d_%H%M).log"
mkdir -p logs
[ $((MB*GA*1024)) -eq 16384 ] || { echo "MB x GA x 1024 must be 16,384 (got $((MB*GA*1024)))"; exit 1; }

if pgrep -f "python -u -m training\.(distill|sft)" >/dev/null; then
  echo "a trainer is running — stop it first"; exit 1; fi
rm -rf "$DIR"; mkdir -p "$DIR"        # never resume a stale profile dir
if [ -n "$SEED" ]; then
  [ -f "$SEED" ] || { echo "missing $SEED"; exit 1; }
  cp "$SEED" "$DIR/step_0000000.pt"    # profile the real weights AND prove they resume
  echo "=== seeded from $SEED ==="
fi

# stall watchdog — a new config on the XPU gets one, per the field notes
STALE_SEC="${STALE_SEC:-300}"
run_watched() {
  local log="$1"; shift; : > "$log"
  "$@" > >(tee -a "$log") 2>&1 & local pid=$!
  while kill -0 "$pid" 2>/dev/null; do
    sleep 15
    local age=$(( $(date +%s) - $(stat -c %Y "$log") ))
    if [ "$age" -gt "$STALE_SEC" ]; then
      echo; echo "!!! WATCHDOG: no output for ${STALE_SEC}s — killing"; tail -3 "$log" | sed 's/^/    /'
      kill -TERM "$pid" 2>/dev/null; sleep 3; kill -KILL "$pid" 2>/dev/null; pkill -KILL -P "$pid" 2>/dev/null
      return 124
    fi
  done
  wait "$pid"
}

echo "=== SCALE PROFILE: $VARIANT, mb${MB}/ga${GA}, alpha=$ALPHA ==="
echo "=== 278M reference at the same recipe: 6,751 ms/step profiled, 7.23 s/step real ==="
echo "=== log: $LOG ==="
run_watched "$LOG" \
  python -u -m training.distill \
    --student-variant "$VARIANT" \
    --student-device xpu:0 --teacher-device xpu:0 --teacher-id "$TEACHER" \
    --seq-len 1024 --micro-batch "$MB" --grad-accum "$GA" \
    --warmup-steps 500 --lr 1e-4 --min-lr 3e-5 --start-loops 4 \
    --loop-loss-weighting exit_pdf --depth-reg-coeff 0.1 \
    --divergence rev_kl \
    --use-sandwich-norm --use-depth-aware-init \
    --teacher-mix-alpha "$ALPHA" --rollout-len 64 --rollout-batch 8 --rollout-reuse 8 \
    --teacher-data-ratio 0.2 --teacher-data-files "$FILES" \
    --onpolicy-lambda 0.7 \
    --ckpt-dir "$DIR" --ckpt-every-mins 999 --ckpt-milestone-every 100000 \
    --num-workers 0 --trust-remote-code --log-every 5 \
    --profile-steps 10 --profile-warmup 5 --total-steps 100000
rc=$?
rm -rf "$DIR"

echo
echo "=== RESULT ==="
python3 - "$LOG" "$MB" <<'PY'
import re, sys
txt=open(sys.argv[1], errors="ignore").read()
m=re.search(r"STEP PROFILE over \d+ steps \(([\d.]+) ms/step", txt)
if not m:
    oom = any(k in txt for k in ("NotPresent","PDE","out of memory","OutOfMemory"))
    print("  no profile produced." + ("  PAGE FAULT / OOM signature present -> rerun with MB=1 GA=16" if oom else "  read the log."))
    sys.exit(1)
tot=float(m.group(1))
print(f"  profiled: {tot:,.0f} ms/step at mb{sys.argv[2]}   (278M same recipe: 6,751 profiled)")
print(f"  ratio vs 278M: {tot/6751:.2f}x   (activated params ratio: 2.55x)")
for r in ("rollout","backward","student_fwd","teacher_fwd"):
    mm=re.search(rf"{r}\s+([\d.]+)\s+([\d.]+)%", txt)
    if mm: print(f"    {r:12s} {float(mm.group(1)):8.0f} ms  {float(mm.group(2)):5.1f}%")
real = tot/6751*7.23
print()
print(f"  projected REAL s/step (profile ratio x the 278M real 7.23): ~{real:.1f} s")
print(f"  projected 3,000-step leg: ~{real*3000/3600:.1f} h    (278M: 6.0 h)")
print()
print("  ⚠ the two 'projected' lines are the profile ratio applied to a measured")
print("    real step; they are the best available estimate, not a measurement.")
print("    The first real leg replaces them.")
PY
