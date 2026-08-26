#!/usr/bin/env bash
# PROFILE THE CURRENT LOOP — where does step time go NOW? ~6 min, saves nothing.
#
#   bash run_profile_now.sh
#
# WHY NOW. The cost breakdown on record is from @64,000 with the OLD teacher
# corpus (10.9M tokens). The corpus is now 92M and the config is reuse=8, where
# the recorded share was rollout 31.2% / teacher 54.2%. The teacher-logit cache
# — the highest-value unbuilt optimization — is justified by that 54.2%, so it
# should be re-measured against the CURRENT setup before anyone builds it or
# sizes a card purchase around it.
#
# --profile-steps is DIAGNOSTIC: it runs --profile-warmup steps (discarding
# compile, allocator growth and SYCL cache misses), profiles N, prints per-region
# ms/step and share, and EXITS WITHOUT SAVING. Nothing is written, no checkpoint
# moves, the card is free again in minutes.
#
# ⚠ READ THE SHARES, NOT THE tok/s. Per-region device syncs suppress pipelining,
# so a profiled run's throughput is pessimistic by design (training_throughput.md).
#
# ⚠ MUST MATCH THE PRODUCTION SCHEDULE, or the profile measures the wrong model.
# The first attempt (2026-08-26 12:44) omitted --warmup-steps/--lr/--min-lr and
# passed --total-steps 999999. LoopCurriculum ramps against total_steps, so step
# 157,240 landed EARLY in a 999,999-step schedule and ran at **n_loops 2**, not
# the production 4 — and LR restarted at 2.84e-04 instead of the 3e-5 floor.
# The teacher is a separate 2.6B model and does NOT scale with n_loops, while
# rollout / student_fwd / backward all do, so that run OVERSTATED the teacher
# share. It reported 45.5%, just over the "build the cache" threshold.
# --total-steps 163238 (= current step + a real 6,000-step leg) puts step 157k
# past the ramp at n_loops 4 with LR at the floor.
#
# ON RECORD at reuse=8, @64,000, old corpus:
#   rollout 31.2% | teacher 54.2% | backward + student_fwd + data ~14%
# FIRST ATTEMPT (n_loops 2, WRONG): rollout 46.3% | teacher 45.5%
set -uo pipefail
trap 'pkill -INT -P $$ 2>/dev/null; true' INT TERM
cd "$(dirname "$0")"
source ../venv-xpu/bin/activate
export SYCL_CACHE_PERSISTENT=1 PYTORCH_ALLOC_CONF=expandable_segments:True TRITON_DEFAULT_BACKEND=intel
mkdir -p logs
if pgrep -f "training[.](distill|sft)" >/dev/null; then echo "a trainer is running"; exit 1; fi

LOG="logs/profile_$(date +%Y%m%d_%H%M).log"
python -u -m training.distill \
  --student-variant mythouro_distill_tiny \
  --student-device xpu:0 --teacher-device xpu:0 \
  --teacher-id ByteDance/Ouro-2.6B-Thinking \
  --seq-len 1024 --micro-batch 8 --grad-accum 2 \
  --warmup-steps 500 --lr 1e-4 --min-lr 3e-5 \
  --depth-reg-coeff 0.3 --divergence rev_kl \
  --use-sandwich-norm --use-depth-aware-init \
  --teacher-mix-alpha 0.45 --rollout-len 64 --rollout-batch 32 --rollout-reuse 8 \
  --teacher-data-ratio 0.2 \
  --teacher-data-files 'data_teacher_code/shard_*.jsonl,data_teacher_math/shard_*.jsonl' \
  --onpolicy-lambda 0.7 \
  --ckpt-dir checkpoints_mathcode \
  --num-workers 0 --trust-remote-code \
  --profile-warmup 5 --profile-steps 20 \
  --total-steps 163238 2>&1 | tee "$LOG"

echo
echo "The teacher's share is the number that decides whether the top-K logit"
echo "cache is worth building. On record it was 54.2% at reuse=8."
