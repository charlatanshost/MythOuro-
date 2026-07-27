#!/usr/bin/env bash
# Extend the confirming A/B leg, then probe the final checkpoint, then drop a
# marker file that a watcher pings on. One launch, walk away.
#
#   bash run_leg_and_probe.sh            # extends to 52,000 (default)
#   TARGET=54000 bash run_leg_and_probe.sh
#
# Resumes from the latest checkpoint in the ckpt dir (currently 46,000), same
# recipe as run_ab_confirm.sh + milestone checkpoints so nothing rotates away.
set -uo pipefail
cd "$(dirname "$0")"
source ../venv-xpu/bin/activate
export SYCL_CACHE_PERSISTENT=1 PYTORCH_ALLOC_CONF=expandable_segments:True TRITON_DEFAULT_BACKEND=intel

TARGET=${TARGET:-52000}
CKPT_DIR=checkpoints_onpolicy_fixed
TEACHER=ByteDance/Ouro-2.6B-Thinking
OK="reports/leg_${TARGET}_DONE"
FAIL="reports/leg_${TARGET}_FAILED"
rm -f "$OK" "$FAIL"
# Any exit path that DIDN'T reach the OK marker leaves a FAILED marker, so the
# watcher never hangs on a crash / OOM / Ctrl-C.
trap '[ -f "$OK" ] || echo "incomplete exit=$? $(date)" > "$FAIL"' EXIT

echo "=== [1/4] extend training -> $TARGET ==="
python -u -m training.distill \
  --student-variant mythouro_distill_tiny \
  --student-device xpu:0 --teacher-device xpu:0 \
  --teacher-id "$TEACHER" \
  --seq-len 1024 --micro-batch 8 --grad-accum 2 \
  --total-steps "$TARGET" --warmup-steps 500 --lr 1e-4 --min-lr 3e-5 \
  --depth-reg-coeff 0.3 --divergence rev_kl \
  --use-sandwich-norm --use-depth-aware-init \
  --onpolicy-lambda 0.7 --teacher-mix-alpha 0.5 --rollout-len 64 \
  --rollout-batch 32 --rollout-reuse 2 \
  --teacher-data-ratio 0.2 --teacher-data-files 'data_teacher_v2/shard_*.jsonl' \
  --ckpt-every-mins 15 --ckpt-milestone-every 2000 --keep-last 5 \
  --num-workers 0 --trust-remote-code --log-every 5 \
  --ckpt-dir "$CKPT_DIR" || exit 1

FINAL=$(ls -t "$CKPT_DIR"/step_*.pt | head -1)
STEP=$(basename "$FINAL" | sed 's/step_0*//; s/\.pt//')
echo "=== training done at step $STEP ($FINAL) ==="

echo "=== [2/5] rollout probe (α-ladder, uncached n=5) ==="
python -u -m tools.onpolicy_rollout_probe \
  --ckpt-dir "$CKPT_DIR" --student-device xpu:0 --teacher-device xpu:0 \
  --teacher-id "$TEACHER" --trust-remote-code --no-kv-cache --samples 5 \
  | tee "reports/onpolicy_rollout_probe_${STEP}_xpu_uncached_n5.txt" || exit 1

echo "=== [3/5] aligned sweep — greedy ==="
python -u tools/collapse_metrics.py -c "$FINAL" --device xpu:0 --generate \
  --probe-set all \
  | tee "reports/collapse_metrics_${STEP}_xpu_aligned.txt" || exit 1

echo "=== [4/5] aligned sweep — sampled T=0.8 ==="
python -u tools/collapse_metrics.py -c "$FINAL" --device xpu:0 --generate \
  --probe-set all --temperature 0.8 --top-k 40 \
  | tee "reports/collapse_metrics_${STEP}_xpu_aligned_t08.txt" || exit 1

# Leg + probe are the result — mark DONE now so the watcher pings and the probe
# gets read WHILE the harvest below runs. Everything past here is bonus.
echo "step=$STEP OK $(date)" > "$OK"
echo "=== LEG+PROBE DONE (pinged) -> $OK ==="

# [5/5] Don't leave the card idle during the workday: harvest corpus toward
# HARVEST_TARGET. Data-gen only — cannot over-train the model — and it grows the
# corpus for longer future legs (the "lean into data" verdict). Uses the fixed
# harvest (seed-mix + shuffle, cumulative target, resumes). STOP IT WHEN HOME
# (stop_gpu_jobs.sh); stopping is expected, not a failure — the DONE marker
# already fired. Bump HARVEST_TARGET if you want a bigger buffer against idle.
echo "=== [5/5] card would be idle — harvesting toward ${HARVEST_TARGET:-12000000} until stopped ==="
CORPUS_TARGET=${HARVEST_TARGET:-12000000} bash run_harvest_v2.sh
