#!/usr/bin/env bash
# First leg trained on the BLENDED teacher corpus — general/math/code + medical.
#
#   bash run_medical_leg.sh            # 58,000 -> 64,000
#   TARGET=66000 bash run_medical_leg.sh
#
# Same recipe as run_ab_confirm.sh; the ONLY changed variable is the teacher
# corpus (medical added). Everything else is held fixed so the probe delta is
# attributable.
set -uo pipefail
# ── SIGNAL FORWARDING (2026-08-15) ────────────────────────────────────────────
# Without this the WRAPPER dies on Ctrl-C while the python child survives,
# orphaned, still holding the GPU. Observed in production on 2026-08-15: the user
# pressed Ctrl-C, reports/harvest_chat_FAILED was written at 22:26, and the
# harvest kept generating until it was signalled by PID. "Ctrl-C did nothing" was
# true from the terminal and false from the GPU. Reproduced in a toy harness:
# under the old pattern the child was still alive after a process-group SIGINT.
# The children here handle SIGINT COOPERATIVELY (flag, finish the step, flush),
# so they need the signal delivered and then time — not a dead parent.
trap 'pkill -INT -P $$ 2>/dev/null; true' INT TERM
cd "$(dirname "$0")"
source ../venv-xpu/bin/activate
export SYCL_CACHE_PERSISTENT=1 PYTORCH_ALLOC_CONF=expandable_segments:True TRITON_DEFAULT_BACKEND=intel

TARGET=${TARGET:-64000}
CKPT_DIR=checkpoints_onpolicy_fixed
OK="reports/medleg_${TARGET}_DONE"; FAIL="reports/medleg_${TARGET}_FAILED"
rm -f "$OK" "$FAIL"
trap '[ -f "$OK" ] || echo "incomplete exit=$? $(date)" > "$FAIL"' EXIT

# Both corpora, named explicitly — a data_teacher_*/ wildcard would pull in the
# retired v1 (boilerplate) and data_teacher_clean. The loader shuffles the
# combined shard list, so these interleave instead of being served as two blocks.
# Blend as of 2026-07-29: 8.65M general/math/code + 3.61M medical = 29.4% medical
# in the teacher stream, i.e. ~5.9% of training tokens at --teacher-data-ratio 0.2.
TEACHER_FILES='data_teacher_v2/shard_*.jsonl,data_teacher_med/shard_*.jsonl'

echo "=== [1/2] leg -> $TARGET  (teacher corpus now includes MEDICAL) ==="
python -u -m training.distill \
  --student-variant mythouro_distill_tiny \
  --student-device xpu:0 --teacher-device xpu:0 \
  --teacher-id ByteDance/Ouro-2.6B-Thinking \
  --seq-len 1024 --micro-batch 8 --grad-accum 2 \
  --total-steps "$TARGET" --warmup-steps 500 --lr 1e-4 --min-lr 3e-5 \
  --depth-reg-coeff 0.3 --divergence rev_kl \
  --use-sandwich-norm --use-depth-aware-init \
  --onpolicy-lambda 0.7 --teacher-mix-alpha 0.5 --rollout-len 64 \
  --rollout-batch 32 --rollout-reuse 2 \
  --teacher-data-ratio 0.2 --teacher-data-files "$TEACHER_FILES" \
  --ckpt-every-mins 15 --ckpt-milestone-every 2000 --keep-last 5 \
  --num-workers 0 --trust-remote-code --log-every 5 \
  --ckpt-dir "$CKPT_DIR" || true      # exit code NOT trusted — XPU teardown crash

# Verify by ARTIFACT, not exit code (torch/XPU can crash on teardown AFTER a
# successful run — that skipped a probe on 2026-07-28).
FINAL=$(ls -t "$CKPT_DIR"/step_*.pt | head -1)
STEP=$(basename "$FINAL" | sed 's/step_0*//; s/\.pt//')
if [ "$STEP" -lt "$TARGET" ]; then
  echo "=== stopped EARLY at $STEP (< $TARGET) — not probing ==="
  echo "early stop at $STEP $(date)" > "$FAIL"; exit 1
fi
echo "=== training reached $STEP ==="

echo "=== [2/2] probes ==="
python -u -m tools.onpolicy_rollout_probe \
  --ckpt-dir "$CKPT_DIR" --student-device xpu:0 --teacher-device xpu:0 \
  --teacher-id ByteDance/Ouro-2.6B-Thinking --trust-remote-code \
  --no-kv-cache --samples 5 \
  | tee "reports/onpolicy_rollout_probe_${STEP}_xpu_uncached_n5.txt" || true

python -u tools/collapse_metrics.py -c "$FINAL" --device xpu:0 --generate \
  --probe-set all \
  | tee "reports/collapse_metrics_${STEP}_xpu_aligned.txt" || true

echo "step=$STEP OK $(date)" > "$OK"
echo "=== DONE -> $OK ==="
