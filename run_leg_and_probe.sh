#!/usr/bin/env bash
# Extend the confirming A/B leg, then probe the final checkpoint, then drop a
# marker file that a watcher pings on. One launch, walk away.
#
#   bash run_leg_and_probe.sh            # extends to 52,000 (default)
#   TARGET=54000 bash run_leg_and_probe.sh
#
# Resumes from the latest checkpoint in the ckpt dir (currently 46,000), same
# recipe as run_ab_confirm.sh + milestone checkpoints so nothing rotates away.
#
# STOPPING (corrected 2026-08-13). "Ctrl-C is safe" is true but INCOMPLETE, and
# the gap has cost hours twice:
#   * The handler is COOPERATIVE — it sets a flag, the loop notices at the next
#     safe point, then writes a ~3.3GB checkpoint. Nothing appears for 30-60s.
#     The terminal looks frozen. It is not.
#   * Do NOT press Ctrl-C twice. A second signal forces KeyboardInterrupt and
#     SKIPS the graceful save.
#   * The XPU/SYCL runtime frequently DEADLOCKS IN TEARDOWN after the work is
#     already done and the checkpoint is written (max1100_field_notes.md). If the
#     process lingers with the GPU idle, it has finished: `kill -9 <pid>`, then
#     confirm memory returned with `xpu-smi dump -d 0 -m 18 -n 1`. A half-dead
#     process holds its allocation and the NEXT job OOMs on a "free" card.
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
  --ckpt-dir "$CKPT_DIR" || true      # exit code is NOT trusted — see below

# Verify by ARTIFACT, not exit code. torch/XPU can crash during interpreter
# teardown ("Fatal Python error: PyGILState_Release ... finalizing") AFTER the
# run has completed and saved — exit 1 on a fully successful leg. That happened
# 2026-07-28: step_0058000.pt was written at 15:54:21, the chain died 15:54:22,
# and the probe never ran. So: did we reach the target checkpoint?
FINAL=$(ls -t "$CKPT_DIR"/step_*.pt | head -1)
STEP=$(basename "$FINAL" | sed 's/step_0*//; s/\.pt//')
if [ "$STEP" -lt "$TARGET" ]; then
  echo "=== training stopped EARLY at $STEP (< $TARGET) — not probing ==="
  echo "stopped early at $STEP (target $TARGET) $(date)" > "$FAIL"
  exit 1
fi
echo "=== training done at step $STEP ($FINAL) — target reached ==="

# Each probe is likewise checked by ARTIFACT (did it write a plausible report?),
# not by exit code — `pipefail` + a teardown crash would otherwise abort the chain
# after the work was already done and written.
check_report() {                      # $1 = path, $2 = min lines expected
  if [ ! -s "$1" ] || [ "$(wc -l < "$1")" -lt "$2" ]; then
    echo "=== probe produced no usable output: $1 — stopping ==="
    echo "probe failed: $1 $(date)" > "$FAIL"; exit 1
  fi
}

echo "=== [2/5] rollout probe (α-ladder, uncached n=5) ==="
python -u -m tools.onpolicy_rollout_probe \
  --ckpt-dir "$CKPT_DIR" --student-device xpu:0 --teacher-device xpu:0 \
  --teacher-id "$TEACHER" --trust-remote-code --no-kv-cache --samples 5 \
  | tee "reports/onpolicy_rollout_probe_${STEP}_xpu_uncached_n5.txt" || true
check_report "reports/onpolicy_rollout_probe_${STEP}_xpu_uncached_n5.txt" 20

echo "=== [3/5] aligned sweep — greedy ==="
python -u tools/collapse_metrics.py -c "$FINAL" --device xpu:0 --generate \
  --probe-set all \
  | tee "reports/collapse_metrics_${STEP}_xpu_aligned.txt" || true
check_report "reports/collapse_metrics_${STEP}_xpu_aligned.txt" 20

echo "=== [4/5] aligned sweep — sampled T=0.8 ==="
python -u tools/collapse_metrics.py -c "$FINAL" --device xpu:0 --generate \
  --probe-set all --temperature 0.8 --top-k 40 \
  | tee "reports/collapse_metrics_${STEP}_xpu_aligned_t08.txt" || true
check_report "reports/collapse_metrics_${STEP}_xpu_aligned_t08.txt" 20

# Leg + probe are the result — mark DONE now so the watcher pings and the probe
# gets read WHILE the harvest below runs. Everything past here is bonus.
echo "step=$STEP OK $(date)" > "$OK"
echo "=== LEG+PROBE DONE (pinged) -> $OK ==="

# [5/5] Don't leave the card idle during the workday: harvest corpus toward
# HARVEST_TARGET. Data-gen only — cannot over-train the model — and it grows the
# corpus for longer future legs (the "lean into data" verdict). Uses the fixed
# harvest (seed-mix + shuffle, cumulative target, resumes).
#
# Target is set HIGH on purpose (16M, vs the 12M plan). Harvest is slow (~93
# tok/s ≈ 2.7M/workday) but training is fast, so the leg may finish early and
# leave a LONG idle window — a 12M target would self-complete (~11h) and then
# idle. 16M can't finish before you're home (~22h from 8.65M), so it runs the
# whole window. STOP IT WHEN HOME (stop_gpu_jobs.sh) wherever it's got to;
# stopping is expected, not a failure (the DONE marker already fired), and any
# corpus past 12M is pure upside (longer legs, less repetition).
echo "=== [5/5] card would be idle — harvesting toward ${HARVEST_TARGET:-16000000} (stop when home) ==="
CORPUS_TARGET=${HARVEST_TARGET:-16000000} bash run_harvest_v2.sh
