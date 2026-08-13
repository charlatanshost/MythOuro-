#!/usr/bin/env bash
# λ SWEEP — is --onpolicy-lambda 0.7 still the right setting now that the collapse
# is cured, or can we trade some on-policy for throughput?
#
#   bash run_lambda_sweep.sh
#
# WHY THIS MATTERS MORE THAN IT LOOKS: we are token-starved, and offline steps run
# ~4.7x faster than on-policy ones (the 1.0 vs 4.7 ktok/s swing in the logs). λ sets
# the ratio, so it sets throughput:
#     λ=0.7 -> 91M tok/day (3B in ~31 days)      <- current
#     λ=0.4 -> 132M tok/day (3B in ~21 days)     <- 1.45x
# λ has only ever been run at 0.0 (mode-collapsed), 0.5 and 0.7 — never swept. 0.7
# was chosen when the goal was MAXIMUM dose to break the collapse; that is a
# different objective from throughput on a multi-billion-token run.
#
# DESIGN: both arms branch from the SAME step-64,000 checkpoint and run the SAME
# 2,000 steps, differing ONLY in --onpolicy-lambda. Separate ckpt dirs keep them
# independent. Each arm is probed with the identical instrument.
#
# Arm A  λ=0.7 CONTROL  64,000 -> 66,000  in checkpoints_lambda07  (~5.2 h from 64,206)
# Arm B  λ=0.4 TEST     64,000 -> 66,000  in checkpoints_lambda04  (~4.0 h — faster by design)
#
# Safe to stop any time (Ctrl-C): checkpoints flush, and re-running resumes each arm
# from where it got to. Nothing is lost, nothing is double-counted.
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
cd "$(dirname "$0")"
source ../venv-xpu/bin/activate
export SYCL_CACHE_PERSISTENT=1 PYTORCH_ALLOC_CONF=expandable_segments:True TRITON_DEFAULT_BACKEND=intel

MAIN=checkpoints_onpolicy_fixed
SEED_CKPT="$MAIN/step_0064000.pt"
TEACHER=ByteDance/Ouro-2.6B-Thinking
FILES='data_teacher_v2/shard_*.jsonl,data_teacher_med/shard_*.jsonl'
TARGET=66000
OK=reports/lambda_sweep_DONE; FAIL=reports/lambda_sweep_FAILED
rm -f "$OK" "$FAIL"
trap '[ -f "$OK" ] || echo "incomplete exit=$? $(date)" > "$FAIL"' EXIT

COMMON=(
  --student-variant mythouro_distill_tiny
  --student-device xpu:0 --teacher-device xpu:0 --teacher-id "$TEACHER"
  --seq-len 1024 --micro-batch 8 --grad-accum 2
  --warmup-steps 500 --lr 1e-4 --min-lr 3e-5
  --depth-reg-coeff 0.3 --divergence rev_kl
  --use-sandwich-norm --use-depth-aware-init
  --teacher-mix-alpha 0.5 --rollout-len 64 --rollout-batch 32 --rollout-reuse 2
  --teacher-data-ratio 0.2 --teacher-data-files "$FILES"
  --ckpt-every-mins 15 --ckpt-milestone-every 2000 --keep-last 5
  --num-workers 0 --trust-remote-code --log-every 5 --total-steps "$TARGET"
)

step_of() { basename "$1" | sed 's/step_0*//; s/\.pt//'; }
latest()  { ls -t "$1"/step_*.pt 2>/dev/null | head -1; }

run_arm() {                            # $1=dir  $2=lambda  $3=label
  local dir=$1 lam=$2 label=$3
  mkdir -p "$dir"
  # Seed from the shared branch point if this arm hasn't started.
  [ -n "$(latest "$dir")" ] || cp "$SEED_CKPT" "$dir/" || {
    echo "cannot seed $dir from $SEED_CKPT"; echo "seed failed $(date)" > "$FAIL"; exit 1; }

  local at; at=$(step_of "$(latest "$dir")")
  if [ "$at" -ge "$TARGET" ]; then
    echo "=== $label already at $at — skipping training ==="
  else
    echo "=== $label: λ=$lam, $at -> $TARGET ==="
    python -u -m training.distill "${COMMON[@]}" \
      --onpolicy-lambda "$lam" --ckpt-dir "$dir" || true   # exit code NOT trusted
    at=$(step_of "$(latest "$dir")")
  fi

  if [ "$at" -lt "$TARGET" ]; then
    echo "=== $label stopped early at $at (< $TARGET) — not probing ==="
    echo "$label early stop at $at $(date)" > "$FAIL"; exit 1
  fi

  local out="reports/onpolicy_rollout_probe_${at}_lambda${lam/./}_n5.txt"
  echo "=== $label: probing $at -> $out ==="
  python -u -m tools.onpolicy_rollout_probe --ckpt-dir "$dir" \
    --student-device xpu:0 --teacher-device xpu:0 --teacher-id "$TEACHER" \
    --trust-remote-code --no-kv-cache --samples 5 | tee "$out" || true
  [ -s "$out" ] && [ "$(wc -l < "$out")" -ge 20 ] || {
    echo "$label probe produced nothing"; echo "$label probe failed $(date)" > "$FAIL"; exit 1; }
}

run_arm checkpoints_lambda07 0.7 "ARM A (control)"
run_arm checkpoints_lambda04 0.4 "ARM B (test)"

echo "sweep OK $(date)" > "$OK"
echo "=== λ SWEEP DONE -> $OK ==="
echo "Compare: reports/onpolicy_rollout_probe_66000_lambda07_n5.txt"
echo "     vs: reports/onpolicy_rollout_probe_66000_lambda04_n5.txt"
