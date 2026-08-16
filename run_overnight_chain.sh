#!/usr/bin/env bash
# OVERNIGHT CHAIN — queue behind run_reuse8_ab.sh so the card never idles.
#
#   bash run_overnight_chain.sh        # launch NOW; it waits, then continues
#
# THE PROBLEM THIS SOLVES. On a dual-boot rig the binding constraint is
# GPU-HOURS, not tokens (docs/training_throughput.md). The A/B finishes ~01:45
# and its probes ~02:05, after which the card is idle until morning — ~5 hours,
# ~3,700 steps, ~61M tokens of pure waste. The fix is not a faster kernel, it is
# never leaving the card with nothing to do.
#
# WHY IT IS SAFE TO CONTINUE BEFORE READING THE PROBE. The quality verdict on
# reuse=8 lands in the morning, but continuing costs nothing we already hold:
#   * the main line `checkpoints_onpolicy_fixed` is NEVER touched;
#   * `checkpoints_lambda07/step_0066000.pt` (reuse=2 control) is the fallback;
#   * step 66,000 in this arm is a milestone save, so the A/B's own comparison
#     point survives regardless of how far training then runs past it.
# If the probe says reuse=8 hurt quality, discard the branch and fall back —
# losing hours that would otherwise have been idle anyway. Upside is 5 hours of
# 2.3x training; downside is bounded. Do NOT promote this arm to the main line
# until the 66,000 probe has actually been read.
#
# TARGET is deliberately far away (72,000 ≈ 8h at 4.81 s/step). A chain that
# FINISHES early leaves the card idle, which is the failure mode being fixed —
# so overshoot and Ctrl-C in the morning. Stopping is free: checkpoints flush
# and re-running resumes.
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

DIR=checkpoints_reuse8
TEACHER=ByteDance/Ouro-2.6B-Thinking
FILES='data_teacher_v2/shard_*.jsonl,data_teacher_med/shard_*.jsonl'
TARGET=72000
AB_OK=reports/reuse8_ab_DONE; AB_FAIL=reports/reuse8_ab_FAILED
OK=reports/overnight_chain_DONE; FAIL=reports/overnight_chain_FAILED
rm -f "$OK" "$FAIL"
trap '[ -f "$OK" ] || echo "incomplete exit=$? $(date)" > "$FAIL"' EXIT

step_of() { basename "$1" | sed 's/step_0*//; s/\.pt//'; }
latest()  { ls -t "$1"/step_*.pt 2>/dev/null | head -1; }

echo "=== waiting for the A/B to finish (checking every 60s) ==="
while [ ! -f "$AB_OK" ] && [ ! -f "$AB_FAIL" ]; do sleep 60; done
if [ -f "$AB_FAIL" ]; then
  echo "=== A/B FAILED — not continuing ==="; cat "$AB_FAIL"
  echo "A/B failed, chain skipped $(date)" > "$FAIL"; exit 1
fi
echo "=== A/B done: $(cat "$AB_OK") ==="

# Belt-and-braces: never run two trainers into the same ckpt dir.
while pgrep -f "training[.]distill" >/dev/null; do
  echo "  a trainer is still alive; waiting 60s"; sleep 60
done

at=$(step_of "$(latest "$DIR")")
echo "=== continuing reuse=8: $at -> $TARGET (Ctrl-C in the morning) ==="
python -u -m training.distill \
  --student-variant mythouro_distill_tiny \
  --student-device xpu:0 --teacher-device xpu:0 --teacher-id "$TEACHER" \
  --seq-len 1024 --micro-batch 8 --grad-accum 2 \
  --warmup-steps 500 --lr 1e-4 --min-lr 3e-5 \
  --depth-reg-coeff 0.3 --divergence rev_kl \
  --use-sandwich-norm --use-depth-aware-init \
  --teacher-mix-alpha 0.5 --rollout-len 64 --rollout-batch 32 \
  --rollout-reuse 8 \
  --teacher-data-ratio 0.2 --teacher-data-files "$FILES" \
  --onpolicy-lambda 0.7 --ckpt-dir "$DIR" \
  --ckpt-every-mins 15 --ckpt-milestone-every 2000 --keep-last 5 \
  --num-workers 0 --trust-remote-code --log-every 20 \
  --total-steps "$TARGET" || true            # exit code NOT trusted

at=$(step_of "$(latest "$DIR")")
echo "chain reached $at $(date)" > "$OK"
echo "=== stopped at step $at ==="
echo "MORNING: read reports/onpolicy_rollout_probe_66000_reuse8_n5.txt"
echo "     vs reports/onpolicy_rollout_probe_66000_lambda07_n5.txt  (reuse=2 control)"
echo "     BEFORE promoting this arm to the main line."
