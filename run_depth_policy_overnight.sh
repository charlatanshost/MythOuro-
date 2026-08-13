#!/usr/bin/env bash
# DEPTH-POLICY OVERNIGHT — teach ACTHalting + UncertaintyHead from a best-exit target.
#
#   bash run_depth_policy_overnight.sh        # Ctrl-C in the morning; resuming is free
#
# WHAT THIS TRAINS, and why it is the next step. Two separate heads both consume
# one broken signal, and both were measured settling shallow while answers stay
# wrong:
#   ACTHalting     -> p = act(h); cumulative_p >= act_threshold decides the EARLY
#                     EXIT. Fired at a constant depth of 2.00 across all 24
#                     measured (domain x alpha) cells, 120 samples, zero variance.
#   UncertaintyHead-> ranks loops for best-of-trajectory. Picked loop 1 or 2 for
#                     98% of 3,161 tokens and drove the prompt-copy rate UP
#                     41.2% -> 50.0%: argmin-over-uncertainty is echo-seeking,
#                     because copying a token just seen is the most confident
#                     prediction available.
# Both now train from the SAME label: the loop with the lowest per-loop CE, taken
# from the model's own forced-depth trajectory (the TEACHER, per docs/ideas.md).
#
# PRECONDITION, MEASURED @100,000 (reports/best_exit_100000.json):
#   best exits spread across ALL 8 depths          -> a policy exists to learn
#   headroom vs trained depth 35.6% (0.093 nats)   -> CE 0.261 -> oracle 0.168
#   head agreement 25.3% (chance 12.5%)            -> it captures a quarter of it
#   head's own selection CE 0.550                  -> WORSE than fixed depth 3
# The bar to beat is FIXED-DEPTH-3 (0.261), not the broken head (0.550).
#
# OVERNIGHT SAFEGUARDS — this trainer has never run on the real card:
#   1. SMOKE PHASE first: 30 steps into a throwaway dir. If the real thing is
#      going to crash on XPU it crashes in minutes, not at hour six, and the main
#      run never starts. Costs ~2 minutes.
#   2. --resume: restarts from the newest checkpoint in the out-dir, so a crash
#      at step 1,800 does not throw the night away.
#   3. --log-file: loguru writes to stderr only. Twice now a result has been lost
#      to scrollback; this puts it on disk.
#   4. Atomic saves (write-then-rename inside the trainer): a crash mid-save
#      cannot leave a truncated .pt that globs fine and fails to load.
#   5. Refuses to start if another trainer is alive — two processes on one ckpt
#      dir corrupts the run and the symptom appears hours later.
#   6. Verifies by ARTIFACT, not exit code (XPU teardown raises after success).
#
# STEPS deliberately overshoots (6,000, saving every 500). A head-only run may be
# far faster than a distillation step, and a job that FINISHES early leaves the
# card idle — the real constraint on a dual-boot rig. Overtraining is recoverable:
# there are 12 checkpoints to choose from in the morning.
#
# The LM body is frozen throughout — only ~411k of 216M params train (0.19%).
# The source checkpoint is never modified.
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

SRC=checkpoints_newmix/step_0100000.pt
DIR=checkpoints_depthpolicy
SMOKE=/tmp/claude-1000/depthpolicy_smoke
LOG=logs/depth_policy_$(date +%Y%m%d_%H%M).log
OK=reports/depth_policy_DONE; FAIL=reports/depth_policy_FAILED
mkdir -p logs reports
rm -f "$OK" "$FAIL"
trap '[ -f "$OK" ] || echo "incomplete exit=$? $(date)" > "$FAIL"' EXIT

if pgrep -f "training[.](distill|train_depth_policy)" >/dev/null; then
  echo "a trainer is already running; refusing to start a second"; exit 1
fi
[ -f "$SRC" ] || { echo "source checkpoint missing: $SRC"; \
                   echo "no source $(date)" > "$FAIL"; exit 1; }

# ── 1. SMOKE: prove it runs on the real card before committing the night ──
echo "=== SMOKE: 30 steps into a throwaway dir ==="
rm -rf "$SMOKE"; mkdir -p "$SMOKE"
python -u -m training.train_depth_policy \
  -c "$SRC" -o "$SMOKE" --device xpu:0 \
  --steps 30 --save-every 30 --log-every 10 --n-loops 8 2>&1 | tail -12
if ! ls "$SMOKE"/step_*.pt >/dev/null 2>&1; then
  echo "SMOKE FAILED — no checkpoint written; not starting the overnight run"
  echo "smoke failed $(date)" > "$FAIL"; exit 1
fi
echo "=== SMOKE OK ($(ls "$SMOKE"/step_*.pt | wc -l) ckpt) — starting the real run ==="
rm -rf "$SMOKE"

# ── 2. THE RUN ──
python -u -m training.train_depth_policy \
  -c "$SRC" -o "$DIR" --device xpu:0 \
  --steps 6000 --save-every 500 --log-every 25 \
  --n-loops 8 --micro-batch 2 --seq-len 512 \
  --lr 3e-4 --act-weight 1.0 \
  --resume --log-file "$LOG" 2>&1 | tail -40 || true   # exit code NOT trusted

n=$(ls "$DIR"/step_*.pt 2>/dev/null | wc -l)
[ "$n" -gt 0 ] || { echo "no checkpoints produced"; \
                    echo "no output $(date)" > "$FAIL"; exit 1; }
echo "depth policy OK — $n checkpoints $(date)" > "$OK"

cat <<EOF

=== DONE — $n checkpoints in $DIR, log at $LOG ===

IN THE MORNING, on the newest checkpoint:

  D=\$(ls -t $DIR/step_*.pt | head -1)

  # did the head learn the target? agreement should climb off 25.3%
  python -u -m tools.best_exit_probe -c "\$D" --device xpu:0 --n-loops 8 \\
      --chunks 12 --seq-len 256 --json reports/best_exit_after_policy.json

  # does better exit selection produce better ANSWERS? (it may not — that is
  # the finding either way)
  python -u -m tools.math_eval -c "\$D" --device xpu:0 --samples 8 --seed 1234 \\
      --json reports/math_eval_after_policy.json

THE BAR: fixed-depth-3 at CE 0.261, NOT the broken head at 0.550. Oracle 0.168.
Also check "ACTUAL halt depth" — if it is still 2.00 the ACT branch did not move.
EOF
