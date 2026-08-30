#!/usr/bin/env bash
# RUNG 3 — LOOP-WEIGHTED SUPERVISION. Distil against EVERY recurrent loop.
#
#   bash run_loopweighted.sh          # Ctrl-C whenever; resuming is free
#
# WHAT THIS TESTS, and why it is a genuine two-sided experiment rather than an
# improvement we expect. The literature CONTRADICTS ITSELF on exactly this:
#
#   supervise EVERY loop   recurrent-depth-ttc (github, MIT): iterative-target
#                          supervision extrapolates up to 24x beyond trained
#                          depth, and FINAL-ONLY SUPERVISION CAUSES ACCURACY
#                          WALLS. Ouro itself (arXiv 2510.25741, our teacher):
#                          L = SUM_t p(t|x)*L^(t). RLTT (2602.10520): +5.8% /
#                          +10.9% over terminal-only credit, measured ON OURO.
#
#   supervise ONLY the     "Thinking Deeper, Not Longer" (2603.21676): the
#   FINAL loop             Silent Thinking Objective computes loss at the final
#                          step "eliminating intermediate shortcuts", and calls
#                          rejecting intermediate supervision CRITICAL.
#
# MythOuro has been final-only since inception — not on theory, but because the
# ACT-weighted-sum output gave the optimiser a lever to pin lambda_0 ~ 1 and
# collapse depth. So the shortcut objection is not hypothetical here; we hit it.
#
# WHY IT MATTERS MORE THAN A LITERATURE SPAT. Our 2026-07-31 --n-loops sweep was
# FLAT at 4/6/8 (L4 0.0%, median rel_err 0.400 in all eight arms) and we
# concluded "depth is not a lever". recurrent-depth-ttc explains that exact shape
# differently: final-only supervision produces an accuracy WALL at the trained
# depth. If that transfers, depth is not dead — it is walled, the wall is our
# OBJECTIVE, and growing depth (rung 5) treats the symptom.
#
# UNIFORM, NOT exit_pdf, ON PURPOSE. --depth-reg-coeff 0.3 is a KL from the halt
# distribution to UNIFORM, so exit_pdf weighting would be actively pulled toward
# uniform anyway and the two arms would be confounded from step one. Uniform is
# the honest first arm. If it collapses depth, the successor is LoopFormer's
# shortcut-consistency (2602.11451): align SHORT trajectories TO THE LONGEST
# rather than supervising each independently, which cannot produce the shortcut
# Silent Thinking warns about because the long trajectory IS the target.
#
# MICRO-BATCH 4 / GRAD-ACCUM 4, not the pour's 8/2. Same effective batch (16),
# but the loop-weighted path holds K sets of logits live for the backward —
# ~805MB each at micro-batch 8, seq 1024, vocab 49,152 — alongside a 2.6B
# teacher. Halving the micro-batch buys that back. If it runs comfortably, raise
# it and take the throughput.
#
# TIMING, measured not guessed. The pour sustained ~5.0 s/step at full depth
# (from checkpoints_newmix timestamps). Loop-weighted costs more; if it lands
# near 2x, a 12h night buys ~4,700 steps. 140,000 is a DELIBERATE overshoot so
# the card is never idle — it is NOT tonight's target. The decision point is the
# depth sweep at the first milestone, not the step count.
#
# ⚠ NO WARMUP TRAP THIS TIME. At step 108,471 the LoopCurriculum has long since
# topped out at n_loops=4, so the rate in the first 100 steps IS the sustained
# rate. (On 2026-08-10 an ETA was called from a warmup-phase reading and missed
# by 3 hours because n_loops was still 2.) Check the first 100 steps before bed.
#
# ABORT CONDITIONS — worth knowing before you start:
#   * halt distribution / n_loops COLLAPSING toward loop 0 => the Silent Thinking
#     shortcut objection is live, and it is the lambda_0->1 failure we already hit
#     once. Kill it; checkpoints_newmix is untouched.
#   * OOM in the first minutes => expected failure mode. Halve --micro-batch to 2
#     and double --grad-accum to 8; effective batch is unchanged.
#   * `ce` sitting HIGHER than the old pour is EXPECTED, not a regression — you
#     are now paying loss at four depths instead of one.
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

SRC=checkpoints_newmix/step_0108471.pt
DIR=checkpoints_loopweighted
TEACHER=ByteDance/Ouro-2.6B-Thinking
FILES='data_teacher_v2/shard_*.jsonl,data_teacher_med/shard_*.jsonl'
TARGET=140000
WEIGHTING=uniform
LOG="logs/loopweighted_$(date +%Y%m%d_%H%M).log"
OK=reports/loopweighted_DONE; FAIL=reports/loopweighted_FAILED
mkdir -p logs reports "$DIR"
rm -f "$OK" "$FAIL"
trap '[ -f "$OK" ] || echo "incomplete exit=$? $(date)" > "$FAIL"' EXIT

step_of() { basename "$1" | sed 's/step_0*//; s/\.pt//'; }
latest()  { ls -t "$1"/step_*.pt 2>/dev/null | head -1; }

# Two trainers on one ckpt dir corrupts the run and the symptom shows up hours
# later. Refuse rather than race.
if pgrep -f "training[.](distill|sft|train_depth_policy)" >/dev/null; then
  echo "a trainer is already running; refusing to start a second"; exit 1
fi

# Seed the fresh dir from the base, so checkpoints_newmix stays the clean
# comparison line and step_0108471.pt remains the fallback.
[ -n "$(latest "$DIR")" ] || cp "$SRC" "$DIR/" || {
  echo "cannot seed $DIR from $SRC"; echo "seed failed $(date)" > "$FAIL"; exit 1; }

at=$(step_of "$(latest "$DIR")")
echo "=== RUNG 3: loop-weighted (--loop-loss-weighting $WEIGHTING) $at -> $TARGET ==="
echo "=== baseline to beat: code L3+ 75.0% | math L3+ 5.0% | depth sweep FLAT at 4/6/8 ==="
echo "=== log: $LOG   (Ctrl-C is safe; milestones every 2000) ==="

python -u -m training.distill \
  --student-variant mythouro_distill_tiny \
  --student-device xpu:0 --teacher-device xpu:0 --teacher-id "$TEACHER" \
  --seq-len 1024 --micro-batch 4 --grad-accum 4 \
  --warmup-steps 500 --lr 1e-4 --min-lr 3e-5 \
  --depth-reg-coeff 0.3 --divergence rev_kl \
  --use-sandwich-norm --use-depth-aware-init \
  --teacher-mix-alpha 0.5 --rollout-len 64 --rollout-batch 32 \
  --rollout-reuse 8 \
  --teacher-data-ratio 0.2 --teacher-data-files "$FILES" \
  --onpolicy-lambda 0.7 \
  --loop-loss-weighting "$WEIGHTING" \
  --ckpt-dir "$DIR" \
  --ckpt-every-mins 15 --ckpt-milestone-every 2000 --keep-last 5 \
  --num-workers 0 --trust-remote-code --log-every 50 \
  --total-steps "$TARGET" 2>&1 | tee "$LOG" || true   # exit code NOT trusted (XPU teardown)

at=$(step_of "$(latest "$DIR")")
echo "loopweighted reached $at $(date)" > "$OK"
echo "=== stopped at step $at ==="
cat <<EOF

THE READOUT THAT DECIDES THIS IS THE DEPTH SWEEP, NOT THE EVALS.

  # 1. Did the accuracy WALL lift? Compare against 2026-07-31, where 4/6/8 were
  #    flat at L4 0.0% / median rel_err 0.400 in ALL EIGHT arms.
  for K in 4 6 8; do
    python -u -m tools.math_eval -c $DIR/step_0*$at.pt --device xpu:0 \\
      --samples 8 --seed 1234 --n-loops \$K --force-full-depth \\
      --json reports/loopw_depth\${K}_$at.json
  done

  # 2. Did depth COLLAPSE instead? (the Silent Thinking failure mode)
  python -u -m tools.per_loop_calibration --checkpoint $DIR/step_0*$at.pt \\
    --device xpu:0 --max-samples 20 --seq-len 256 \\
    --out reports/per_loop_ece_$at.json

  # 3. Capability, same seed/settings as every prior baseline.
  python -u -m tools.code_eval -c $DIR/step_0*$at.pt --device xpu:0 \\
    --samples 8 --temperature 0.4 --seed 1234 --repetition-penalty 1.15 \\
    --json reports/code_loopw_$at.json

READ (1) FIRST. If 6 or 8 now beats 4, the flat sweep was our OBJECTIVE, not our
architecture — and that reopens depth, rung 5, and the whole test-time-compute
axis at once. If depth collapsed toward loop 0 in (2), the shortcut objection
won and LoopFormer's shortcut-consistency is the next design, not more uniform.
EOF
