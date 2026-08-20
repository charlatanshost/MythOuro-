#!/usr/bin/env bash
# α-ANNEAL 0.5 → 0.45 — reduce teacher help, the second time in the project.
#
#   bash run_anneal.sh          # overnight; Ctrl-C ONCE and WAIT (see STOPPING)
#
# THE TRIGGER, and it is the project's own documented one. On 2026-06-30 the rule
# for annealing α was: capability clearly PRESENT at high α but NOT internalized
# into α=0.0, with α=0.0 FLAT. The reasoning — fixed high α keeps most of every
# rollout teacher-driven, so the student rarely recovers from its OWN errors and
# the exposure-bias gap cannot close by grinding tokens.
#
# α=0.0 (pure student) top_share, measured across the whole pour:
#     90,351  0.161  |  100,000  0.201  |  108,471  0.170
#    125,181  0.155  |  140,000  0.154
# FLAT across 50,000 steps. For scale, the 0.6→0.5 anneal moved that same metric
# 0.18 → 0.12 in 216 STEPS. The trigger has been firing for a long time.
#
# WHY 0.45 AND NOT LOWER. `onpolicy_plan.md` already named 0.45 as the next rung
# and shelved it as "optional/secondary" under the decision "HOLD 0.5, pour TOKENS
# (the Max) to push fluency→meaning". Those tokens are poured — the pour reached
# its 140,000 target — so the condition the hold depended on is satisfied. Small
# step, same as last time (0.6→0.5), because the last one was validated as SAFE at
# that size and nothing argues for a bigger jump.
#
# ONE VARIABLE. Every other flag is byte-identical to the pour. The only second-
# order change is the LR schedule: cosine is tied to --total-steps, so raising the
# target from 140,000 to 146,000 lifts LR at step 140,000 from exactly 3.000e-5 to
# ~3.0003e-5 — a 0.01% difference against the floor, i.e. nothing.
#
# FRESH DIR ON PURPOSE. checkpoints_newmix is the COMPLETED α=0.5 line and the
# control for this experiment; step_0140000.pt is its endpoint and must stay
# reachable. This seeds checkpoints_anneal045 instead. (Also: the trainer's own
# --keep-last 5 deleted step_0108471.pt during the pour — non-milestone
# checkpoints are NOT safe, whatever prune_checkpoints._PROTECTED claims. 140,000
# is an even milestone so it survives, and checkpoints_base/ holds copies.)
#
# ⚠ WHAT TO EXPECT, from the 2026-06-30/07-01 precedent:
#   * LOSS WILL PROBABLY RISE. Documented as EXPECTED AND GOOD, not a regression —
#     the student is being made to carry more of each rollout itself.
#   * WATCH FRAGILE SEEDS FOR RE-COLLAPSE. That was the stated risk last time and
#     it did NOT happen: the fragile bacterial/LaTeX seed actually DE-fragilized
#     (α=0.0 top_share 0.47 → 0.18). A re-collapse here is the abort signal.
#   * MILESTONES EVERY 500. The last anneal's verdict was read at ~216 steps, so
#     the early points are the informative ones. Do not wait for 6,000.
#
# STOPPING (max1100_field_notes.md):
#   * Ctrl-C ONCE, then WAIT — cooperative handler, writes a ~3.3GB checkpoint.
#     30-60s of an apparently frozen terminal. Do NOT press it twice.
#   * XPU/SYCL often deadlocks in TEARDOWN after the checkpoint is written. If it
#     lingers with the GPU idle it has finished: kill -9, then confirm memory
#     returned with `xpu-smi dump -d 0 -m 18 -n 1`.
set -uo pipefail
trap 'pkill -INT -P $$ 2>/dev/null; true' INT TERM
cd "$(dirname "$0")"
source ../venv-xpu/bin/activate
export SYCL_CACHE_PERSISTENT=1 PYTORCH_ALLOC_CONF=expandable_segments:True TRITON_DEFAULT_BACKEND=intel

SRC=checkpoints_newmix/step_0140000.pt
DIR=checkpoints_anneal045
TEACHER=ByteDance/Ouro-2.6B-Thinking
FILES='data_teacher_v2/shard_*.jsonl,data_teacher_med/shard_*.jsonl'
ALPHA=0.45                       # THE ONE VARIABLE. the pour ran 0.5.
TARGET=143500
LOG="logs/anneal045_$(date +%Y%m%d_%H%M).log"
OK=reports/anneal045_DONE; FAIL=reports/anneal045_FAILED
mkdir -p logs reports "$DIR"
rm -f "$OK" "$FAIL"
trap '[ -f "$OK" ] || echo "incomplete exit=$? $(date)" > "$FAIL"' EXIT

step_of() { basename "$1" | sed 's/step_0*//; s/\.pt//'; }
latest()  { ls -t "$1"/step_*.pt 2>/dev/null | head -1; }

if pgrep -f "training[.](distill|sft|train_depth_policy)" >/dev/null; then
  echo "a trainer is already running; refusing to start a second"; exit 1
fi
if pgrep -f "tools[.](gen_teacher_corpus|code_eval|best_exit_probe|onpolicy_rollout_probe)" >/dev/null; then
  echo "a probe is running — it needs the card; let it finish"; exit 1
fi
[ -f "$SRC" ] || { echo "base $SRC missing"; echo "no base $(date)" > "$FAIL"; exit 1; }

[ -n "$(latest "$DIR")" ] || cp "$SRC" "$DIR/" || {
  echo "cannot seed $DIR from $SRC"; echo "seed failed $(date)" > "$FAIL"; exit 1; }

at=$(step_of "$(latest "$DIR")")
if [ "$at" -ge "$TARGET" ]; then
  echo "already at $at >= target $TARGET — raise TARGET to continue"; exit 1
fi
echo "=== α-ANNEAL 0.5 -> $ALPHA :  $at -> $TARGET ==="
echo "=== control: checkpoints_newmix (same recipe at α=0.5, completed at 140,000) ==="
echo "=== α=0.0 baseline to beat: top_share 0.154, distinct1 0.477 @140,000 ==="
echo "=== log: $LOG ==="

python -u -m training.distill \
  --student-variant mythouro_distill_tiny \
  --student-device xpu:0 --teacher-device xpu:0 --teacher-id "$TEACHER" \
  --seq-len 1024 --micro-batch 8 --grad-accum 2 \
  --warmup-steps 500 --lr 1e-4 --min-lr 3e-5 \
  --depth-reg-coeff 0.3 --divergence rev_kl \
  --use-sandwich-norm --use-depth-aware-init \
  --teacher-mix-alpha "$ALPHA" --rollout-len 64 --rollout-batch 32 \
  --rollout-reuse 8 \
  --teacher-data-ratio 0.2 --teacher-data-files "$FILES" \
  --onpolicy-lambda 0.7 \
  --ckpt-dir "$DIR" \
  --ckpt-every-mins 15 --ckpt-milestone-every 500 --keep-last 5 \
  --num-workers 0 --trust-remote-code --log-every 50 \
  --total-steps "$TARGET" 2>&1 | tee "$LOG" || true   # exit code NOT trusted (XPU teardown)

at=$(step_of "$(latest "$DIR")")
echo "anneal045 reached $at $(date)" > "$OK"
echo "=== stopped at step $at ==="
cat <<EOF

THE READOUT IS α=0.0, NOT THE LOSS. Loss rising is expected here.

  python -u -m tools.onpolicy_rollout_probe --ckpt-dir $DIR \\
    --student-device xpu:0 --teacher-device xpu:0 --teacher-id $TEACHER \\
    --trust-remote-code --no-kv-cache --samples 5 \\
    --json reports/probe_anneal045_${at}.json | tee reports/probe_anneal045_${at}.txt

READ IT AGAINST @140,000: α=0.0 top_share 0.154 | distinct1 0.477 | halt 2.00/4.
  top_share DOWN and no seed re-collapsing  -> the anneal is working, same as
      0.6->0.5 did (0.18 -> 0.12, 4/6 seeds improved, fragile seed de-fragilized).
  any seed spiking back toward 0.4+         -> re-collapse. That is the abort
      signal; checkpoints_newmix is untouched and step_0140000.pt is the fallback.
  FLAT again                                -> α is not the lever at this scale
      and the exposure-bias gap needs a different attack (rung 6, on-policy SFT).
EOF
