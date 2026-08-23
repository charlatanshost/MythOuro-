#!/usr/bin/env bash
# CODE-MIX LEG — the first training data that demonstrates FINISHING a task.
#
#   bash run_codemix.sh          # ~8.5h; Ctrl-C ONCE and WAIT (see STOPPING)
#
# THE PROBLEM THIS TARGETS. Form is fixed; content has never moved. Strict L4 has
# been 0-2/320 since step 64,000 and median `body_stmts` is 1 at EVERY α and EVERY
# checkpoint ever measured. The model writes one plausible statement and stops,
# because nothing in its training signal ever rewarded finishing a task — it
# trained almost entirely on CONTINUATION data (codeparrot files, teacher
# continuations of web text). α, λ, loop-weighting and dose all reshape the
# sampling distribution; none of them add task competence.
#
# WHY NOT data_teacher_chat. That corpus is passage EXPLANATION — its templates
# are "Explain the following passage in your own words" and only ~17% of rows
# contain code at all. It teaches passage QA, which is not the gap.
#
# WHY THIS IS DIFFERENT FROM THE TWO CHAT-MIX LEGS THAT FAILED:
#   * the DATA is task→complete-solution, pre-verified by unit tests
#     (median 4 statements, 79% have >=3, only 1% one-liners — against a model
#     stuck at median 1). Inspected 2026-08-22 on 2,000 streamed rows.
#   * the BASE is stable now. 0/80 char-degenerate, acronym salad gone, restart
#     rate a quarter of what it was at 64,000. The failed legs ran on a degenerate
#     base — that was the owner's argument for annealing first, and it held.
#   * the CHANNEL is distillation, not SFT. SFT has collapsed this model at every
#     dose and every batch size; teacher-data through distillation is what took
#     code L3+ 51.2% → 75.0%.
#
# ⚠ DOSE. ~100,000 rows ≈ 35M tokens. At 16 seq x 1024 tok x ratio 0.2 = 3,277
# teacher tokens/step, one epoch is ~10,700 steps, so this 6,000-step leg is
# ~0.56 EPOCHS. The 2026-08-15 chat leg collapsed at 10.3 epochs and recovered at
# 1.35; this is comfortably below both. Do NOT raise --teacher-data-ratio to
# "get more signal" — that is the knob that sank the first attempt.
#
# α STAYS 0.45. The base was trained at 0.45 and n=320 validated it (+18pp over
# α=0.5). The DATA is the one variable in this leg.
#
# GATE IT ON `body_stmts` AND `committed`, NOT L3+. L3+ climbed 27 points during
# the α ladder while strict L4 never left 0-2/320. If median body_stmts moves off
# 1 for the first time in project history, that is the result.
#
# STOPPING (max1100_field_notes.md):
#   * Ctrl-C ONCE, then WAIT — cooperative handler, writes a ~3.3GB checkpoint.
#     30-60s of an apparently frozen terminal. Do NOT press twice.
#   * XPU/SYCL often deadlocks in TEARDOWN after the checkpoint is written. If it
#     lingers with the GPU idle it has finished: kill -9, then confirm memory
#     returned with `xpu-smi dump -d 0 -m 18 -n 1`.
set -uo pipefail
trap 'pkill -INT -P $$ 2>/dev/null; true' INT TERM
cd "$(dirname "$0")"
source ../venv-xpu/bin/activate
export SYCL_CACHE_PERSISTENT=1 PYTORCH_ALLOC_CONF=expandable_segments:True TRITON_DEFAULT_BACKEND=intel

SRC=checkpoints_anneal045/step_0143500.pt
DIR=checkpoints_codemix
TEACHER=ByteDance/Ouro-2.6B-Thinking
FILES='data_teacher_code/shard_*.jsonl'
ALPHA=0.45
STEPS=6000
LOG="logs/codemix_$(date +%Y%m%d_%H%M).log"
OK=reports/codemix_DONE; FAIL=reports/codemix_FAILED
mkdir -p logs reports "$DIR"
rm -f "$OK" "$FAIL"
trap '[ -f "$OK" ] || echo "incomplete exit=$? $(date)" > "$FAIL"' EXIT

step_of() { basename "$1" | sed 's/step_0*//; s/\.pt//'; }
latest()  { ls -t "$1"/step_*.pt 2>/dev/null | head -1; }

if pgrep -f "training[.](distill|sft|train_depth_policy)" >/dev/null; then
  echo "a trainer is already running; refusing to start a second"; exit 1; fi
if pgrep -f "tools[.](gen_teacher_corpus|make_code_corpus|code_eval)" >/dev/null; then
  echo "a corpus build or probe is running; let it finish"; exit 1; fi

# Fail before the teacher loads if the corpus is missing or short. A truncated
# corpus is what turned the 2026-08-15 leg into 10 epochs of memorisation.
rows=$(cat data_teacher_code/shard_*.jsonl 2>/dev/null | wc -l)
if [ "${rows:-0}" -lt 50000 ]; then
  echo "data_teacher_code has only ${rows:-0} rows — expected ~100,000."
  echo "Run: python -m tools.make_code_corpus --target-rows 100000 --out-dir data_teacher_code"
  echo "refusing to start on a short corpus"; exit 1
fi
echo "=== code corpus: $rows rows ==="

[ -f "$SRC" ] || { echo "base $SRC missing"; echo "no base $(date)" > "$FAIL"; exit 1; }
[ -n "$(latest "$DIR")" ] || cp "$SRC" "$DIR/" || {
  echo "cannot seed $DIR from $SRC"; echo "seed failed $(date)" > "$FAIL"; exit 1; }

at=$(step_of "$(latest "$DIR")")
TARGET=$((at + STEPS))
echo "=== CODE-MIX: $at -> $TARGET  (α=$ALPHA, task-completion data @0.2) ==="
echo "=== base: step_0143500 — L3+ 76.6 ±4.6 | committed 25% | body_stmts 1 ==="
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
echo "codemix reached $at $(date)" > "$OK"
cat <<EOF

=== READ THIS AT --samples 32, NOT 8 ===
n=80 inflated every headline by 5-6pp last week and could not separate 8-20pp gaps.

  python -u -m tools.code_eval -c $DIR/step_0*$at.pt --device xpu:0 \\
    --samples 32 --temperature 0.4 --seed 1234 --repetition-penalty 1.15 \\
    --json reports/code_codemix_${at}.json

BASE TO BEAT (step_0143500, n=320): L3+ 76.6 ±4.6 | committed 25% | L4 5/320
                                     median body_stmts 1

  body_stmts MOVES OFF 1  -> the result. First time in project history. Nothing
      else this year has touched it, and it is what fibonacci/is_prime need.
  committed RISES from 25% -> the model finishes more answers. Second-best outcome.
  L3+ moves, body_stmts and L4 flat -> parseability again. NOT a win; the α ladder
      already showed L3+ can climb 27 points while the model solves nothing.
  degeneracy returns -> dose. Cut STEPS, do not raise the ratio.
EOF
