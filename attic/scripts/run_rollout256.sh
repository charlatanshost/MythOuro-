# ROLLOUT WINDOW 64 → 256 — does the on-policy window reach the answer?
#
#   bash run_rollout256.sh          # ~9.4h; Ctrl-C ONCE and WAIT
#
# THE HYPOTHESIS. Chat framing is 0.0% and three explanations died (corpus think
# block, eval max_new, closing-is-enough). What survives: every run in project
# history used --rollout-len 64, and under chat framing the model's first 64
# tokens sit entirely inside a think block that runs ~495 tokens. So on-policy
# training has NEVER shaped the exit from reasoning, or the answer after it — only
# "how to reason for 64 more tokens". That accounts for both the unbounded
# expansion and the missing answer.
#
# WHY 256 AND NOT 128. A null at 128 would be uninterpretable: mechanism wrong, or
# window still far short of 495? Since the night costs the same either way, buy
# the bigger window and trade STEPS for it. 256 is still short of 495 — this tests
# whether PARTIALLY extending the window starts shaping the exit, not whether it
# covers think+answer. 512 would, and extrapolates to ~13h of rollout alone.
#
# THE BUDGET, measured not projected (run_bench_rollout.sh, 9 configs):
#   rollout  500 regenerations (4,000 steps / reuse 8) x 36.60 s = 5.1 h
#   training 4,000 steps x ~3.84 s (from the 6,000-step leg: 8.5h - 2.1h rollout)
#   total    ~9.4 h
# Scaling measured at ~7x for 64→256, NOT the 16x O(n^2) predicts — an earlier
# note called this "not viable" from projection alone and was wrong by >2x.
#
# ⚠ BATCH 32 → 16 HALVES the on-policy sequences per regeneration. That is a real
# trade, not free throughput; tok/s actually RISES with batch (144→205 at len 64),
# so the big batch is more efficient per token. Batch 8 would be cheaper still and
# quarter them — 16 is the middle.
#
# ⚠ 4,000 STEPS, not 6,000. Trading steps for window. The α-anneal showed a
# visible effect by 1,500 steps, so this is not obviously too few, and milestones
# every 500 surface movement early.
#
# BASE: checkpoints_codemix/step_0149500.pt — the code-mix leg that moved content
# (RAW committed 42%, L4 14/320, reverse_string solved). checkpoints_codemix is
# untouched as the control.
#
# GATE: chat-framed L3+ off 0.0%, and the think-block LENGTH falling. Measure with
#   --chat-template --extract --samples 32 --max-new 512
# and read the closed-block fraction, not just the score.
#
set -uo pipefail
trap 'pkill -INT -P $$ 2>/dev/null; true' INT TERM
cd "$(dirname "$0")"
source ../venv-xpu/bin/activate
export SYCL_CACHE_PERSISTENT=1 PYTORCH_ALLOC_CONF=expandable_segments:True TRITON_DEFAULT_BACKEND=intel

SRC=checkpoints_codemix/step_0149500.pt
DIR=checkpoints_rollout256
TEACHER=ByteDance/Ouro-2.6B-Thinking
FILES='data_teacher_code/shard_*.jsonl'
ALPHA=0.45
STEPS=4000
LOG="logs/rollout256_$(date +%Y%m%d_%H%M).log"
OK=reports/rollout256_DONE; FAIL=reports/rollout256_FAILED
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

# Attempt 1 failed on exactly this. A corpus without the closed think block
# think-locks the model under chat framing; refuse rather than spend the night.
if ! head -1 data_teacher_code/shard_0000.jsonl | grep -q '</think>'; then
  echo "corpus rows do NOT contain a closed think block — this is the attempt-1 bug."
  echo "Rebuild: python -m tools.make_code_corpus --target-rows 100000 --out-dir data_teacher_code"
  echo "refusing to start"; exit 1
fi
echo "=== corpus carries the closed think block ✓ ==="

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
  --teacher-mix-alpha "$ALPHA" --rollout-len 256 --rollout-batch 16 \
  --rollout-reuse 8 \
  --teacher-data-ratio 0.2 --teacher-data-files "$FILES" \
  --onpolicy-lambda 0.7 \
  --ckpt-dir "$DIR" \
  --ckpt-every-mins 15 --ckpt-milestone-every 500 --keep-last 5 \
  --num-workers 0 --trust-remote-code --log-every 50 \
  --total-steps "$TARGET" 2>&1 | tee "$LOG" || true   # exit code NOT trusted (XPU teardown)

at=$(step_of "$(latest "$DIR")")
echo "rollout256 reached $at $(date)" > "$OK"
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
