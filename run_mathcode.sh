# MATH+CODE TEACHER STREAM — the token-starvation fix.
#
#   bash run_mathcode.sh          # ~8.5h; Ctrl-C ONCE and WAIT
#
# THE ACTUAL BLOCKER. The 80% base stream comes from HF and is effectively
# unlimited, but the TEACHER stream — the 20% carrying the distillation signal —
# was data_teacher_v2 + data_teacher_med = 10.9M tokens, ONE EPOCH EVERY 3,339
# STEPS. By step 140,000 that corpus had been re-read dozens of times. That is the
# most plausible root of the 116k-120k regression, and no amount of alpha, lambda,
# loop-weighting or window tuning touches it.
#
# THE FIX: 92M teacher tokens, an 8x expansion, ONE EPOCH EVERY 28,229 STEPS.
# A 6,000-step leg is 0.21 epochs — nothing gets re-read.
#   data_teacher_code  100k rows / ~35M tok  (OpenCodeInstruct, unit-test verified)
#   data_teacher_math  250k rows / ~58M tok  (OpenMathInstruct-2, Llama-3.1-405B)
# Both provenance-cleared in the 2026-07-28 pass; both routed through their
# sft_data.py adapters, never the raw dataset.
#
# WHY MATH. Inspected 2026-08-25 on 2,000 streamed rows: **100% of solutions have
# >=3 statements** (code is 75%), 0% exceed the 1024-token budget. The model is
# stuck at median body_stmts 1 at every alpha ever measured, and math is the
# densest source of multi-step solutions we have. Math capability has also been
# FLAT at 5-11% L3+ all year — it is the domain with the most headroom.
#
# WHY NOT TULU-3 OR PUBMEDQA YET. Tulu-3 is median 0 statements / 81% at one or
# fewer — chat prose, wrong shape for the gap — and its WildChat filter returned
# 0 drops on 2,000 rows where ~10% was expected, which is unresolved. PubMedQA is
# clean and is the mission domain but carries no code, so it cannot move
# body_stmts. Both are worth adding later, neither is tonight's lever.
#
# ROLLOUT WINDOW BACK TO 64. The 256 experiment is done: 1,738 steps changed
# think-block length 381 -> 408 words and closure 48 -> 46 of 320. The window is
# not the mechanism, and 64 is 4x cheaper.
#
# BASE: checkpoints_rollout256/step_0151238 — RAW L3+ 78.8%, committed 41%, the
# best measured checkpoint. checkpoints_codemix stays as the control.
#
# GATE: body_stmts and committed at --samples 32. NOT L3+, which moved 27 points
# during the alpha ladder while strict L4 never left 0-2/320.
#
set -uo pipefail
trap 'pkill -INT -P $$ 2>/dev/null; true' INT TERM
cd "$(dirname "$0")"
source ../venv-xpu/bin/activate
export SYCL_CACHE_PERSISTENT=1 PYTORCH_ALLOC_CONF=expandable_segments:True TRITON_DEFAULT_BACKEND=intel

SRC=checkpoints_rollout256/step_0151238.pt
DIR=checkpoints_mathcode
TEACHER=ByteDance/Ouro-2.6B-Thinking
FILES='data_teacher_code/shard_*.jsonl,data_teacher_math/shard_*.jsonl'
ALPHA=0.45
STEPS=6000
LOG="logs/mathcode_$(date +%Y%m%d_%H%M).log"
OK=reports/mathcode_DONE; FAIL=reports/mathcode_FAILED
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
rows=$(cat data_teacher_code/shard_*.jsonl data_teacher_math/shard_*.jsonl 2>/dev/null | wc -l)
if [ "${rows:-0}" -lt 250000 ]; then
  echo "code+math have only ${rows:-0} rows — expected ~350,000."
  echo "Run: python -m tools.make_code_corpus --target-rows 100000 --out-dir data_teacher_code"
  echo "refusing to start on a short corpus"; exit 1
fi
echo "=== code corpus: $rows rows ==="

# Attempt 1 failed on exactly this. A corpus without the closed think block
# think-locks the model under chat framing; refuse rather than spend the night.
if ! head -1 data_teacher_code/shard_0000.jsonl | grep -q '</think>'; then
  echo "corpus rows do NOT contain a closed think block — this is the attempt-1 bug."
  echo "Build math: python -m tools.make_code_corpus --source clean_math --target-rows 250000 --out-dir data_teacher_math"
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
  --teacher-mix-alpha "$ALPHA" --rollout-len 64 --rollout-batch 32 \
  --rollout-reuse 8 \
  --teacher-data-ratio 0.2 --teacher-data-files "$FILES" \
  --onpolicy-lambda 0.7 \
  --ckpt-dir "$DIR" \
  --ckpt-every-mins 15 --ckpt-milestone-every 500 --keep-last 5 \
  --num-workers 0 --trust-remote-code --log-every 50 \
  --total-steps "$TARGET" 2>&1 | tee "$LOG" || true   # exit code NOT trusted (XPU teardown)

at=$(step_of "$(latest "$DIR")")
echo "mathcode reached $at $(date)" > "$OK"
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
