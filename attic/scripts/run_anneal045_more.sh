# CONTINUE α=0.45 — it was still climbing when we left it.
#
#   bash run_anneal045_more.sh          # ~9h; Ctrl-C ONCE and WAIT
#
# WHY THIS AND NOT A DEEPER ANNEAL. Going 0.45 → 0.40 was premature and it cost:
#
#   α      step      RAW    CHAT   α=0.0 prose
#   0.50   140,000   65.0   33.8      0.139
#   0.45   143,500   82.5   31.2      0.092    <- best RAW *and* best prose
#   0.40   149,500   58.8   50.0*     0.108
#
#   * CHAT at 0.40 is NOT a stable 50%. Across the leg it read 40.0 → 3.8 → 50.0.
#     At 147,500 the model opened <think> in 79/80 samples and produced a code
#     fence in 6/80 — the chat-mix "reasons forever, never answers" pathology,
#     returning transiently. A number that swings 46 points in 2,000 steps is not
#     a capability, and RAW at 0.40 was flat and low throughout (58.8/60.0/58.8).
#
# THE ACTUAL REASON TO COME BACK: the 0.45 leg NEVER SATURATED. Its three points
# were 76.2 → 80.0 → 82.5, still rising when the leg hit its 3,500-step target.
# We changed α instead of finishing the experiment. This resumes it — same α, same
# dir, 6,500 more steps — and the milestone spacing will show where it flattens.
#
# ⚠ RESUMES IN PLACE in checkpoints_anneal045, continuing the same line rather
# than branching. step_0143500.pt (82.5% RAW, the project best) is also in
# checkpoints_base/, and checkpoints_newmix remains the untouched α=0.5 control.
#
# ABORT: a PROSE seed past 0.40 at α=0.0. A code/math seed worsening is not an
# abort signal — that happened at 0.45 while capability rose 17.5 points.
#
set -uo pipefail
trap 'pkill -INT -P $$ 2>/dev/null; true' INT TERM
cd "$(dirname "$0")"
source ../venv-xpu/bin/activate
export SYCL_CACHE_PERSISTENT=1 PYTORCH_ALLOC_CONF=expandable_segments:True TRITON_DEFAULT_BACKEND=intel

SRC=checkpoints_anneal045/step_0143500.pt
DIR=checkpoints_anneal045
TEACHER=ByteDance/Ouro-2.6B-Thinking
FILES='data_teacher_v2/shard_*.jsonl,data_teacher_med/shard_*.jsonl'
ALPHA=0.45                       # UNCHANGED from the leg that produced 82.5%.
TARGET=150000
LOG="logs/anneal045cont_$(date +%Y%m%d_%H%M).log"
OK=reports/anneal045cont_DONE; FAIL=reports/anneal045cont_FAILED
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
echo "anneal045cont reached $at $(date)" > "$OK"
echo "=== stopped at step $at ==="
cat <<EOF

THE READOUT IS α=0.0, NOT THE LOSS. Loss rising is expected here.

  python -u -m tools.onpolicy_rollout_probe --ckpt-dir $DIR \\
    --student-device xpu:0 --teacher-device xpu:0 --teacher-id $TEACHER \\
    --trust-remote-code --no-kv-cache --samples 5 \\
    --json reports/probe_anneal045cont_${at}.json | tee reports/probe_anneal045cont_${at}.txt

READ IT AGAINST @140,000: α=0.0 top_share 0.154 | distinct1 0.477 | halt 2.00/4.
  top_share DOWN and no seed re-collapsing  -> the anneal is working, same as
      0.6->0.5 did (0.18 -> 0.12, 4/6 seeds improved, fragile seed de-fragilized).
  any seed spiking back toward 0.4+         -> re-collapse. That is the abort
      signal; checkpoints_newmix is untouched and step_0140000.pt is the fallback.
  FLAT again                                -> α is not the lever at this scale
      and the exposure-bias gap needs a different attack (rung 6, on-policy SFT).
EOF
