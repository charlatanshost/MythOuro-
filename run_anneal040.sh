# α-ANNEAL 0.45 → 0.40 — third anneal. The 0.45 leg is why this is running.
#
#   bash run_anneal040.sh          # ~8.5h; Ctrl-C ONCE and WAIT
#
# THE 0.45 RESULT (2026-08-20). Raw code L3+ 65.0 → 76.2 → 80.0 → 82.5 across
# 3,500 steps — monotone, past the old 75.0% record after 1,500 steps, degeneracy
# 0/80 throughout. Chat flat (33.8 → 31.2, inside ±10pp) so it was a raw-frame
# gain, not a trade. α=0.0 prose moved for the first time since step 90,351
# (top_share 0.139 → 0.092 over four seeds; the C-C-C letter-salad mode is gone).
#
# So this leg is justified by RESULTS, not by the plateau rule that justified 0.45.
# The 50,000-step α=0.0 plateau has broken; we are now following a measured +17.5pp.
#
# ⚠ WHAT COULD GO WRONG, and it is not the same risk as last time:
#   * SATURATION — the 0.45 gain may not repeat. Milestones every 500 will show
#     where it flattens. If raw code stops climbing by ~146,000, that is the answer
#     and the leg can be stopped early; nothing is lost.
#   * REGRESSION at α=0.0 on code/math. The 0.45 leg already made fibonacci WORSE
#     at α=0.0 (0.246 → 0.370) while capability went UP, because the rollout probe
#     samples at T=1.0 with no repetition penalty. Do NOT abort on that signal
#     alone — confirm with code_eval at deployment settings before concluding.
#   * TRUE re-collapse — a PROSE seed spiking past 0.40 at α=0.0. That one is real
#     and is the abort signal.
#
# FALLBACK: checkpoints_anneal045/step_0143500.pt is the project's best checkpoint
# (82.5% raw) and is also copied to checkpoints_base/. checkpoints_newmix remains
# the untouched α=0.5 control.
#
set -uo pipefail
trap 'pkill -INT -P $$ 2>/dev/null; true' INT TERM
cd "$(dirname "$0")"
source ../venv-xpu/bin/activate
export SYCL_CACHE_PERSISTENT=1 PYTORCH_ALLOC_CONF=expandable_segments:True TRITON_DEFAULT_BACKEND=intel

SRC=checkpoints_anneal045/step_0143500.pt
DIR=checkpoints_anneal040
TEACHER=ByteDance/Ouro-2.6B-Thinking
FILES='data_teacher_v2/shard_*.jsonl,data_teacher_med/shard_*.jsonl'
ALPHA=0.40                       # THE ONE VARIABLE. the previous leg ran 0.45.
TARGET=149500
LOG="logs/anneal040_$(date +%Y%m%d_%H%M).log"
OK=reports/anneal040_DONE; FAIL=reports/anneal040_FAILED
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
echo "anneal040 reached $at $(date)" > "$OK"
echo "=== stopped at step $at ==="
cat <<EOF

THE READOUT IS α=0.0, NOT THE LOSS. Loss rising is expected here.

  python -u -m tools.onpolicy_rollout_probe --ckpt-dir $DIR \\
    --student-device xpu:0 --teacher-device xpu:0 --teacher-id $TEACHER \\
    --trust-remote-code --no-kv-cache --samples 5 \\
    --json reports/probe_anneal040_${at}.json | tee reports/probe_anneal040_${at}.txt

READ IT AGAINST @140,000: α=0.0 top_share 0.154 | distinct1 0.477 | halt 2.00/4.
  top_share DOWN and no seed re-collapsing  -> the anneal is working, same as
      0.6->0.5 did (0.18 -> 0.12, 4/6 seeds improved, fragile seed de-fragilized).
  any seed spiking back toward 0.4+         -> re-collapse. That is the abort
      signal; checkpoints_newmix is untouched and step_0140000.pt is the fallback.
  FLAT again                                -> α is not the lever at this scale
      and the exposure-bias gap needs a different attack (rung 6, on-policy SFT).
EOF
