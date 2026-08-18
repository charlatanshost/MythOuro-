#!/usr/bin/env bash
# MID-POUR PROBE — the standing test set, run at the current checkpoint.
#
#   bash run_probe_pour.sh          # ~20 min, no training, safe to Ctrl-C
#
# This is the trio documented in run_newmix_pour.sh's own tail, with the one
# addition the 2026-07-30 protocol requires: code is measured at BOTH repetition
# penalties, because they answer different questions and reporting only one has
# misled us before. pen 1.0 measures DEGENERACY (does it loop?), pen 1.15 measures
# CAPABILITY (can it write runnable code when not looping?). At 108,471 those were
# 48.8% and 75.0% L3+ on the same weights — quoting either alone is a half-truth.
#
# EVERYTHING USES THE BASELINE SETTINGS (--samples 8, --seed 1234, temp 0.4 for
# code) so the numbers drop straight into the tracker's series. Do not tune them
# to make a run look better; the comparability IS the value.
#
# BASELINES TO BEAT — all at step 108,471, the pour's resume point:
#   code pen 1.15   75.0% L3+   0/80 char-degenerate   1/80 looped
#   code pen 1.0    48.8% L3+   67/80 char-degenerate
#   math            5.0% L3+    37.5% copied-from-prompt
#   rollout probe   4th point in the series (90,351 -> 100,000 -> 108,471 -> now)
#
# ⚠ MATH IS OSCILLATING, NOT CLIMBING: 1.2 -> 3.8 -> 11.2 -> 5.0 across four
# points. A single better math number is NOT a trend — that exact mistake was
# corrected in the tracker on 2026-08-06 after a two-point claim failed at four.
# The same caution applies harder to L4, which at n=80 is counting noise.
#
# ⚠ READ THE TEXT. Aggregates have pointed the wrong way four times on this
# project. The JSONs keep completions; spot-read before concluding anything.
set -uo pipefail
trap 'pkill -INT -P $$ 2>/dev/null; true' INT TERM
cd "$(dirname "$0")"
source ../venv-xpu/bin/activate
export SYCL_CACHE_PERSISTENT=1 PYTORCH_ALLOC_CONF=expandable_segments:True TRITON_DEFAULT_BACKEND=intel

DIR=checkpoints_newmix
TEACHER=ByteDance/Ouro-2.6B-Thinking
LOG="logs/probe_pour_$(date +%Y%m%d_%H%M).log"
mkdir -p logs reports

# Evals need the whole card; against a live trainer one or both OOM, and the
# trainer is the expensive one to lose.
if pgrep -f "training[.](distill|sft|train_depth_policy)" >/dev/null; then
  echo "a trainer is running — it needs the card; stop the pour first"; exit 1
fi
if pgrep -f "tools[.]gen_teacher_corpus" >/dev/null; then
  echo "a harvest is running; refusing to start"; exit 1
fi

CKPT=$(ls -t "$DIR"/step_*.pt 2>/dev/null | head -1)
[ -n "$CKPT" ] || { echo "no checkpoint in $DIR"; exit 1; }
S=$(basename "$CKPT" | sed 's/step_0*//; s/\.pt//')

{
echo "=== MID-POUR PROBE @ step $S   $(date) ==="
echo "=== baselines @108,471: code 75.0%/48.8% | math 5.0% | probe = 4th in series ==="

echo; echo "--- 1/4  code, pen 1.15 (CAPABILITY) ---"
python -u -m tools.code_eval -c "$CKPT" --device xpu:0 \
  --samples 8 --temperature 0.4 --seed 1234 --repetition-penalty 1.15 \
  --json "reports/code_eval_${S}_pen115.json"

echo; echo "--- 2/4  code, pen 1.0 (DEGENERACY) ---"
python -u -m tools.code_eval -c "$CKPT" --device xpu:0 \
  --samples 8 --temperature 0.4 --seed 1234 \
  --json "reports/code_eval_${S}_pen100.json"

echo; echo "--- 3/4  math ---"
python -u -m tools.math_eval -c "$CKPT" --device xpu:0 \
  --samples 8 --seed 1234 --json "reports/math_eval_${S}.json"

echo; echo "--- 4/4  on-policy rollout probe (teacher loads here; slowest) ---"
python -u -m tools.onpolicy_rollout_probe --ckpt-dir "$DIR" \
  --student-device xpu:0 --teacher-device xpu:0 --teacher-id "$TEACHER" \
  --trust-remote-code --no-kv-cache --samples 5 \
  --json "reports/probe_${S}.json" | tee "reports/probe_${S}.txt"

} 2>&1 | tee "$LOG"

cat <<EOF

=== reports written ===
  reports/code_eval_${S}_pen115.json   vs 75.0% L3+
  reports/code_eval_${S}_pen100.json   vs 48.8% L3+, 67/80 degenerate
  reports/math_eval_${S}.json          vs 5.0% L3+, 37.5% copied
  reports/probe_${S}.json / .txt       4th point in the rollout series

Use the .json for the probe — the .txt keeps only sample #1 of n.

WHAT WOULD ACTUALLY BE NEWS:
  * pen 1.0 degeneracy FALLING from 67/80 — that is the attractor weakening on
    its own, which no amount of penalty tuning can fake.
  * pen 1.15 L3+ meaningfully above 75.0% — the pour buying real capability.
  * math copied-from-prompt down from 37.5% WITHOUT L3+ falling.
  * FLAT everything: also news. 17k steps for nothing changes the queue — the
    weak base stops being the reason rungs 2/5/9 are gated.
EOF
