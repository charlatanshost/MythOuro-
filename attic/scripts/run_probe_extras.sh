#!/usr/bin/env bash
# PROBE EXTRAS — the three tests that are NOT in run_probe_pour.sh's standard set.
#
#   bash run_probe_extras.sh        # ~13 min, run AFTER run_probe_pour.sh finishes
#
# Kept as a separate file on purpose: run_probe_pour.sh was already executing when
# these were written, and editing a live script is how you corrupt a run (bash
# reads by byte offset). New file, no risk, same result.
#
# 1. THE <think> COUNT — the only cheap probe of queue rung 9, which is stuck.
#    The behaviour does NOT originate in the chat corpus: the base at 108,471
#    opens <think> in 51/80 chat-framed samples having never trained on chat data.
#    Something in the CONTINUATION pipeline puts it there — most plausibly 108k
#    steps of on-policy KL against a THINKING teacher. The pour added ~17k more
#    steps of exactly that with zero chat data, so this is a clean read: drift off
#    51/80 implicates the pour and locates the mechanism; flat rules the pour out
#    and points further back. λ was refuted yesterday, so there is no other lead.
#
# 2. best_exit_probe — NEXT-TOKEN, not generation. The disambiguator when the task
#    evals come back flat. At 108,471 the LM improved enormously (CE 0.261 ->
#    0.211, 98% next-token accuracy at the trained depth) while task evals showed
#    little, because task evals score multi-token generation. "Flat" therefore has
#    two very different meanings and only this separates them. Also continues the
#    depth-headroom series (0.093 -> 0.058 nats), the gate on rungs 2 and 5.
#
# 3. A SECOND CODE POINT at 120,000 — so tonight is a line, not a dot. On
#    2026-08-17 code L3+ across six milestones ran 75.0 -> 41.2 -> 53.8 -> 60.0 ->
#    66.2 -> 68.8; the dip is a transient, and an endpoint read called it a
#    permanent -6.2pp loss.
set -uo pipefail
trap 'pkill -INT -P $$ 2>/dev/null; true' INT TERM
cd "$(dirname "$0")"
source ../venv-xpu/bin/activate
export SYCL_CACHE_PERSISTENT=1 PYTORCH_ALLOC_CONF=expandable_segments:True TRITON_DEFAULT_BACKEND=intel

DIR=checkpoints_newmix
LOG="logs/probe_extras_$(date +%Y%m%d_%H%M).log"
mkdir -p logs reports

if pgrep -f "training[.](distill|sft|train_depth_policy)" >/dev/null; then
  echo "a trainer is running — it needs the card; stop the pour first"; exit 1
fi
if pgrep -f "tools[.](code_eval|math_eval|onpolicy_rollout_probe)" >/dev/null; then
  echo "the standard probe is still running — let it finish first"; exit 1
fi

CKPT=$(ls -t "$DIR"/step_*.pt 2>/dev/null | head -1)
[ -n "$CKPT" ] || { echo "no checkpoint in $DIR"; exit 1; }
S=$(basename "$CKPT" | sed 's/step_0*//; s/\.pt//')

{
echo "=== PROBE EXTRAS @ step $S   $(date) ==="

echo; echo "--- 1/3  chat-framed <think> rate (rung 9)   vs 51/80 at the base ---"
python -u -m tools.code_eval -c "$CKPT" --device xpu:0 \
  --samples 8 --temperature 0.4 --seed 1234 --repetition-penalty 1.15 \
  --chat-template --extract --json "reports/code_eval_${S}_CHAT.json"

echo; echo "--- 2/3  best_exit_probe   vs CE 0.211, headroom 0.058 nats ---"
python -u -m tools.best_exit_probe -c "$CKPT" --device xpu:0 \
  --n-loops 8 --trained-loops 4 --by-domain \
  --json "reports/best_exit_${S}.json"

echo; echo "--- 3/3  code @120,000 (trajectory anchor) ---"
if [ -f "$DIR/step_0120000.pt" ]; then
  python -u -m tools.code_eval -c "$DIR/step_0120000.pt" --device xpu:0 \
    --samples 8 --temperature 0.4 --seed 1234 --repetition-penalty 1.15 \
    --json "reports/code_eval_120000_pen115.json"
else
  echo "step_0120000.pt is gone — skipping"
fi
} 2>&1 | tee "$LOG"

echo
echo "=== reports ==="
echo "  reports/code_eval_${S}_CHAT.json      <think> count vs 51/80"
echo "  reports/best_exit_${S}.json           next-token CE + headroom"
echo "  reports/code_eval_120000_pen115.json  trajectory anchor"
