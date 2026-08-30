#!/usr/bin/env bash
# CE BRACKET — WHEN did next-token prediction start degrading, and how fast?
#
#   bash run_ce_bracket.sh          # ~20 min, no training, safe to Ctrl-C
#
# THE FINDING THIS CHASES (2026-08-18). best_exit_probe at step 125,181 vs the
# 108,471 baseline, same tool and same 3,072 positions, showed next-token CE
# 1.7-3.5x WORSE on all six domains — general 0.272->0.454, code 0.175->0.610,
# medical 0.260->0.846. Task evals agree: raw code L3+ 75.0 -> 60.0 -> 55.0
# monotone. The pour is not strengthening the base; it is degrading it.
#
# WHY IT MATTERS WHICH SHAPE. Smooth degradation from 108,471 means DOSE — the
# knob is --onpolicy-lambda or --teacher-data-ratio, and the fix is a milder
# recipe. A STEP CHANGE at one point means an event, and the ce curve would name
# it — except that stretch ran unlogged (fixed 2026-08-18, b98b177, but too late
# for these 17k steps). This probe is the only way back to the shape.
#
# THE SUSPECTED MECHANISM, for what the numbers should be read against: chat-framed
# code L3+ went 6.2% -> 37.5% over the same steps WITH NO CHAT DATA, 42/80 raw
# completions now emit <|im_end|> unprompted, and CE on CONTINUATION corpora
# collapsed. That is a model drifting off the continuation distribution toward its
# teacher's chat/thinking shape — 17k steps of on-policy KL at lambda 0.7 against
# Ouro-2.6B-THINKING. If so, CE should climb roughly monotonically with steps.
#
# ⚠ THE 108,471 CHECKPOINT NO LONGER EXISTS. The trainer's own --keep-last 5
# rotated it out (it was a resume checkpoint, not an even-2000 milestone). The
# pruner never touched it — checkpoints_newmix is in its _PROTECTED list — so the
# protection everyone trusted was guarding the wrong door. step_0108000.pt is 471
# steps away and is measured here as the anchor, which also re-validates the old
# 0.2036-mean number with today's tooling.
set -uo pipefail
trap 'pkill -INT -P $$ 2>/dev/null; true' INT TERM
cd "$(dirname "$0")"
source ../venv-xpu/bin/activate     # `python` does not exist outside the venv
export SYCL_CACHE_PERSISTENT=1 PYTORCH_ALLOC_CONF=expandable_segments:True TRITON_DEFAULT_BACKEND=intel

DIR=checkpoints_newmix
STEPS="108000 112000 116000 120000"   # 125,181 already measured
LOG="logs/ce_bracket_$(date +%Y%m%d_%H%M).log"
mkdir -p logs reports

if pgrep -f "training[.](distill|sft|train_depth_policy)" >/dev/null; then
  echo "a trainer is running — it needs the card; stop it first"; exit 1
fi
if pgrep -f "tools[.](code_eval|math_eval|best_exit_probe|onpolicy_rollout_probe)" >/dev/null; then
  echo "another probe is running; let it finish"; exit 1
fi

{
echo "=== CE BRACKET  $(date) ==="
for S in $STEPS; do
  C="$DIR/step_0$S.pt"
  [ -f "$C" ] || { echo "MISSING $C — skipping"; continue; }
  echo; echo "--- $S ---"
  python -u -m tools.best_exit_probe -c "$C" --device xpu:0 \
    --n-loops 8 --trained-loops 4 --by-domain \
    --json "reports/best_exit_$S.json"
done
} 2>&1 | tee "$LOG"

echo
echo "=== CE at the trained depth, by step ==="
python3 - <<'PY'
import json, glob, re
pts=[]
for f in glob.glob("reports/best_exit_*.json"):
    m=re.search(r"best_exit_(\d+)\.json$", f)
    if not m: continue
    d=json.load(open(f)); bd=d.get("by_domain")
    if not bd: continue
    pts.append((int(m.group(1)), {e["domain"]: e["ce_trained_depth"] for e in bd}))
pts.sort()
if pts:
    doms=list(pts[0][1])
    print("  step     " + "".join(f"{d[:9]:>10}" for d in doms) + f"{'MEAN':>9}")
    for s,v in pts:
        mean=sum(v.values())/len(v)
        print(f"  {s:<9}" + "".join(f"{v[d]:10.3f}" for d in doms) + f"{mean:9.3f}")
print("\n  108,471 reference (file since rotated away): mean 0.2036")
print("  125,181 measured 2026-08-18:                 mean 0.5608")
print("\n  SMOOTH climb -> dose; lower --onpolicy-lambda or --teacher-data-ratio.")
print("  STEP change  -> an event; find which 2,000-step window owns it.")
PY
