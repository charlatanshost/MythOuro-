#!/usr/bin/env bash
# n=32 — the first eval this project has run that can DISTINGUISH these checkpoints.
#
#   bash run_n32.sh          # ~40 min, no training
#
# WHY. Every comparison this week ran at --samples 8 = 80 samples, giving ±10pp
# 95% intervals, while the differences being argued over were 8-20pp. Four metrics
# picked three different "best" checkpoints:
#   L3+        -> 143,500 (82%)
#   committed  -> 141,500 (50%)
#   restart    -> 140,000 (16%)
#   strict L4  -> cannot tell, everything is 0-2/80
# That is the signature of measuring noise, not finding a lever.
#
# 32 samples x 10 tasks = 320 per checkpoint, roughly halving the interval to
# ~±5pp. Three checkpoints only — the two anneal candidates and the pre-anneal
# control — because resolution matters more than coverage here.
#
# NOTE the seed is unchanged (1234) so the first 8 samples per task reproduce the
# existing runs exactly; this EXTENDS those measurements rather than replacing them.
set -uo pipefail
trap 'pkill -INT -P $$ 2>/dev/null; true' INT TERM
cd "$(dirname "$0")"
source ../venv-xpu/bin/activate
export SYCL_CACHE_PERSISTENT=1 PYTORCH_ALLOC_CONF=expandable_segments:True TRITON_DEFAULT_BACKEND=intel
mkdir -p logs reports
if pgrep -f "training[.](distill|sft)" >/dev/null; then echo "a trainer is running"; exit 1; fi

LOG="logs/n32_$(date +%Y%m%d_%H%M).log"
{
for S in 140000 141500 143500; do
  C="checkpoints_anneal045/step_0$S.pt"
  [ -f "$C" ] || C="checkpoints_base/step_0$S.pt"
  [ -f "$C" ] || C="checkpoints_newmix/step_0$S.pt"
  [ -f "$C" ] || { echo "MISSING step_0$S.pt"; continue; }
  echo; echo "--- $S  ($C) ---"
  python -u -m tools.code_eval -c "$C" --device xpu:0 --samples 32 \
    --temperature 0.4 --seed 1234 --repetition-penalty 1.15 \
    --json "reports/code_n32_${S}.json"
done
} 2>&1 | tee "$LOG"

echo
echo "=== n=320 per checkpoint — intervals now ~±5pp ==="
python - <<'EOFPY'
import json,os,re,math
print(f"  {'step':>8}{'L3+':>16}{'committed':>12}{'restart':>9}{'L4':>8}")
for S in (140000,141500,143500):
    p=f"reports/code_n32_{S}.json"
    if not os.path.exists(p): continue
    d=json.load(open(p)); dg=d["diagnostics"]; n=rs=l4=0
    for t in d["tasks"]:
        for s in t["samples"]:
            n+=1; l4+= s["rung"]==4
            if re.findall(r"^\s*def\s+\w+", s["completion"].replace("\\n","\n"), re.M): rs+=1
    p3=dg["per_sample_l3plus"]; ci=1.96*math.sqrt(p3*(1-p3)/n)
    print(f"  {S:>8}{100*p3:>10.1f}±{100*ci:<4.1f}"
          f"{100*dg.get('committed_frac',0):>11.0f}%{100*rs/n:>8.0f}%{l4:>5}/{n}")
print("\n  Non-overlapping intervals = a real difference. Overlapping = these")
print("  checkpoints are the same model and the week's rankings were noise.")
EOFPY
