#!/usr/bin/env bash
# ROLLOUT-256 READOUT — did a wider on-policy window shape the exit from reasoning?
#
#   bash run_rollout256_read.sh        # ~35 min, no training
#
# 1,738 steps at --rollout-len 256 (vs 64 everywhere before), stopped early: the
# leg ran at 28.9 s/step against a planned 8.5, because the cost bench measured
# the student generating ALONE (teacher=None) while training runs
# --teacher-mix-alpha 0.45, so a 2.6B teacher forwards at every generated token
# too. 1,738 steps still exceeds the 1,500 where the α-anneal first showed.
#
# THE HEADLINE IS NOT THE SCORE. Chat L3+ has been 0.0% through every variant.
# The diagnostic is THINK-BLOCK LENGTH: it expands to fill whatever budget it is
# given — 94 tokens at a 96 cap, 495 at a 512 cap. If the on-policy window is what
# shapes the exit, that length should FALL even before the score moves.
#
# REFERENCES (step_0149500, the codemix base):
#   CHAT @max_new 512: L3+ 0.0% | think 320/320 | closed 48/320 | median 381 words
#   RAW  @max_new 96 : L3+ 76.9 ±4.6 | committed 42% | L4 14/320
set -uo pipefail
trap 'pkill -INT -P $$ 2>/dev/null; true' INT TERM
cd "$(dirname "$0")"
source ../venv-xpu/bin/activate
export SYCL_CACHE_PERSISTENT=1 PYTORCH_ALLOC_CONF=expandable_segments:True TRITON_DEFAULT_BACKEND=intel
mkdir -p logs reports
if pgrep -f "training[.](distill|sft)" >/dev/null; then echo "a trainer is running"; exit 1; fi

C=$(ls -t checkpoints_rollout256/step_*.pt | head -1)
S=$(basename "$C" | sed 's/step_0*//; s/\.pt//')
LOG="logs/rollout256_read_$(date +%Y%m%d_%H%M).log"
{
echo "--- CHAT @ max_new 512 (the diagnostic) ---"
python -u -m tools.code_eval -c "$C" --device xpu:0 --samples 32 \
  --temperature 0.4 --seed 1234 --repetition-penalty 1.15 \
  --chat-template --extract --max-new 512 \
  --json "reports/code_rollout256_${S}_CHAT512.json"
echo; echo "--- RAW @ default (does the content gain survive?) ---"
python -u -m tools.code_eval -c "$C" --device xpu:0 --samples 32 \
  --temperature 0.4 --seed 1234 --repetition-penalty 1.15 \
  --json "reports/code_rollout256_${S}_RAW.json"
} 2>&1 | tee "$LOG"

echo
echo "=== THINK-BLOCK LENGTH — the number that decides this ==="
python - "$S" <<'EOFPY'
import json,sys,os,statistics as st
S=sys.argv[1]
def think(p,tag):
    if not os.path.exists(p): return
    d=json.load(open(p)); lens=[]; closed=0; n=0; fence=0
    for t in d["tasks"]:
        for s in t["samples"]:
            c=s["completion"]; n+=1; fence += "```" in c
            if "<think>" in c:
                a=c.split("<think>",1)[1]
                if "</think>" in a: closed+=1; lens.append(len(a.split("</think>")[0].split()))
                else: lens.append(len(a.split()))
    dg=d["diagnostics"]
    print(f"  {tag}")
    print(f"    L3+ {100*dg['per_sample_l3plus']:.1f}%  committed {100*dg.get('committed_frac',0):.0f}%  fence {fence}/{n}")
    if lens:
        print(f"    think: {len(lens)}/{n} opened, {closed} CLOSED, median {st.median(lens):.0f} words")
think(f"reports/code_rollout256_{S}_CHAT512.json","rollout256 CHAT @512")
print("  base 149,500 CHAT @512")
print("    L3+ 0.0%  committed 0%  fence 5/320")
print("    think: 320/320 opened, 48 CLOSED, median 381 words")
print()
think(f"reports/code_rollout256_{S}_RAW.json","rollout256 RAW")
print("  base 149,500 RAW: L3+ 76.9 ±4.6  committed 42%  L4 14/320")
EOFPY
