#!/usr/bin/env bash
# ATTEMPT 2 READOUT — did the think-block fix restore CHAT without costing RAW?
#
#   bash run_codemix2_read.sh        # ~26 min, no training
#
# THE QUESTION. Attempt 1 (no closed think block in the corpus) moved CONTENT for
# the first time all year in the RAW frame — L4 5/320 -> 30/320 across four tasks
# instead of two, reverse_string 0 -> 13 — while CHAT framing broke completely,
# think-locked in 320/320. The fix targets chat; the gain came from raw; they
# could interact. Only measuring both settles it.
#
# THREE REFERENCES, all n=320:
#   base 143,500        RAW  L3+ 76.6 ±4.6 | committed 25% | L4  5/320 (2 tasks)
#   attempt 1 RAW       L3+ 54.1 ±5.5 | committed 43% | L4 30/320 (4 tasks)
#   attempt 1 CHAT      L3+  0.0       | committed  0% | L4  0/320, think 320/320
#
# READ IT AS:
#   CHAT recovers AND raw L4 holds near 30  -> the fix worked, attempt 2 is the
#       checkpoint to keep, and the corpus recipe is settled.
#   CHAT recovers but raw L4 falls back to ~5 -> the think block cost the content
#       gain. Keep checkpoints_codemix_nothink for raw work and think again about
#       how to get both.
#   CHAT still think-locked -> the think block was NOT the mechanism, and attempt
#       1's chat failure has another cause. Do not run a third leg before finding it.
set -uo pipefail
trap 'pkill -INT -P $$ 2>/dev/null; true' INT TERM
cd "$(dirname "$0")"
source ../venv-xpu/bin/activate
export SYCL_CACHE_PERSISTENT=1 PYTORCH_ALLOC_CONF=expandable_segments:True TRITON_DEFAULT_BACKEND=intel
mkdir -p logs reports
if pgrep -f "training[.](distill|sft)" >/dev/null; then echo "a trainer is running"; exit 1; fi

C=checkpoints_codemix/step_0149500.pt
LOG="logs/codemix2_read_$(date +%Y%m%d_%H%M).log"
{
echo "--- RAW ---"
python -u -m tools.code_eval -c "$C" --device xpu:0 --samples 32 \
  --temperature 0.4 --seed 1234 --repetition-penalty 1.15 \
  --json reports/code_codemix2_149500_RAW.json
echo; echo "--- CHAT ---"
python -u -m tools.code_eval -c "$C" --device xpu:0 --samples 32 \
  --temperature 0.4 --seed 1234 --repetition-penalty 1.15 \
  --chat-template --extract --json reports/code_codemix2_149500_CHAT.json
} 2>&1 | tee "$LOG"

echo
echo "=== ATTEMPT 2 vs BASE vs ATTEMPT 1 ==="
python - <<'EOFPY'
import json,os,re,math,collections,statistics as st
def row(tag,p):
    if not os.path.exists(p): return None
    d=json.load(open(p)); dg=d["diagnostics"]; n=l4=th=0; stm=[]; by=collections.Counter()
    for t in d["tasks"]:
        for s in t["samples"]:
            n+=1
            if s["rung"]==4: l4+=1; by[t["task"]]+=1
            if s["rung"]>=3: stm.append(s.get("body_stmts",0))
            if "<think>" in s["completion"]: th+=1
    p3=dg["per_sample_l3plus"]; ci=1.96*math.sqrt(p3*(1-p3)/n)
    return (tag,100*p3,100*ci,100*dg.get("committed_frac",0),
            st.median(stm) if stm else 0,l4,n,th,dict(by))
rows=[r for r in [
  row("base 143,500 RAW","reports/code_n32_143500.json"),
  row("attempt1  RAW","reports/code_codemix_149500.json"),
  row("attempt1  CHAT","reports/code_codemix_149500_CHAT.json"),
  row("ATTEMPT2  RAW","reports/code_codemix2_149500_RAW.json"),
  row("ATTEMPT2  CHAT","reports/code_codemix2_149500_CHAT.json")] if r]
print(f"  {'checkpoint':>18}{'L3+':>14}{'commit':>8}{'body':>6}{'L4':>9}{'think':>8}")
for t,p3,ci,cm,md,l4,n,th,by in rows:
    print(f"  {t:>18}{p3:>9.1f}±{ci:<4.1f}{cm:>7.0f}%{md:>6.0f}{l4:>6}/{n}{th:>5}/{n}")
print()
for t,p3,ci,cm,md,l4,n,th,by in rows:
    if by: print(f"  {t}: {by}")
EOFPY
