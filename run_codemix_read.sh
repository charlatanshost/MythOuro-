#!/usr/bin/env bash
# CODE-MIX READOUT — did task-completion data move CONTENT? ~26 min, no training.
#
#   bash run_codemix_read.sh
#
# n=32 (320 samples), because n=80 inflated every headline by 5-6pp last week and
# could not separate 8-20pp gaps. Two checkpoints, so this is a trajectory rather
# than an endpoint — the U-shape has fooled us three times.
#
# THE NUMBER THAT DECIDES IT IS body_stmts. Median 1 at every α and every
# checkpoint in project history; the training corpus sits at median 4.
#
# BASE (step_0143500, n=320): L3+ 76.6 ±4.6 | committed 25% | L4 5/320 | body 1
set -uo pipefail
trap 'pkill -INT -P $$ 2>/dev/null; true' INT TERM
cd "$(dirname "$0")"
source ../venv-xpu/bin/activate
export SYCL_CACHE_PERSISTENT=1 PYTORCH_ALLOC_CONF=expandable_segments:True TRITON_DEFAULT_BACKEND=intel
mkdir -p logs reports
if pgrep -f "training[.](distill|sft)" >/dev/null; then echo "a trainer is running"; exit 1; fi

LOG="logs/codemix_read_$(date +%Y%m%d_%H%M).log"
{
for S in 146500 149500; do
  C="checkpoints_codemix/step_0$S.pt"
  [ -f "$C" ] || { echo "MISSING $C"; continue; }
  echo; echo "--- $S ---"
  python -u -m tools.code_eval -c "$C" --device xpu:0 --samples 32 \
    --temperature 0.4 --seed 1234 --repetition-penalty 1.15 \
    --json "reports/code_codemix_${S}.json"
done
} 2>&1 | tee "$LOG"

echo
echo "=== CONTENT vs FORM — base 143,500 is the control ==="
python - <<'EOFPY'
import json,os,re,math,statistics as st
def row(tag,p):
    if not os.path.exists(p): return None
    d=json.load(open(p)); dg=d["diagnostics"]; n=rs=l4=0; stm=[]
    for t in d["tasks"]:
        for s in t["samples"]:
            n+=1; l4+= s["rung"]==4
            if s["rung"]>=3: stm.append(s.get("body_stmts",0))
            if re.findall(r"^\s*def\s+\w+", s["completion"].replace("\\n","\n"), re.M): rs+=1
    p3=dg["per_sample_l3plus"]; ci=1.96*math.sqrt(p3*(1-p3)/n)
    return (tag,100*p3,100*ci,100*dg.get("committed_frac",0),
            st.median(stm) if stm else 0, l4, n, 100*rs/n)
rows=[r for r in [
    row("143,500 base","reports/code_n32_143500.json"),
    row("146,500 code","reports/code_codemix_146500.json"),
    row("149,500 code","reports/code_codemix_149500.json")] if r]
print(f"  {'checkpoint':>14}{'L3+':>14}{'committed':>11}{'body':>7}{'L4':>9}{'restart':>9}")
for tag,p3,ci,cm,md,l4,n,rs in rows:
    print(f"  {tag:>14}{p3:>9.1f}±{ci:<4.1f}{cm:>10.0f}%{md:>7.0f}{l4:>6}/{n}{rs:>8.0f}%")
if len(rows)>1:
    base=rows[0]
    print()
    for r in rows[1:]:
        moved = r[4] > base[4]
        print(f"  {r[0]}: body_stmts {base[4]:.0f} -> {r[4]:.0f}  "
              + ("*** MOVED — first time in project history ***" if moved
                 else "unchanged — the corpus did not transfer"))
EOFPY
