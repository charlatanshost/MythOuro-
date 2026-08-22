#!/usr/bin/env bash
# DID `committed` RECOVER? — the readout for the 143,500 -> 150,000 leg. ~12 min.
#
#   bash run_committed_trend.sh
#
# THE QUESTION. `committed` (returned a value rather than falling off the end)
# ran 50% at 141,500 -> 48% at 142,500 -> 28% at 143,500, while L3+ went the
# other way (76 -> 80 -> 82). Either the 28% was noise, or 141,500 was the peak
# and the late leg was degrading. This leg ran 6,500 more steps at the same α, so
# three points across it settle which.
#
#   recovers toward 50% and climbs  -> 28% was noise, the anneal still works
#   stays near 28% through 150,000  -> 141,500 WAS the peak; re-seed from it
#
# ⚠ THE ONE THAT WOULD MATTER MOST: median body_stmts. It has been 1 at every α
# and every checkpoint in project history — the model writes a single statement
# and stops. If it moves off 1, the model is reaching answers that need more than
# one line, which is what fibonacci and is_prime require and what is_even (the
# only task that ever grades correct) never did. That is a bigger result than any
# L3+ number.
#
# Judge on `committed` and body_stmts. NOT on L3+, which climbed 27 points this
# week while strict L4 stayed at 0-2/80.
set -uo pipefail
trap 'pkill -INT -P $$ 2>/dev/null; true' INT TERM
cd "$(dirname "$0")"
source ../venv-xpu/bin/activate
export SYCL_CACHE_PERSISTENT=1 PYTORCH_ALLOC_CONF=expandable_segments:True TRITON_DEFAULT_BACKEND=intel
mkdir -p logs reports
if pgrep -f "training[.](distill|sft)" >/dev/null; then echo "a trainer is running"; exit 1; fi

LOG="logs/committed_trend_$(date +%Y%m%d_%H%M).log"
{
for S in 145000 147500 150000; do
  C="checkpoints_anneal045/step_0$S.pt"
  [ -f "$C" ] || { echo "MISSING $C"; continue; }
  echo; echo "--- $S ---"
  python -u -m tools.code_eval -c "$C" --device xpu:0 --samples 8 --temperature 0.4 \
    --seed 1234 --repetition-penalty 1.15 \
    --json "reports/code_anneal045_${S}_pen115.json"
done
} 2>&1 | tee "$LOG"

echo
echo "=== committed / L3+ / body across the whole project ==="
python - <<'EOFPY'
import json,os,sys,subprocess,tempfile,statistics as st
sys.path.insert(0,".")
from tools.code_eval import _TASKS,_truncate_to_parseable,_PROBE
pr={n:p for n,p,_ in _TASKS}
def an(path):
    d=json.load(open(path)); n80=l3=com=0; stm=[]
    for t in d["tasks"]:
        f=t["task"]
        if f not in pr: continue
        for s in t["samples"]:
            n80+=1
            if s["rung"]<3: continue
            l3+=1; stm.append(s.get("body_stmts",0))
            if "committed" in s: com+=bool(s["committed"]); continue
            code,_=_truncate_to_parseable((pr[f]+s["completion"]).replace("\\n","\n").replace("\\t","\t"))
            if not code: continue
            sc=f"{code}\ntry:\n    r={_PROBE[f]}\n    print('RET',repr(r))\nexcept Exception:\n    print('E')\n"
            with tempfile.TemporaryDirectory() as td:
                p=os.path.join(td,"t.py"); open(p,"w").write(sc)
                try: r=subprocess.run([sys.executable,p],capture_output=True,text=True,timeout=5)
                except Exception: continue
            if r.stdout.startswith("RET") and not r.stdout.startswith("RET None"): com+=1
    return n80,l3,com,(st.median(stm) if stm else 0)
rows=[(64000,"0.50","reports/code_eval_pen1.15_64000.json"),
      (108471,"0.50","reports/code_eval_108471_pen1.15.json"),
      (140000,"0.50","reports/code_eval_140000_pen115.json"),
      (141500,"0.45","reports/code_anneal045_141500_pen115.json"),
      (143500,"0.45","reports/code_anneal045_143500_pen115.json"),
      (145000,"0.45","reports/code_anneal045_145000_pen115.json"),
      (147500,"0.45","reports/code_anneal045_147500_pen115.json"),
      (150000,"0.45","reports/code_anneal045_150000_pen115.json")]
print(f"  {'step':>8}{'α':>6}{'L3+':>7}{'committed':>11}{'body':>6}")
for s,a,p in rows:
    if not os.path.exists(p): continue
    n80,l3,com,med=an(p)
    print(f"  {s:>8}{a:>6}{100*l3/n80:>6.0f}%{100*com/n80:>10.0f}%{med:>6.0f}")
print("\n  141,500 = 50% committed is the mark. body has never left 1.")
EOFPY
