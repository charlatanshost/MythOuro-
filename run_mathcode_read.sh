#!/usr/bin/env bash
# MATH+CODE READOUT — did an 8x teacher corpus move CONTENT? ~30 min, no training.
#
#   bash run_mathcode_read.sh
#
# 6,000 steps on 92M teacher tokens (code 35M + math 58M), one epoch every 28,229
# steps — against the 10.9M / 3,339-step corpus the pour was re-reading when it
# regressed. This is the token-starvation fix, measured.
#
# BOTH AXES, because math data was added for the first time:
#   code  RAW n=320 — body_stmts and committed, NOT L3+
#   math  n=320     — flat at 5-11% L3+ all year, the domain with the most headroom
#
# BASELINES
#   rollout256 @151,238 RAW : L3+ 78.8%  committed 41%  body 1
#   codemix    @149,500 RAW : L3+ 76.9%  committed 42%  L4 14/320  body 1
#   math       @125,181     : L3+ 11.25% (n=80, pen 1.0) — the series has been
#                             1.2 / 3.8 / 11.2 / 5.0 / 11.25, oscillating, no trend
#
# ⚠ body_stmts is median 1 at EVERY α and EVERY checkpoint ever measured. The math
# corpus is 100% >=3 statements. If it moves, that is the result.
set -uo pipefail
trap 'pkill -INT -P $$ 2>/dev/null; true' INT TERM
cd "$(dirname "$0")"
source ../venv-xpu/bin/activate
export SYCL_CACHE_PERSISTENT=1 PYTORCH_ALLOC_CONF=expandable_segments:True TRITON_DEFAULT_BACKEND=intel
mkdir -p logs reports
if pgrep -f "training[.](distill|sft)" >/dev/null; then echo "a trainer is running"; exit 1; fi

C=$(ls -t checkpoints_mathcode/step_*.pt | head -1)
S=$(basename "$C" | sed 's/step_0*//; s/\.pt//')
LOG="logs/mathcode_read_$(date +%Y%m%d_%H%M).log"
{
echo "--- code RAW n=320 ---"
python -u -m tools.code_eval -c "$C" --device xpu:0 --samples 32 \
  --temperature 0.4 --seed 1234 --repetition-penalty 1.15 \
  --json "reports/code_mathcode_${S}.json"
echo; echo "--- math n=320 ---"
python -u -m tools.math_eval -c "$C" --device xpu:0 --samples 32 --seed 1234 \
  --json "reports/math_mathcode_${S}.json"
} 2>&1 | tee "$LOG"

echo
echo "=== vs the base it was seeded from ==="
python - "$S" <<'EOFPY'
import json,sys,os,statistics as st,collections
S=sys.argv[1]
def code(p,tag):
    if not os.path.exists(p): return
    d=json.load(open(p)); dg=d["diagnostics"]; stm=[]; l4=0; n=0; by=collections.Counter()
    for t in d["tasks"]:
        for s in t["samples"]:
            n+=1
            if s["rung"]==4: l4+=1; by[t["task"]]+=1
            if s["rung"]>=3: stm.append(s.get("body_stmts",0))
    print(f"  {tag}: L3+ {100*dg['per_sample_l3plus']:.1f}%  committed "
          f"{100*dg.get('committed_frac',0):.0f}%  body med {st.median(stm) if stm else 0:.0f} "
          f"mean {st.mean(stm) if stm else 0:.2f}  L4 {l4}/{n}")
    if by: print(f"    L4 tasks: {dict(by)}")
code(f"reports/code_mathcode_{S}.json", f"mathcode @{S}")
print("  rollout256 @151,238 : L3+ 78.8%  committed 41%  body med 1")
print("  codemix    @149,500 : L3+ 76.9%  committed 42%  body med 1  L4 14/320")
p=f"reports/math_mathcode_{S}.json"
if os.path.exists(p):
    d=json.load(open(p))
    print(f"\n  math @{S}: L3+ {100*d['per_sample_l3plus']:.2f}%  L4 {100*d['per_sample_l4']:.2f}%"
          f"  copied {d['copied_from_prompt']}  rel_err {d['median_rel_err']}")
    print("  math series (n=80): 1.2 / 3.8 / 11.2 / 5.0 / 11.25 — oscillating, no trend")
EOFPY
