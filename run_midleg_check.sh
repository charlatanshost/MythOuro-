#!/usr/bin/env bash
# MID-LEG CHECK — was the multi-statement collapse a dip or a reversal? ~26 min.
#
#   bash run_midleg_check.sh
#
# Leg 3 (157,238 -> 163,238, SAME corpus and config as leg 2) reversed the metric
# that had been climbing:
#
#   checkpoint          L3+    committed  >=2 stmt  mean  L4/320
#   base    @143,500   76.6      25%         3%     1.03    5
#   mathcode@157,238   54.7      42%        28%     1.35   26
#   mathcode@163,238   77.5      64%         4%     1.05   19
#
# Multi-statement solutions went 28% -> 4%, essentially back to baseline, while
# L3+ and committed hit their best values ever. The model returned to writing ONE
# statement and finishing it.
#
# ⚠ ONE ENDPOINT. Single endpoints have misled this project three times — the
# 2026-08-17 U-shape (75.0/41.2/53.8/60.0/66.2/68.8), the "phase transition", and
# the 82.5% "record" that was 76.6 at proper power. Two mid-leg points say which
# this is:
#   ~28% at 159,500 and 161,000  -> collapse is LATE. Over-training on unchanged
#       data; the fix is new sources, and there is a usable checkpoint mid-leg.
#   already ~4% at 159,500       -> reversal began immediately; the corpus stopped
#       teaching length early and more steps on it actively undo the gain.
#
# Read >=2 stmt and mean body_stmts. NOT L3+ — it is high at both ends of this
# leg and low in the middle, which is exactly the U-shape signature.
set -uo pipefail
trap 'pkill -INT -P $$ 2>/dev/null; true' INT TERM
cd "$(dirname "$0")"
source ../venv-xpu/bin/activate
export SYCL_CACHE_PERSISTENT=1 PYTORCH_ALLOC_CONF=expandable_segments:True TRITON_DEFAULT_BACKEND=intel
mkdir -p logs reports
if pgrep -f "training[.](distill|sft)" >/dev/null; then echo "a trainer is running"; exit 1; fi

LOG="logs/midleg_$(date +%Y%m%d_%H%M).log"
{
for S in 159500 161000; do
  C="checkpoints_mathcode/step_0$S.pt"
  [ -f "$C" ] || { echo "MISSING $C"; continue; }
  echo; echo "--- code $S ---"
  python -u -m tools.code_eval -c "$C" --device xpu:0 --samples 32 \
    --temperature 0.4 --seed 1234 --repetition-penalty 1.15 \
    --json "reports/code_mathcode_${S}.json"
done

# PROSE regressed too — the acronym salad returned on the bacterial seed at
# 163,238 after being absent at both 140,000 and 157,238. One rollout probe
# mid-leg says whether prose turned at the same point as solution length.
echo; echo "--- rollout probe 160,000 (prose) ---"
python -u -m tools.onpolicy_rollout_probe --ckpt-dir checkpoints_mathcode \
  --student-device xpu:0 --teacher-device xpu:0 \
  --teacher-id ByteDance/Ouro-2.6B-Thinking --trust-remote-code \
  --no-kv-cache --samples 5 \
  --json reports/probe_mathcode_160000.json | tee reports/probe_mathcode_160000.txt
} 2>&1 | tee "$LOG"

echo
echo "=== SOLUTION LENGTH ACROSS LEG 3 ==="
python - <<'EOFPY'
import json,os,statistics as st
rows=[(143500,"reports/code_n32_143500.json","base"),
      (157238,"reports/code_mathcode_157238.json","leg2 end"),
      (159500,"reports/code_mathcode_159500.json","leg3 +2.3k"),
      (161000,"reports/code_mathcode_161000.json","leg3 +3.8k"),
      (163238,"reports/code_mathcode_163238.json","leg3 end")]
print(f"  {'step':>8}{'':>12}{'>=2 stmt':>10}{'mean':>7}{'commit':>8}{'L4':>9}")
for s,p,lbl in rows:
    if not os.path.exists(p): continue
    d=json.load(open(p)); dg=d["diagnostics"]; stm=[]; l4=0; n=0
    for t in d["tasks"]:
        for x in t["samples"]:
            n+=1
            if x["rung"]==4: l4+=1
            if x["rung"]>=3: stm.append(x.get("body_stmts",0))
    ge2=100*sum(1 for v in stm if v>=2)/len(stm) if stm else 0
    print(f"  {s:>8}{lbl:>12}{ge2:>9.0f}%{st.mean(stm) if stm else 0:>7.2f}"
          f"{100*dg.get('committed_frac',0):>7.0f}%{l4:>6}/{n}")
print("\n  monotone fall -> the corpus stops teaching length and more steps undo it")
print("  high-then-drop -> late collapse; a mid-leg checkpoint is the one to keep")

# prose: where did the salad come back?
import re, statistics as _st
def salad(t): return len(re.findall(r"(?:\b[A-Z]\.){4,}",t))+len(re.findall(r"(?:\b[A-Z]\b[ .]){5,}",t))
print("\n  PROSE (alpha=0.0):")
for lbl,p in [("157,238","reports/probe_mathcode_157238.json"),
              ("160,000","reports/probe_mathcode_160000.json"),
              ("163,238","reports/probe_163238.json")]:
    if not os.path.exists(p): continue
    d=json.load(open(p))
    ts=[_st.mean(v["0.0"]["top_share"]) for v in d["seeds"].values() if "0.0" in v]
    sl=sum(salad(x) for v in d["seeds"].values() if "0.0" in v for x in v["0.0"]["texts"])
    print(f"    {lbl}  mean top_share {_st.mean(ts):.3f}   salad hits {sl}")
print("    157,238 was 0.150 / 0 hits;  163,238 was 0.242 / 1 hit")
EOFPY
