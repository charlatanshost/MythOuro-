#!/usr/bin/env bash
# THE CEILING — score the TEACHER on the student's own instruments.
#
#   bash run_teacher_readout.sh          # ~15 min code eval + ~35 min prose probe
#
# WHY. Two token curves (278M, then 240M-activated after Net2Wider) went flat
# at the SAME place — distinct1 ~0.56, halt ~2.75, L3+ ~68-75% — while soft-KL
# to the teacher kept falling across all six legs (0.868 -> 0.787). The student
# keeps matching the teacher better on the training mix and the probes do not
# move. Nobody has ever measured what Ouro-2.6B-Thinking itself scores on these
# exact prompts with these exact settings. Until that number exists, "flat"
# cannot be read:
#
#   teacher near the student   -> the student is AT THE CEILING. Pouring,
#                                 widening and depth are spent on an objective
#                                 with nothing left to give. The lever is a
#                                 stronger teacher (or a different objective).
#   teacher well above         -> the student is BELOW the ceiling and something
#                                 else binds: the probe prompts vs the training
#                                 mix, the LR schedule, depth. Worth finding.
#
# HOW THE NUMBERS STAY COMPARABLE
#   code   code_eval --hf-model: the teacher is loaded through
#          load_distillation_teacher and sampled through the SAME generate()
#          loop as every checkpoint — T=0.4, penalty 1.15, seed 1234, 96 tokens,
#          bare framing, EOS break. Only the model behind the logits differs.
#   prose  onpolicy_rollout_probe --alphas 1.0: the rollout mix is
#          1.0*softmax(teacher/T) + 0.0*softmax(student/T) — pure teacher
#          samples on the same six seeds, scored by the same _rep_numbers.
#          A student checkpoint still has to load (it is in the call) but
#          contributes nothing at alpha=1. halt_depth in that output is the
#          STUDENT's and is meaningless here; ignore it.
#
# ⚠ Bare framing is the student's native framing, not the teacher's. Ouro is a
# chat model; scoring it on raw `def f(x):` continuations may UNDERSTATE it.
# That is the right comparison anyway — it is the framing the student is scored
# in — but if the teacher reads low, run it once more with --chat-template
# --extract before concluding the ceiling is low.
set -uo pipefail
cd "$(dirname "$0")"
source ../venv-xpu/bin/activate
export TRITON_DEFAULT_BACKEND=intel
unset SYCL_CACHE_PERSISTENT PYTORCH_ALLOC_CONF
export PYTHONFAULTHANDLER=1 PYTHONUNBUFFERED=1
export SYCL_QUEUE_THREAD_POOL_SIZE=1
export ZE_SERIALIZE=2
TEACHER="${TEACHER:-ByteDance/Ouro-2.6B-Thinking}"
TAG="${TAG:-}"                      # report-name suffix. REQUIRED when TEACHER is
                                    # overridden, so the archived 2.6B ceiling
                                    # (reports/code_teacher.json — the number every
                                    # "ceiling vs plateau" call rests on) is never
                                    # overwritten by a different model.
CODE_ONLY="${CODE_ONLY:-0}"         # 1 = skip the 35-min prose stage. Use this when
                                    # the question is only "how good is this teacher
                                    # at code", e.g. sizing a throughput trade.
if [ "$TEACHER" != "ByteDance/Ouro-2.6B-Thinking" ] && [ -z "$TAG" ]; then
  echo "TEACHER is overridden ($TEACHER) but TAG is empty."
  echo "That would overwrite reports/code_teacher.json, the archived 2.6B ceiling."
  echo "Pass a suffix, e.g.:  TEACHER=$TEACHER TAG=_1p4b bash $0"; exit 1; fi
CODE_JSON="reports/code_teacher${TAG}.json"
PROSE_JSON="reports/prose_teacher${TAG}.json"
export CODE_JSON PROSE_JSON TEACHER
ANY_CKPT="${ANY_CKPT:-checkpoints_wide/step_0009000.pt}"   # loaded, contributes 0 at alpha=1
mkdir -p reports logs /tmp/prose_probe
if pgrep -f "[p]ython -u -m training\.(distill|sft)" >/dev/null; then
  echo "a trainer is running — stop it first"; exit 1; fi

echo "=== 1/2  CODE: teacher through code_eval, bare framing, n=320 ==="
python -u -m tools.code_eval --hf-model "$TEACHER" --device xpu:0 \
  --samples 32 --temperature 0.4 --seed 1234 --repetition-penalty 1.15 \
  --max-new 96 --json "$CODE_JSON" 2>&1 | tee -a "logs/eval_teacher${TAG}.log"

echo
if [ "$CODE_ONLY" = "1" ]; then
  echo "=== 2/2  PROSE: SKIPPED (CODE_ONLY=1) ==="
else
echo "=== 2/2  PROSE: teacher through the prose probe at alpha=1.0, six seeds ==="
D=/tmp/prose_probe/teacher; rm -rf "$D"; mkdir -p "$D"; cp "$ANY_CKPT" "$D/step_0000000.pt"
python -u -m tools.onpolicy_rollout_probe --ckpt-dir "$D" \
  --student-device xpu:0 --teacher-device xpu:0 --teacher-id "$TEACHER" \
  --trust-remote-code --no-kv-cache --samples 5 --alphas 1.0 \
  --json "$PROSE_JSON" 2>&1 | tee -a "logs/prose_teacher${TAG}.log"
rm -rf "$D"
fi

echo
echo "=================================================================="
echo "  THE CEILING vs THE PLATEAU"
echo "=================================================================="
python3 - <<'PY'
import json, os, statistics as st
from collections import Counter
def code(p):
    d=json.load(open(p)); S=[s for t in d["tasks"] for s in t["samples"]]; n=len(S)
    r=lambda f: sum(1 for s in S if f(int(s.get("rung",0))))/n
    return r(lambda x:x==0), r(lambda x:x>=3), r(lambda x:x>=4)
def prose(p, alpha):
    d=json.load(open(p)); ts,ds,lo=[],[],0
    for a in d["seeds"].values():
        v=a.get(alpha,{}); ts+=v.get("top_share",[]); ds+=v.get("distinct1",[])
        for t in v.get("texts",[]):
            L=[l.strip() for l in t.split("\n") if len(l.strip())>3]
            if L and max(Counter(L).values())>=3: lo+=1
    return st.mean(ts), st.mean(ds), lo
CJ=os.environ["CODE_JSON"]; PJ=os.environ["PROSE_JSON"]
NAME="TEACHER  "+os.environ["TEACHER"].split("/")[-1].replace("-Thinking","")
rows=[]
if os.path.exists(CJ):
    l0,l3,l4=code(CJ); rows.append((NAME, l0,l3,l4))
print(f"  {'':22s} {'L0':>6} {'L3+':>7} {'L4':>6}")
for lbl,l0,l3,l4 in rows: print(f"  {lbl:22s} {l0:5.1%} {l3:6.1%} {l4:5.1%}")
print(f"  {'student wide pt3 (3ck)':22s} {'7.7%':>6} {'68.2%':>7} {'4.5%':>6}")
print(f"  {'student LARGE pt3 (3ck)':22s} {'71.5%':>6} {'16.8%':>7} {'0.0%':>6}   <- live lineage, 139M tok")
print( "      ⚠ large pt3's L3+ is INDENTATION, not computation (2026-09-26): 52%% of")
print( "        completions start with an indent, 1%% of its L3 returns a value. Compare")
print( "        this teacher's L4 and committed, not its L3+, when sizing the trade.")
print(f"  {'student 278M pt3 (1ck)':22s} {'6.6%':>6} {'75.0%':>7} {'5.0%':>6}")
print()
if os.path.exists(PJ):
    t,d,lo=prose(PJ,"1.0")
    print(f"  {'':22s} {'top_sh':>7} {'dist1':>6} {'LOOP':>6}")
    print(f"  {NAME:22s} {t:7.3f} {d:6.3f} {lo:3d}/30")
    print(f"  {'student wide pt3':22s} {'0.098':>7} {'0.559':>6} {'4/90':>6}")
    print(f"  {'student 278M pt3':22s} {'0.098':>7} {'0.556':>6} {'4/90':>6}")
print()
print("  READ THE TEXTS in both jsons. A teacher that scores near the student on")
print("  these prompts is the ceiling; one that scores well above means the")
print("  student is not there yet and the question is why.")
PY
