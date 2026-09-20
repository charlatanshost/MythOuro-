#!/usr/bin/env bash
# INSTRUCTION-PROMPT PROBE — what would rung 6's rollouts look like, at each α?
#
#   bash run_instruct_probe.sh [checkpoint]      # default: the annealed best, ~35 min
#
# WHY TONIGHT. Rung 6 (on-policy SFT) seeds rollouts with INSTRUCTION prompts
# instead of corpus slices, and the student generates the answer under teacher
# correction. Everything about that path is built except the seeding — and one
# design question decides whether it works: does the student's own answer to an
# instruction contain anything worth correcting, or is it noise?
#
# The data already leans one way. Under chat framing at 512 tokens the student
# was 69% adjacent-token degenerate (2026-09-09) — that IS rung 6's raw rollout
# at α=0. The α that was annealed to 0 for corpus continuation, where the
# student is healthy, may be needed for instruction answering, where it is not.
#
# This runs the existing prose probe on six instruction-shaped seeds in the
# exact ChatML the model was trained on, at α = 0 / 0.25 / 0.5 / 0.7 — so it
# shows the rollout rung 6 would produce at each teacher-mix, on the same
# text metrics as every other prose readout. No new code.
#
# READ IT LIKE THIS
#   α=0 clean (LOOPING ~0, distinct1 ~0.55+)     -> rung 6 can run at α=0.
#   α=0 degenerate, α=0.25-0.5 clean             -> rung 6 needs a small α back.
#                                                   That is a design input, not
#                                                   a reversal of the anneal.
#   degenerate at every α                        -> the student cannot answer
#                                                   even steered; rung 6 at this
#                                                   size is a harder bet.
# READ THE TEXT. The e.g. lines are the actual rollouts.
set -uo pipefail
cd "$(dirname "$0")"
source ../venv-xpu/bin/activate
export TRITON_DEFAULT_BACKEND=intel
unset SYCL_CACHE_PERSISTENT PYTORCH_ALLOC_CONF
export PYTHONFAULTHANDLER=1 PYTHONUNBUFFERED=1
export SYCL_QUEUE_THREAD_POOL_SIZE=1
export ZE_SERIALIZE=2
CK="${1:-checkpoints_anneal_lr/step_0003000.pt}"
[ -f "$CK" ] || { echo "missing $CK"; exit 1; }
if pgrep -f "[p]ython -u -m (training|tools)\." >/dev/null; then echo "card busy"; exit 1; fi
TAG=$(basename "$(dirname "$CK")" | sed 's/^checkpoints_//')_$(basename "$CK" .pt | sed 's/step_0*//')
D=/tmp/prose_probe/instruct_$TAG; rm -rf "$D"; mkdir -p "$D" reports logs; cp "$CK" "$D/step_0000000.pt"

S='<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
'
E='<|im_end|>
<|im_start|>assistant
'
python -u -m tools.onpolicy_rollout_probe --ckpt-dir "$D" \
  --student-device xpu:0 --teacher-device xpu:0 --teacher-id ByteDance/Ouro-2.6B-Thinking \
  --trust-remote-code --no-kv-cache --samples 5 \
  --seeds \
    "${S}Write a Python function that returns the sum of a list of numbers.${E}" \
    "${S}Write a Python function that returns True if a number is prime.${E}" \
    "${S}Write a Python function that reverses a string.${E}" \
    "${S}What are the common symptoms of type 2 diabetes?${E}" \
    "${S}What is ibuprofen used to treat, and what are its main risks?${E}" \
    "${S}How is a bacterial infection usually treated?${E}" \
  --json "reports/prose_instruct_${TAG}.json" 2>&1 | tee -a "logs/prose_instruct_${TAG}.log"
rm -rf "$D"
echo
echo "=== per-α summary (instruction prompts) ==="
python3 - "reports/prose_instruct_${TAG}.json" <<'PY'
import json, sys, statistics as st
from collections import Counter
d=json.load(open(sys.argv[1]))
print(f"  {'α':>5} {'top_sh':>7} {'dist1':>6} {'LOOP':>6}   (30 samples each)")
for al in ("0.0","0.25","0.5","0.7"):
    ts,ds,lo,n=[],[],0,0
    for a in d["seeds"].values():
        v=a.get(al,{}); ts+=v.get("top_share",[]); ds+=v.get("distinct1",[])
        for t in v.get("texts",[]):
            n+=1; L=[l.strip() for l in t.split("\n") if len(l.strip())>3]
            if L and max(Counter(L).values())>=3: lo+=1
    if ts: print(f"  {al:>5} {st.mean(ts):7.3f} {st.mean(ds):6.3f} {lo:3d}/{n}")
print("  reference — corpus-continuation seeds, same checkpoint, α=0: distinct1 0.546, LOOP 4/90")
print("  reference — the teacher alone on corpus seeds:              distinct1 0.615, LOOP 0/30")
PY
