#!/usr/bin/env bash
# REFERENCE MODELS — is irrelevant output normal for ~300M params, or normal
# for ~3B tokens?
#
#   bash run_reference_eval.sh          # ~45-60 min, three models
#
# THE QUESTION (owner, 2026-09-22). The smallest model he has used is Bonsai
# 1B; he has no reference for whether a 278M-700M model *should* stay on the
# question. Every other small model he has tried (Mistral-class) wanders. So:
# does our irrelevance come from the size, or from the token budget?
#
# THE MODELS, all instruction-tuned, all trained on 100x-1000x our tokens:
#   SmolLM2-360M-Instruct   362M   ~4T tokens
#   Qwen2.5-0.5B-Instruct   494M   ~18T tokens
#   gemma-3-270m-it         270M   ~6T tokens
# Ours: 278M at 2.85B tokens (wide lineage), 695M at 29M (fresh lineage).
#
# ⚠ CODE LADDER ONLY. L0/L3+/L4 grade EXECUTABLE PYTHON, so they compare across
# tokenizers exactly. top_share/distinct1 count token TYPES and do not compare —
# do not run the prose probe on a foreign tokenizer and put the number in a
# table next to ours.
#
# ⚠ BARE FRAMING is our native framing and these are chat models, so this
# UNDERSTATES them. That is the honest direction for the comparison: if a
# 360M model beats us on our own home ground, the gap is real. Chat-framed
# numbers for them would be higher still.
#
# ⚠ These download from HF on first run (~0.5-2 GB each).
set -uo pipefail
cd "$(dirname "$0")"
source ../venv-xpu/bin/activate
export TRITON_DEFAULT_BACKEND=intel
unset SYCL_CACHE_PERSISTENT PYTORCH_ALLOC_CONF
export PYTHONFAULTHANDLER=1 PYTHONUNBUFFERED=1
export SYCL_QUEUE_THREAD_POOL_SIZE=1
export ZE_SERIALIZE=2
mkdir -p reports logs
if pgrep -f "[p]ython -u -m training\.(distill|sft)" >/dev/null; then echo "a trainer is running"; exit 1; fi

for M in HuggingFaceTB/SmolLM2-360M-Instruct Qwen/Qwen2.5-0.5B-Instruct google/gemma-3-270m-it; do
  TAG=$(basename "$M" | tr 'A-Z' 'a-z')
  OUT="reports/code_ref_${TAG}.json"
  if [ -f "$OUT" ]; then echo "=== $M — already measured, skipping ==="; continue; fi
  echo; echo "=== $M ==="
  python -u -m tools.code_eval --hf-model "$M" --reference --device xpu:0 \
    --samples 32 --temperature 0.4 --seed 1234 --repetition-penalty 1.15 \
    --max-new 96 --json "$OUT" 2>&1 | tee -a "logs/eval_ref_${TAG}.log" | grep -E "reference model|per-sample L3|COMMITTED" || true
done

echo
echo "=================================================================="
echo "  SIZE vs TOKENS — code ladder, bare framing, n=320, identical settings"
echo "=================================================================="
python3 - <<'PY'
import json, os, glob, statistics as st
def sc(p):
    d=json.load(open(p)); S=[s for t in d["tasks"] for s in t["samples"]]; n=len(S)
    r=lambda f: sum(1 for s in S if f(int(s.get("rung",0))))/n
    return r(lambda x:x==0), r(lambda x:x>=3), r(lambda x:x>=4)
def mean(paths):
    o=[sc(p) for p in paths if os.path.exists(p)]
    return tuple(st.mean(x) for x in zip(*o)) if o else None
rows=[("SmolLM2-360M-Instruct","362M","~4T",   ["reports/code_ref_smollm2-360m-instruct.json"]),
      ("Qwen2.5-0.5B-Instruct","494M","~18T",  ["reports/code_ref_qwen2.5-0.5b-instruct.json"]),
      ("gemma-3-270m-it",      "270M","~6T",   ["reports/code_ref_gemma-3-270m-it.json"]),
      ("OURS wide (278M tot)", "240M act","2.85B", sorted(glob.glob("reports/code_wide_leg3_*.json"))),
      ("OURS anneal (best)",   "240M act","2.9B",  sorted(glob.glob("reports/code_anneal_lr_*.json"))),
      ("OURS large (fresh)",   "695M act","29M",   sorted(glob.glob("reports/code_large_leg1_*.json"))),
      ("TEACHER Ouro-2.6B",    "2.6B","?",      ["reports/code_teacher.json"])]
print(f"  {'model':24s} {'active':>9} {'tokens':>7} {'L0':>7} {'L3+':>7} {'L4':>7}")
for lbl,sz,tk,paths in rows:
    m=mean(paths)
    if not m: print(f"  {lbl:24s} {sz:>9} {tk:>7}   (not measured)"); continue
    print(f"  {lbl:24s} {sz:>9} {tk:>7} {m[0]:6.1%} {m[1]:6.1%} {m[2]:6.1%}")
print()
print("  L4 is correctness. If the 270-500M references land far above ours, the")
print("  gap is TOKENS, not size. If they land near ours, it is size.")
PY
