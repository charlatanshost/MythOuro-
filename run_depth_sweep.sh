#!/usr/bin/env bash
# DEPTH SWEEP — did the accuracy wall lift? NO TRAINING REQUIRED.
#
#   bash run_depth_sweep.sh
#
# THE QUESTION. `looped_lm_landscape.md` §0.1: iterative-target supervision
# "extrapolates up to 24x beyond trained depth; final-only supervision causes
# accuracy WALLS". Our --n-loops sweep was FLAT at 4/6/8 — measured under
# final-only supervision, which is exactly the condition that predicts a wall at
# the trained depth.
#
# The model is now trained with exit_pdf (Ouro's own p(t|x)) and has a genuinely
# learned halt distribution: mean depth 2.49 -> 3.21/4, loop 0 abandoned, mass on
# loops 3-4, stable across three checkpoints. If the wall was the objective, the
# sweep should now RISE with depth. This costs one eval per point and no training.
#
# ⚠ THE COMPETING ACCOUNT, and it predicts the opposite. `recurrent.injection` is
# contractive: ρ(A) mean 0.310 before, 0.289 after — working per-loop supervision
# did NOT loosen it. At ρ≈0.29 the LINEAR term is 0.7% of initial by loop 4 and
# 0.005% by loop 8, so loops 5+ have almost nothing left to change and no
# objective produces extrapolation without also changing the contraction.
#
# ⚠ BUT ρ(A) COVERS ONLY THE LINEAR TERM. The Transformer term in the same update
# is not in that number, and the model demonstrably learned to put halt mass on
# loops 3-4 — so something there is doing work. The 2026-08-11 contractivity
# measurement was also taken on the BUGGED rung-3 leg, which supervised
# coda(prelude(x)) with the recurrent block bypassed, so it never actually tested
# per-loop supervision. This sweep is the readout that settles it.
#
# READ IT AS:
#   L3+/L4 RISE with n_loops   -> the wall was the objective; test-time depth is a
#                                 free capability lever and rung 5 reopens.
#   FLAT again                 -> contractivity is the binding constraint, not the
#                                 objective. Next lever is the contraction itself
#                                 (DeepLoop 2607.13491 scales residuals by visit
#                                 count; ours does not vary with loop at all).
#   FALLS with n_loops         -> the model is overfit to exactly 4; do not raise
#                                 depth at inference.
set -uo pipefail
cd "$(dirname "$0")"
source ../venv-xpu/bin/activate
CK="${CK:-checkpoints_exitpdf/step_0004200.pt}"
[ -f "$CK" ] || { echo "missing $CK"; exit 1; }
if pgrep -f "python -u -m training\.(distill|sft)" >/dev/null; then
  echo "a trainer is running"; exit 1; fi
mkdir -p reports logs

for L in 4 6 8; do
  OUT="reports/code_depth_${L}.json"
  if [ -f "$OUT" ]; then echo "  n_loops $L already done"; continue; fi
  echo "=== n_loops $L ==="
  python -u -m tools.code_eval -c "$CK" --device xpu:0 --n-loops "$L" \
    --samples 32 --temperature 0.4 --seed 1234 --repetition-penalty 1.15 \
    --json "$OUT" 2>&1 | tee -a "logs/depth_sweep_${L}.log"
done

python - <<'PY'
import json, os
print("\n" + "="*56)
print("  DEPTH SWEEP — trained at 4 loops, evaluated deeper")
print("="*56)
print(f"  {'n_loops':>8} {'L3+':>8} {'L0':>6} {'L4':>9}")
rows=[]
for L in (4,6,8):
    p=f"reports/code_depth_{L}.json"
    if not os.path.exists(p): continue
    d=json.load(open(p)); S=[s for t in d["tasks"] for s in t["samples"]]; n=len(S)
    l0=sum(1 for s in S if s.get("rung",0)==0)*100//n
    l4=sum(1 for s in S if s.get("rung",0)>=4)
    l3=d["diagnostics"]["per_sample_l3plus"]*100
    rows.append((L,l3,l0,l4)); print(f"  {L:>8} {l3:7.1f}% {l0:>5}% {l4:>4}/{n}")
if len(rows)>=3:
    a,b=rows[0][1],rows[-1][1]
    print()
    if b>a+5:   print("  ⇒ RISES with depth — the wall LIFTED. Rung 5 reopens.")
    elif b<a-5: print("  ⇒ FALLS with depth — overfit to 4 loops; do not raise at inference.")
    else:       print("  ⇒ FLAT — contractivity binds, not the objective. Next lever is")
                print("     the contraction itself (DeepLoop 2607.13491, residual scaling")
                print("     by visit count; ours does not vary with loop at all).")
PY
