#!/usr/bin/env bash
# THROUGHPUT PROBE — four PROFILE runs of the production recipe, one knob each.
#
#   bash run_throughput_probe.sh          # ~15 min total, saves nothing
#
# WHY NOW. The token curve is the roadmap's go/no-go and it is ~10 nights of
# card time. Anything that cuts the per-step cost is worth an hour BEFORE
# committing those nights, and two candidates have never been measured on this
# recipe.
#
# MEASURED, 2026-09-12, profile with OPT-B applied:
#   rollout 69.7% | backward 20.9% | student_fwd 7.7% | teacher_fwd 1.1%
#
# ⚠ The 1.1% teacher share is a REGION BOUNDARY, not the teacher being cheap.
# `generate_rollout` runs the teacher at every generated token
# (teacher_mix_alpha, α·softmax(teacher/T) + (1−α)·softmax(student/T)) and bills
# that time to `rollout`. Variant C isolates it.
#
# THE VARIANTS
#   A baseline      the production recipe, for a same-session reference
#   B mb4/ga4       IDENTICAL tokens/step (16,384), 4x fewer kernel launches.
#                   ⚠ NOT "mathematically identical" — corrected 2026-09-13.
#                   load_balance_loss is E·Σ f_i·P_i with f_i and P_i averaged
#                   over the MICRO-BATCH, so a product of two batch means is not
#                   linear in the batch: 8 accumulations over N=2,048 differ from
#                   4 over N=4,096 on the auxiliary gradient, and the router-bias
#                   update cadence moves with them. Main loss unaffected, MoE
#                   ROUTING is not. Still cheap and still worth taking, but it
#                   is a quality decision of its own, not a free win.
#
#                   ⚠ docs/training_throughput.md lists "--micro-batch 16
#                   --grad-accum 1" as the cheapest untested idea. DO NOT RUN
#                   THAT. It was written 2026-07-30, BEFORE exit_pdf made the
#                   loss per-loop. run_exitpdf.sh:44 measured the per-loop
#                   distillation loss at ~19.3 GB at micro-batch 8 with all K
#                   autograd graphs live — that is what caused the "NotPresent /
#                   PDE / Write" page fault, an OOM in disguise. mb2 is ~4.8 GB,
#                   so scaling linearly mb16 is ~38 GB before the 4.8 GB teacher
#                   and ~4.2 GB of weights/optimiser. It will not fit in 48 GB.
#                   mb4 (~9.6 GB) is the largest step worth probing.
#   C alpha 0       DIAGNOSTIC ONLY. Skips the teacher inside generation, so the
#                   drop in `rollout` measures the teacher's true share of it.
#                   ⚠ NOT a training recommendation: α is the un-collapse lever
#                   from docs/onpolicy_plan.md that made on-policy viable from a
#                   degenerate checkpoint. Using it in a throw-away profile is
#                   free; lowering it in training is a quality decision.
#   D rollout-len 32  generation is O(L²) by design (use_kv_cache=False, the ACT
#                   early-exit KL finding), so halving L should cut generation
#                   ~3x. Also alters the on-policy signal → quality decision.
#
# Each run: 5 warmup + 10 profiled steps, then EXITS WITHOUT SAVING. Read the
# SHARES — the profiler syncs around every region, so its totals are pessimistic.
set -uo pipefail
trap 'pkill -INT -P $$ 2>/dev/null; true' INT TERM
cd "$(dirname "$0")"
source ../venv-xpu/bin/activate
export TRITON_DEFAULT_BACKEND=intel
unset SYCL_CACHE_PERSISTENT PYTORCH_ALLOC_CONF
export PYTHONFAULTHANDLER=1 PYTHONUNBUFFERED=1
export SYCL_QUEUE_THREAD_POOL_SIZE=1
export ZE_SERIALIZE=2

SRC=checkpoints_exitpdf/step_0007200.pt
TEACHER=ByteDance/Ouro-2.6B-Thinking
FILES="data_teacher_code/shard_*.jsonl,data_teacher_math/shard_*.jsonl,data_teacher_v2/shard_*.jsonl,data_teacher_med/shard_*.jsonl"
DIR=checkpoints_tput_probe
STAMP=$(date +%Y%m%d_%H%M)
mkdir -p logs "$DIR"

if pgrep -f "python -u -m training\.(distill|sft)" >/dev/null; then
  echo "a trainer is running — stop it first"; exit 1; fi
[ -f "$SRC" ] || { echo "missing $SRC"; exit 1; }

# OPT-B is applied to distill.py but NO --teacher-logit-cache is passed here, so
# every offline micro-step takes the dense-teacher path (sparse_pack = None).
# These numbers are therefore comparable to the pre-B production recipe.
rm -rf "$DIR"; mkdir -p "$DIR"
cp "$SRC" "$DIR/step_0000000.pt"
python - <<'PY'
import torch, os
p="checkpoints_tput_probe/step_0000000.pt"
ck=torch.load(p,map_location="cpu",weights_only=False); ck["step"]=0
torch.save(ck,p+".tmp"); os.replace(p+".tmp",p); print("  seeded at step 0")
PY

run_variant() {   # run_variant <tag> <extra flags...>
  local tag="$1"; shift
  local log="logs/tput_${tag}_${STAMP}.log"
  echo
  echo "=================================================================="
  echo "  VARIANT $tag   extra: $*"
  echo "=================================================================="
  python -u -m training.distill \
    --student-variant mythouro_distill_tiny \
    --student-device xpu:0 --teacher-device xpu:0 --teacher-id "$TEACHER" \
    --seq-len 1024 \
    --warmup-steps 500 --lr 1e-4 --min-lr 3e-5 --start-loops 4 \
    --loop-loss-weighting exit_pdf --depth-reg-coeff 0.1 \
    --divergence rev_kl \
    --use-sandwich-norm --use-depth-aware-init \
    --teacher-mix-alpha 0.45 --rollout-len 64 --rollout-batch 8 --rollout-reuse 8 \
    --teacher-data-ratio 0.2 --teacher-data-files "$FILES" \
    --onpolicy-lambda 0.7 \
    --ckpt-dir "$DIR" --ckpt-every-mins 999 --ckpt-milestone-every 100000 \
    --num-workers 0 --trust-remote-code --log-every 25 \
    --profile-steps 10 --profile-warmup 5 --total-steps 100000 \
    "$@" \
    > >(tee -a "$log") 2>&1
  echo "  -> $log"
}

#                      micro-batch / grad-accum keep tokens/step at 16,384
run_variant A_baseline   --micro-batch 2  --grad-accum 8
run_variant B_mb4        --micro-batch 4  --grad-accum 4
run_variant C_alpha0     --micro-batch 2  --grad-accum 8 --teacher-mix-alpha 0.0
run_variant D_rlen32     --micro-batch 2  --grad-accum 8 --rollout-len 32

echo
echo "=================================================================="
echo "  SUMMARY"
echo "=================================================================="
python3 - logs/tput_*_${STAMP}.log <<'PY'
import re, sys, os
print(f"  {'variant':12s} {'ms/step':>9} {'rollout':>9} {'backward':>9} {'stud_fwd':>9} {'teacher':>8}")
base=None
for p in sorted(sys.argv[1:]):
    tag=os.path.basename(p).split("_",1)[1].rsplit("_",2)[0]
    txt=open(p, errors="ignore").read()
    m=re.search(r"STEP PROFILE over \d+ steps \(([\d.]+) ms/step", txt)
    if not m:
        print(f"  {tag:12s}  no profile — check {p}"); continue
    tot=float(m.group(1))
    reg={}
    for r in ("rollout","backward","student_fwd","teacher_fwd"):
        mm=re.search(rf"{r}\s+([\d.]+)\s+([\d.]+)%", txt)
        reg[r]=(float(mm.group(1)), float(mm.group(2))) if mm else (0.0,0.0)
    if base is None: base=tot
    print(f"  {tag:12s} {tot:9.0f} "
          f"{reg['rollout'][1]:8.1f}% {reg['backward'][1]:8.1f}% "
          f"{reg['student_fwd'][1]:8.1f}% {reg['teacher_fwd'][1]:7.1f}%"
          + (f"   {base/tot:.2f}x" if base else ""))
print()
print("  READ IT THIS WAY")
print("   B_mb4 faster   -> take it. Identical optimisation, no quality decision.")
print("   C_alpha0 much faster -> the teacher IS most of `rollout`, and a cheaper")
print("     mixing schedule is the lever. NOT a licence to train at alpha 0.")
print("   C_alpha0 barely moves -> `rollout` is genuine student decode, and the")
print("     O(L^2) KV-cache fix (needs a KL equivalence gate) is the root attack.")
print("   D_rlen32 faster -> real, but it changes the on-policy signal; gate on a")
print("     quality leg before adopting, the same rule OPT-B is held to.")
print()
print("  ⚠ Profiler totals are pessimistic (per-region syncs). Compare SHARES and")
print("    ratios between variants, not any single ms/step against production.")
PY
