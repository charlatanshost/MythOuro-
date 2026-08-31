#!/usr/bin/env bash
# GROWTH v2 — re-promote with ROUTER symmetry-breaking, then test whether the
# experts actually differentiate this time. ~3,000 steps, ~9.5h.
#
#   bash run_growth_v2.sh
#
# WHY REDO A PROMOTION WE ALREADY DID. The 2026-08-30 397M run is at a dead end,
# measured rather than guessed. Its new experts converged to their parents and
# STOPPED:
#
#   step        0     500    1000   2000   4000   6000   8000   8696
#   cos(gate) 1.000  0.996  0.982  0.960  0.930  0.915  0.910  0.909   <- asymptote
#   |down|    0.000  0.048  0.157  0.253  0.346  0.385  0.401  0.404   <- asymptote
#
# The last 696 steps moved cos by 0.0010; the first 1,000 moved it 0.018. This is
# converging, not progressing. MORE TOKENS WILL NOT FIX IT — that leg is 24
# trained experts plus 24 half-strength echoes, which is why the G2 readout came
# out flat (+3.4 pp against a 13.6 pp within-leg sd).
#
# ROOT CAUSE, in grow.py:319. Router rows were TILED — `w_src.repeat(...)` — so
# expert i and i+24 had IDENTICAL routing directions, attracted statistically
# identical token distributions, and gradient descent had no reason to
# specialise them. `--perturb-scale` never touched the router, only gate/up, and
# it was 0.0 anyway.
#
# THE FIX IS FREE. Perturbing the NEW router rows is still function-preserving,
# because the -100 sentinel blocks those experts from top-k at step 0 no matter
# which direction they point. Verified: 0 new-expert selections at step 0 both
# with and without perturbation.
#
# --router-perturb-scale 1.0 starts the new rows at cos ~0.70 to their parents —
# where the LAST run's router took 8,696 steps to drift on its own. Calibrated:
#   0.5 -> 0.89   0.75 -> 0.80   1.0 -> 0.70   1.5 -> 0.55
#
# THE GATE. This is a 3,000-step TEST, not a leg. Compare against the old run at
# the same step counts (the table above). Success is cos(gate) BELOW those
# numbers AND still falling at 3,000 — the old run was already at 0.9433 by then
# and flattening.
#   still falling  -> symmetry breaking works; continue into a full pour.
#   flat near 0.91 -> the ceiling is not initialisation. Do not grow again;
#                     the v5 post-mortem's "can't find distinct work for the
#                     experts" would then apply to 48 on this lineage too.
set -uo pipefail
STOP=0
trap 'STOP=1; pkill -INT -P $$ 2>/dev/null; true' INT TERM
cd "$(dirname "$0")"
source ../venv-xpu/bin/activate
# 48-expert crash-avoidance config — unchanged, and load-bearing. See run_grown48.sh.
export TRITON_DEFAULT_BACKEND=intel
unset SYCL_CACHE_PERSISTENT PYTORCH_ALLOC_CONF
export PYTHONFAULTHANDLER=1 PYTHONUNBUFFERED=1
export SYCL_QUEUE_THREAD_POOL_SIZE=1
export ZE_SERIALIZE=2

SRC=checkpoints_base/step_0157000.pt
DIR=checkpoints_growth_v2
TEACHER=ByteDance/Ouro-2.6B-Thinking
FILES='data_teacher_code/shard_*.jsonl,data_teacher_math/shard_*.jsonl,data_teacher_v2/shard_*.jsonl,data_teacher_med/shard_*.jsonl'
STEPS=3000
LOG="logs/growth_v2_$(date +%Y%m%d_%H%M).log"
mkdir -p logs reports "$DIR"

if pgrep -f "python -u -m training\.(distill|sft)" >/dev/null; then
  echo "a trainer is already running"; exit 1; fi
[ -f "$SRC" ] || { echo "missing $SRC"; exit 1; }

# ── promote, if not already done ──────────────────────────────────────────────
if [ ! -f "$DIR/step_0000000.pt" ]; then
  echo "=== promoting 24 -> 48 with router symmetry-breaking ==="
  python -u -m tools.grow_checkpoint --src "$SRC" --dst "$DIR/step_0000000.pt" \
    --expansion-factor 2 --router-perturb-scale 1.0 --perturb-scale 1e-3 \
    --n-decay-steps 500 2>&1 | tee -a "$LOG"
  python - <<'PY' 2>&1 | tee -a "$LOG"
import torch, torch.nn.functional as F
sd=torch.load("checkpoints_growth_v2/step_0000000.pt",map_location="cpu",
              mmap=True,weights_only=False)["model"]
W=sd["recurrent.block.ffn.router.weight"].float()
E=W.shape[0]//2
c=sum(F.cosine_similarity(W[i],W[i+E],dim=0).item() for i in range(E))/E
print(f"  promoted router cos(new, parent) = {c:.4f}  (old promotion: 1.0000)")
assert c < 0.85, "router symmetry NOT broken — check --router-perturb-scale"
print("  symmetry broken. proceeding.")
PY
fi

at=$(basename "$(ls -t $DIR/step_*.pt | head -1)" | sed 's/step_0*//; s/\.pt//'); at=${at:-0}
TARGET=$((at + STEPS))
echo "=== growth v2: $at -> $TARGET  (397M, router-perturbed) ==="
echo "=== log: $LOG ==="

python -u -m training.distill \
  --student-variant mythouro_distill_small \
  --student-device xpu:0 --teacher-device xpu:0 --teacher-id "$TEACHER" \
  --seq-len 1024 --micro-batch 4 --grad-accum 4 \
  --warmup-steps 500 --lr 1e-4 --min-lr 3e-5 --start-loops 4 \
  --depth-reg-coeff 0.3 --divergence rev_kl \
  --use-sandwich-norm --use-depth-aware-init --no-gradient-checkpointing \
  --teacher-mix-alpha 0.45 --rollout-len 64 --rollout-batch 8 --rollout-reuse 8 \
  --teacher-data-ratio 0.2 --teacher-data-files "$FILES" \
  --onpolicy-lambda 0.7 \
  --ckpt-dir "$DIR" \
  --ckpt-every-mins 15 --ckpt-milestone-every 500 --keep-last 20 \
  --num-workers 0 --trust-remote-code --log-every 50 \
  --total-steps "$TARGET" \
  > >(tee -a "$LOG") 2> >(tee -a "$LOG.err" >&2)

echo
echo "=== DIFFERENTIATION vs the old promotion (the gate) ==="
python - <<'PY' 2>&1 | tee -a "$LOG"
import torch, glob, re, torch.nn.functional as F
OLD={0:1.0000,500:0.9963,1000:0.9822,2000:0.9598,3000:0.9433}
P="recurrent.block.ffn.routed_experts."
print(f"  {'step':>6} {'v2 cos(gate)':>13} {'old':>8} {'delta':>8}")
for f in sorted(glob.glob("checkpoints_growth_v2/step_0*.pt")):
    s=int(re.search(r"step_(\d+)",f).group(1))
    if s % 500: continue
    sd=torch.load(f,map_location="cpu",mmap=True,weights_only=False)["model"]
    E=sd["recurrent.block.ffn.router_bias"].shape[0]//2
    c=sum(F.cosine_similarity(sd[f"{P}{i}.gate.weight"].float().flatten(),
                              sd[f"{P}{i+E}.gate.weight"].float().flatten(),dim=0).item()
          for i in range(E))/E
    o=OLD.get(s); d=f"{c-o:+.4f}" if o else "   —"
    print(f"  {s:>6} {c:13.4f} {o if o else float('nan'):8.4f} {d:>8}")
print("\n  PASS = below the old curve AND still falling at 3,000.")
print("  FAIL = flat near 0.91 -> initialisation was not the ceiling; do not grow again.")
PY
