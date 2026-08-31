#!/usr/bin/env bash
# BALANCER TEST — is the uniformity controller MANUFACTURING the twin experts?
#
#   bash run_balancer_test.sh
#
# THE CLEANEST A/B THIS PROJECT HAS HAD. Same checkpoint, same corpus, same
# every other flag. ONE variable: --router-bias-lr 0.0 instead of 1e-3.
#
# WHY. Two promotions have now ended with the new experts as ~90% twins of their
# parents, on curves that are asymptotic rather than slow:
#
#   grown48 (8,696 steps)  cos(gate) -> 0.909, exponential fit asymptote 0.893
#   growth_v2 (3,000 steps) cos(gate) -> 0.960, fit asymptote 0.937
#
# G2b concluded from that: "the model cannot find distinct work for 48 experts."
# But there is a competing explanation nobody had checked, because the knob was
# never exposed. The DeepSeek-V3 aux-loss-free balancer runs EVERY step:
#
#     target  = mean(counts)                      # UNIFORM
#     bias[i] += bias_lr * sign(target - count[i])
#
# Measured 2026-08-31 on both grown checkpoints, the accumulated bias reaches a
# spread COMPARABLE to the content logit spread:
#
#   step_0008696  content 2.83  bias 3.03  ratio 1.07
#   step_0003000  content 3.48  bias 2.48  ratio 0.71
#
# So roughly HALF of what decides which experts fire is a controller whose only
# goal is that they fire equally often. On a grown model whose new experts start
# as exact clones, forcing them to stay equally USED is a plausible mechanism for
# forcing them to stay equally TRAINED. Every leg this project has ever run used
# bias_lr=1e-3, including both promotions -- so this has never been tested.
#
# ⚠ EARLIER TODAY I claimed this ratio was 29x and that routing was therefore
# content-blind. That was WRONG: it assumed unit-norm router inputs, but the
# router reads an RMSNorm output (norm ~27, main.py:1116). Corrected to ~1x
# before this script was written. The balancer is a co-driver, not a blindfold.
#
# THE GATE, pre-registered. growth_v2 IS the balancer-on control from this exact
# checkpoint, and its fitted curve predicts cos(gate) ~0.944 at step 6,000.
#
#   cos(gate) at 6,000 well below 0.944, and cv RISING above ~0.4
#       -> the balancer was manufacturing the twins. G2b REOPENS: expert growth
#          was never given a chance to specialise, and the uniformity target is
#          the thing to fix, not the expert count.
#   cos(gate) ~0.944 with cv risen anyway
#       -> utilisation was free to diverge and the experts twinned regardless.
#          G2b is then established properly and Net2Wider is unambiguous.
#   cv > 1.0 with min% -> 0
#       -> experts are DYING, not specialising. Also a result: it would mean the
#          balancer is load-bearing for stability and cannot simply be removed.
set -uo pipefail
STOP=0
trap 'STOP=1; pkill -INT -P $$ 2>/dev/null; true' INT TERM
cd "$(dirname "$0")"
source ../venv-xpu/bin/activate
export TRITON_DEFAULT_BACKEND=intel
unset SYCL_CACHE_PERSISTENT PYTORCH_ALLOC_CONF
export PYTHONFAULTHANDLER=1 PYTHONUNBUFFERED=1
export SYCL_QUEUE_THREAD_POOL_SIZE=1
export ZE_SERIALIZE=2

SRC=checkpoints_growth_v2/step_0003000.pt
DIR=checkpoints_balancer_off
TEACHER=ByteDance/Ouro-2.6B-Thinking
FILES='data_teacher_code/shard_*.jsonl,data_teacher_math/shard_*.jsonl,data_teacher_v2/shard_*.jsonl,data_teacher_med/shard_*.jsonl'
STEPS=3000
LOG="logs/balancer_off_$(date +%Y%m%d_%H%M).log"
mkdir -p logs reports "$DIR"

if pgrep -f "python -u -m training\.(distill|sft)" >/dev/null; then
  echo "a trainer is already running"; exit 1; fi
[ -f "$SRC" ] || { echo "missing $SRC"; exit 1; }
[ -f "$DIR/step_0003000.pt" ] || cp "$SRC" "$DIR/step_0003000.pt"

echo "=== balancer OFF: 3000 -> $((3000+STEPS)) from growth_v2/step_0003000 ==="
echo "=== control = growth_v2 itself (balancer on, same base, same corpus) ==="
echo "=== log: $LOG ==="

python -u -m training.distill \
  --student-variant mythouro_distill_small \
  --student-device xpu:0 --teacher-device xpu:0 --teacher-id "$TEACHER" \
  --seq-len 1024 --micro-batch 4 --grad-accum 4 \
  --warmup-steps 500 --lr 1e-4 --min-lr 3e-5 --start-loops 4 \
  --depth-reg-coeff 0.3 --divergence rev_kl \
  --use-sandwich-norm --use-depth-aware-init --no-gradient-checkpointing \
  --router-bias-lr 0.0 \
  --teacher-mix-alpha 0.45 --rollout-len 64 --rollout-batch 8 --rollout-reuse 8 \
  --teacher-data-ratio 0.2 --teacher-data-files "$FILES" \
  --onpolicy-lambda 0.7 \
  --ckpt-dir "$DIR" \
  --ckpt-every-mins 15 --ckpt-milestone-every 500 --keep-last 8 \
  --num-workers 0 --trust-remote-code --log-every 50 \
  --total-steps $((3000+STEPS)) \
  > >(tee -a "$LOG") 2> >(tee -a "$LOG.err" >&2)

echo
echo "=== VERDICT ==="
python - <<'PY' 2>&1 | tee -a "$LOG"
import torch, glob, re, math, torch.nn.functional as F
P="recurrent.block.ffn.routed_experts."
def cosg(f):
    sd=torch.load(f,map_location="cpu",mmap=True,weights_only=False)["model"]
    E=sd["recurrent.block.ffn.router_bias"].shape[0]//2
    return sum(F.cosine_similarity(sd[f"{P}{i}.gate.weight"].float().flatten(),
                                   sd[f"{P}{i+E}.gate.weight"].float().flatten(),dim=0).item()
               for i in range(E))/E
A,k=0.9370,0.00037     # growth_v2 balancer-ON fit
print(f"  {'step':>6} {'cos(gate)':>10} {'balancer-ON pred':>18} {'delta':>9}")
last=None
for f in sorted(glob.glob("checkpoints_balancer_off/step_0*.pt")):
    s=int(re.search(r"step_(\d+)",f).group(1))
    if s % 500: continue
    c=cosg(f); pred=A+(1-A)*math.exp(-k*s)
    print(f"  {s:>6} {c:10.4f} {pred:18.4f} {c-pred:+9.4f}")
    last=(s,c,pred)
if last:
    s,c,pred=last
    print()
    if c < pred-0.02: print("  ⇒ BELOW the control curve. The balancer WAS suppressing")
    else:            print("  ⇒ NOT below the control curve.")
    print("     Read the cv trajectory in the log alongside this.")
PY
grep -ohE "cv=[0-9.]+ min=[0-9.]+%" "$LOG.err" | awk 'NR%10==1' | tail -12 | sed 's/^/    /'
