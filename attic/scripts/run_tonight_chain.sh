#!/usr/bin/env bash
# Unattended chain: finish the medical leg, probe it, then run the λ=0.7 CONTROL
# branch — so the card is never idle and both results are waiting when you're home.
#
#   bash run_tonight_chain.sh
#
# Phase 1  medical leg 60,154 -> 64,000  (~11.1 h)   in checkpoints_onpolicy_fixed
# Phase 2  probes on 64,000              (~45 min)
# Phase 3  λ=0.7 CONTROL 64,000 -> 66,000 (~5.8 h)   in checkpoints_lambda07
# Phase 4  probe the control              (~35 min)
#
# WHY A SEPARATE CKPT DIR FOR PHASE 3: the λ experiment BRANCHES. Both arms must
# start from the SAME 64,000 checkpoint, so the control cannot simply continue in
# checkpoints_onpolicy_fixed — that would leave the λ=0.4 arm starting from 66,000
# and the comparison would be meaningless. Each arm gets its own dir seeded with a
# copy of step_0064000.pt. Tomorrow's λ=0.4 arm mirrors this into
# checkpoints_lambda04.
set -uo pipefail
# ── SIGNAL FORWARDING (2026-08-15) ────────────────────────────────────────────
# Without this the WRAPPER dies on Ctrl-C while the python child survives,
# orphaned, still holding the GPU. Observed in production on 2026-08-15: the user
# pressed Ctrl-C, reports/harvest_chat_FAILED was written at 22:26, and the
# harvest kept generating until it was signalled by PID. "Ctrl-C did nothing" was
# true from the terminal and false from the GPU. Reproduced in a toy harness:
# under the old pattern the child was still alive after a process-group SIGINT.
# The children here handle SIGINT COOPERATIVELY (flag, finish the step, flush),
# so they need the signal delivered and then time — not a dead parent.
trap 'pkill -INT -P $$ 2>/dev/null; true' INT TERM
cd "$(dirname "$0")"
source ../venv-xpu/bin/activate
export SYCL_CACHE_PERSISTENT=1 PYTORCH_ALLOC_CONF=expandable_segments:True TRITON_DEFAULT_BACKEND=intel

MAIN=checkpoints_onpolicy_fixed
BRANCH=checkpoints_lambda07
TEACHER=ByteDance/Ouro-2.6B-Thinking
FILES='data_teacher_v2/shard_*.jsonl,data_teacher_med/shard_*.jsonl'
OK=reports/chain_DONE; FAIL=reports/chain_FAILED
rm -f "$OK" "$FAIL"
trap '[ -f "$OK" ] || echo "incomplete exit=$? $(date)" > "$FAIL"' EXIT

# ARRAY, not a string: `$(fn)` unquoted would word-split AND glob-expand. The
# teacher-data pattern only survives that today because the comma makes it match
# no file — drop the comma and bash would expand it into filenames and silently
# mangle the command line. An array passes each argument through verbatim.
# Shared so the two legs differ ONLY in --onpolicy-lambda, which is the point of
# a control.
COMMON=(
  --student-variant mythouro_distill_tiny
  --student-device xpu:0 --teacher-device xpu:0 --teacher-id "$TEACHER"
  --seq-len 1024 --micro-batch 8 --grad-accum 2
  --warmup-steps 500 --lr 1e-4 --min-lr 3e-5
  --depth-reg-coeff 0.3 --divergence rev_kl
  --use-sandwich-norm --use-depth-aware-init
  --teacher-mix-alpha 0.5 --rollout-len 64 --rollout-batch 32 --rollout-reuse 2
  --teacher-data-ratio 0.2 --teacher-data-files "$FILES"
  --ckpt-every-mins 15 --ckpt-milestone-every 2000 --keep-last 5
  --num-workers 0 --trust-remote-code --log-every 5
)

step_of() { basename "$1" | sed 's/step_0*//; s/\.pt//'; }
latest()  { ls -t "$1"/step_*.pt 2>/dev/null | head -1; }

echo "=== [1/4] medical leg -> 64,000 (λ=0.7) ==="
python -u -m training.distill "${COMMON[@]}" \
  --onpolicy-lambda 0.7 --total-steps 64000 --ckpt-dir "$MAIN" || true

# Verify by ARTIFACT — torch/XPU can crash on teardown AFTER succeeding.
S=$(step_of "$(latest $MAIN)")
if [ "$S" -lt 64000 ]; then
  echo "=== leg stopped early at $S — no probe, no control ==="
  echo "leg early stop at $S $(date)" > "$FAIL"; exit 1
fi
echo "=== reached $S ==="

echo "=== [2/4] probes on $S ==="
python -u -m tools.onpolicy_rollout_probe --ckpt-dir "$MAIN" \
  --student-device xpu:0 --teacher-device xpu:0 --teacher-id "$TEACHER" \
  --trust-remote-code --no-kv-cache --samples 5 \
  | tee "reports/onpolicy_rollout_probe_${S}_xpu_uncached_n5.txt" || true
python -u tools/collapse_metrics.py -c "$(latest $MAIN)" --device xpu:0 \
  --generate --probe-set all \
  | tee "reports/collapse_metrics_${S}_xpu_aligned.txt" || true

echo "=== [3/4] λ=0.7 CONTROL branch: seed $BRANCH from 64,000 ==="
mkdir -p "$BRANCH"
cp -n "$MAIN/step_0064000.pt" "$BRANCH/" 2>/dev/null || true
ls "$BRANCH"/step_0064000.pt >/dev/null || {
  echo "branch seed missing"; echo "no 64000 ckpt to branch $(date)" > "$FAIL"; exit 1; }

python -u -m training.distill "${COMMON[@]}" \
  --onpolicy-lambda 0.7 --total-steps 66000 --ckpt-dir "$BRANCH" || true

SB=$(step_of "$(latest $BRANCH)")
echo "=== control reached $SB ==="

echo "=== [4/4] probe the control ==="
python -u -m tools.onpolicy_rollout_probe --ckpt-dir "$BRANCH" \
  --student-device xpu:0 --teacher-device xpu:0 --teacher-id "$TEACHER" \
  --trust-remote-code --no-kv-cache --samples 5 \
  | tee "reports/onpolicy_rollout_probe_${SB}_lambda07_n5.txt" || true

echo "step=$S control=$SB OK $(date)" > "$OK"
echo "=== CHAIN DONE -> $OK ==="
echo "Tomorrow: λ=0.4 arm from the SAME 64,000 ->"
echo "  mkdir -p checkpoints_lambda04 && cp $MAIN/step_0064000.pt checkpoints_lambda04/"
