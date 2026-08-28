#!/usr/bin/env bash
# CONTROL — does the 278M model segfault too? ~10 min, saves nothing.
#
#   bash run_control_278m.sh
#
# WHY. Every 48-expert crash has this shape:
#     429 Too Many Requests -> "Retrying in 60sec [1/20]" -> ~9 min -> first
#     forward -> Segmentation fault (core dumped)
# and every passing test used RANDOM tensors with no HF stream. The 429s started
# TODAY; last night's 278M mathcode legs ran clean on the same code.
#
# So two things changed at once and I have been attributing the crash to ours:
#   (a) the model went 24 -> 48 experts
#   (b) HuggingFace began rate-limiting the base stream
#
# This runs the OLD 278M checkpoint through the SAME profiling path. If it also
# segfaults, the growth is exonerated and the fault is in the data stream under
# rate-limiting — which would mean tonight's five hours of elimination were
# chasing the wrong variable.
#
# ALREADY RULED OUT for the 48-expert model (all passed):
#   bench_step fwd+bwd, batch 8 seq 1024        727 ms/step
#   forward with router_bias[24:] = -100        OK
#   generate_rollout, uncached, ACT live        OK
#   teacher resident + rollout at alpha=0.45    OK, peak 21.4 GB of 48
set -uo pipefail
trap 'pkill -INT -P $$ 2>/dev/null; true' INT TERM
cd "$(dirname "$0")"
source ../venv-xpu/bin/activate
export SYCL_CACHE_PERSISTENT=1 PYTORCH_ALLOC_CONF=expandable_segments:True TRITON_DEFAULT_BACKEND=intel
mkdir -p logs
LOG="logs/control278_$(date +%Y%m%d_%H%M).log"

python -u -m training.distill \
  --student-variant mythouro_distill_tiny \
  --student-device xpu:0 --teacher-device xpu:0 \
  --teacher-id ByteDance/Ouro-2.6B-Thinking \
  --seq-len 1024 --micro-batch 8 --grad-accum 2 \
  --warmup-steps 500 --lr 1e-4 --min-lr 3e-5 \
  --depth-reg-coeff 0.3 --divergence rev_kl \
  --use-sandwich-norm --use-depth-aware-init \
  --teacher-mix-alpha 0.45 --rollout-len 64 --rollout-batch 32 --rollout-reuse 8 \
  --teacher-data-ratio 0.2 \
  --teacher-data-files 'data_teacher_code/shard_*.jsonl,data_teacher_math/shard_*.jsonl' \
  --onpolicy-lambda 0.7 \
  --ckpt-dir checkpoints_mathcode \
  --num-workers 0 --trust-remote-code \
  --profile-warmup 2 --profile-steps 3 \
  --total-steps 170000 \
  > >(tee -a "$LOG") 2> >(tee -a "$LOG.err" >&2) || true

echo
echo "  SEGFAULTS TOO  -> not the growth. The fault is the data stream under"
echo "                    rate-limiting, and the 48-expert path is exonerated."
echo "  RUNS CLEAN     -> the crash really is specific to 48 experts, and the"
echo "                    429 is coincidental."
