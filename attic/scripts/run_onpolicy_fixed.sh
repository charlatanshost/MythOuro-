#!/usr/bin/env bash
# Main on-policy run, post cached-rollout fix (2026-07-16).
# Resumes from the latest ckpt in checkpoints_onpolicy_fixed (warm-started
# from clean step_0009780). See docs/training_commands.md for context.
set -euo pipefail
cd "$(dirname "$0")"

source ../venv-xpu/bin/activate
export SYCL_CACHE_PERSISTENT=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TRITON_DEFAULT_BACKEND=intel

exec python -m training.distill \
  --student-variant mythouro_distill_tiny \
  --student-device xpu:0 --teacher-device xpu:0 \
  --teacher-id ByteDance/Ouro-2.6B-Thinking \
  --seq-len 1024 --micro-batch 8 --grad-accum 2 \
  --total-steps 30000 --warmup-steps 500 --lr 1e-4 --min-lr 3e-5 \
  --depth-reg-coeff 0.3 --divergence rev_kl \
  --use-sandwich-norm --use-depth-aware-init \
  --onpolicy-lambda 0.7 --teacher-mix-alpha 0.5 --rollout-len 64 \
  --rollout-batch 32 --rollout-reuse 2 \
  --ckpt-every-mins 15 --num-workers 0 --trust-remote-code --log-every 5 \
  --ckpt-dir checkpoints_onpolicy_fixed
