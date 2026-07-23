#!/usr/bin/env bash
# Confirming teacher-corpus A/B: resume 36,658 with the CLEAN v2 corpus at R=0.2.
# One variable changed vs the first A/B (corpus quality). Stop whenever; ckpts
# every 15 min. See docs/training_commands.md + teacher_corpus_plan.md.
set -euo pipefail
cd "$(dirname "$0")"
source ../venv-xpu/bin/activate
export SYCL_CACHE_PERSISTENT=1 PYTORCH_ALLOC_CONF=expandable_segments:True TRITON_DEFAULT_BACKEND=intel
exec python -m training.distill \
  --student-variant mythouro_distill_tiny \
  --student-device xpu:0 --teacher-device xpu:0 \
  --teacher-id ByteDance/Ouro-2.6B-Thinking \
  --seq-len 1024 --micro-batch 8 --grad-accum 2 \
  --total-steps 46000 --warmup-steps 500 --lr 1e-4 --min-lr 3e-5 \
  --depth-reg-coeff 0.3 --divergence rev_kl \
  --use-sandwich-norm --use-depth-aware-init \
  --onpolicy-lambda 0.7 --teacher-mix-alpha 0.5 --rollout-len 64 \
  --rollout-batch 32 --rollout-reuse 2 \
  --teacher-data-ratio 0.2 --teacher-data-files 'data_teacher_v2/*.jsonl' \
  --ckpt-every-mins 15 --num-workers 0 --trust-remote-code --log-every 5 \
  --ckpt-dir checkpoints_onpolicy_fixed
