#!/usr/bin/env bash
# Teacher-corpus harvest, validated fast config (b30 + prealloc + on-device
# sampling, ~100 accepted tok/s). Resumes shard numbering automatically.
set -euo pipefail
cd "$(dirname "$0")"
source ../venv-xpu/bin/activate
export SYCL_CACHE_PERSISTENT=1 PYTORCH_ALLOC_CONF=expandable_segments:True TRITON_DEFAULT_BACKEND=intel
exec python -u -m tools.gen_teacher_corpus --device xpu:0 --trust-remote-code \
  --out-dir data_teacher_v2 --target-tokens 12000000 --batch 30 --prealloc-cache
