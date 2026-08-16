#!/usr/bin/env bash
# Chain: (1) validate the teacher-exhaustion gauge across the checkpoint series,
# then (2) harvest a MEDICAL teacher corpus until stopped — so the card never
# idles between the two.
#
#   bash run_gauge_then_medical.sh
#   MED_TARGET=4000000 bash run_gauge_then_medical.sh
#
# Stop the harvest whenever (Ctrl-C, or stop_gpu_jobs.sh). Stopping is EXPECTED —
# phase 1's result is already on disk by then.
#
# STOPPING (corrected 2026-08-13). "Ctrl-C is safe" is true but INCOMPLETE, and
# the gap has cost hours twice:
#   * The handler is COOPERATIVE — it sets a flag, the loop notices at the next
#     safe point, then writes a ~3.3GB checkpoint. Nothing appears for 30-60s.
#     The terminal looks frozen. It is not.
#   * Do NOT press Ctrl-C twice. A second signal forces KeyboardInterrupt and
#     SKIPS the graceful save.
#   * The XPU/SYCL runtime frequently DEADLOCKS IN TEARDOWN after the work is
#     already done and the checkpoint is written (max1100_field_notes.md). If the
#     process lingers with the GPU idle, it has finished: `kill -9 <pid>`, then
#     confirm memory returned with `xpu-smi dump -d 0 -m 18 -n 1`. A half-dead
#     process holds its allocation and the NEXT job OOMs on a "free" card.
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

CKPT_DIR=checkpoints_onpolicy_fixed
GAUGE_OUT=reports/kd_exhaustion_46k_58k.txt
MED_DIR=${MED_DIR:-data_teacher_med}
MED_TARGET=${MED_TARGET:-6000000}

echo "=== [1/2] teacher-exhaustion gauge: soft-KL across 46k->58k ==="
# Real sample size this time (the 1,024-token smoke was pure noise). Verified by
# ARTIFACT below, not exit code — torch/XPU can crash on interpreter teardown
# AFTER completing successfully (that skipped a probe on 2026-07-28).
python -u -m tools.kd_exhaustion \
  --ckpt-dir "$CKPT_DIR" --device xpu:0 --trust-remote-code \
  --every 2000 --batches 8 --batch-size 4 --seq-len 512 \
  2>&1 | tee "$GAUGE_OUT" || true

if grep -qE "^ *[0-9]{4,} +[0-9]+\.[0-9]{4}" "$GAUGE_OUT"; then
  echo "=== gauge produced measurements -> $GAUGE_OUT ==="
else
  echo "=== ⚠ gauge produced NO measurements (see $GAUGE_OUT) — continuing to harvest anyway ==="
fi

echo
echo "=== [2/2] MEDICAL teacher-corpus harvest -> $MED_DIR (stop when you like) ==="
# Separate out-dir on purpose: keeps the medical corpus independently accounted
# (its own MANIFEST) so the blend is chosen at TRAINING time via
# --teacher-data-files, rather than being baked irreversibly into data_teacher_v2.
#
# Pure medical seeds: the domain currently gets ZERO training signal, so this is
# the fastest way to add what's missing. MedRAG/pubmed = human-written PubMed
# abstracts — none of the LLM-provenance problems that disqualified the SFT
# medical sources.
python -u -m tools.gen_teacher_corpus --device xpu:0 --trust-remote-code \
  --out-dir "$MED_DIR" --target-tokens "$MED_TARGET" \
  --batch 30 --prealloc-cache --seed-mix 'medical=1.0' --telemetry
