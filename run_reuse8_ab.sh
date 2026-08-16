#!/usr/bin/env bash
# REUSE A/B — is --rollout-reuse 8 free, or does staleness cost quality?
#
#   bash run_reuse8_ab.sh
#
# WHY: the step profiler (2026-07-30) found rollout generation was 72% of the
# step. Raising reuse cuts refill frequency and measured, pre-SDPA:
#     reuse 2  10,737.9 ms/step  (rollout 72.2%)  ~1,530 tok/s   <- validated setting
#     reuse 4   6,478.3 ms/step  (rollout 48.0%)  ~2,530 tok/s   1.66x
#     reuse 8   4,949.7 ms/step  (rollout 31.8%)  ~3,310 tok/s   2.17x
# On a dual-boot duty cycle 2.17x is the difference between 3B tokens being a
# season and being a month. But reuse buys speed with STALENESS: at reuse=8 the
# buffer serves 8*(32/8)=32 draws per fill, ~23 optimizer steps of drift at
# λ=0.7 — inside max_age_steps=50, so nothing trips, but 4x what the project has
# actually validated. On-policy dose is what broke the collapse, so this needs a
# quality gate, not a throughput argument.
#
# DESIGN: ONE new arm. The control already exists — checkpoints_lambda07 is
# 64,000 -> 66,000 at λ=0.7 with reuse=2, same data, already probed to
# reports/onpolicy_rollout_probe_66000_lambda07_n5.txt. So this arm branches
# from the SAME step-64,000 checkpoint, runs the SAME 2,000 steps, with the SAME
# λ, differing ONLY in --rollout-reuse.
#
# KNOWN CONFOUND, stated so nobody rediscovers it later: the control was trained
# before XPU SDPA was re-enabled for GQA (commit b83d1dc) and this arm runs with
# it on. Equivalence was gated on the trained 64,000 checkpoint at worst
# next-token KL 0.00200 nats vs the manual path — ~3 orders of magnitude inside
# the ~1 nat that got cached rollouts rejected — so it is treated as negligible.
# If the probe shows a difference, re-run with the SDPA path forced off before
# blaming reuse.
#
# Safe to Ctrl-C: checkpoints flush and re-running resumes. Exit codes are NOT
# trusted (XPU teardown crashes after success) — every stage verifies by ARTIFACT.
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

MAIN=checkpoints_onpolicy_fixed
SEED_CKPT="$MAIN/step_0064000.pt"
TEACHER=ByteDance/Ouro-2.6B-Thinking
FILES='data_teacher_v2/shard_*.jsonl,data_teacher_med/shard_*.jsonl'
DIR=checkpoints_reuse8
TARGET=66000
OK=reports/reuse8_ab_DONE; FAIL=reports/reuse8_ab_FAILED
rm -f "$OK" "$FAIL"
trap '[ -f "$OK" ] || echo "incomplete exit=$? $(date)" > "$FAIL"' EXIT

step_of() { basename "$1" | sed 's/step_0*//; s/\.pt//'; }
latest()  { ls -t "$1"/step_*.pt 2>/dev/null | head -1; }

mkdir -p "$DIR"
[ -n "$(latest "$DIR")" ] || cp "$SEED_CKPT" "$DIR/" || {
  echo "cannot seed $DIR from $SEED_CKPT"; echo "seed failed $(date)" > "$FAIL"; exit 1; }

at=$(step_of "$(latest "$DIR")")
if [ "$at" -ge "$TARGET" ]; then
  echo "=== already at $at — skipping training ==="
else
  echo "=== REUSE=8 arm: $at -> $TARGET (control = checkpoints_lambda07, reuse=2) ==="
  python -u -m training.distill \
    --student-variant mythouro_distill_tiny \
    --student-device xpu:0 --teacher-device xpu:0 --teacher-id "$TEACHER" \
    --seq-len 1024 --micro-batch 8 --grad-accum 2 \
    --warmup-steps 500 --lr 1e-4 --min-lr 3e-5 \
    --depth-reg-coeff 0.3 --divergence rev_kl \
    --use-sandwich-norm --use-depth-aware-init \
    --teacher-mix-alpha 0.5 --rollout-len 64 --rollout-batch 32 \
    --rollout-reuse 8 \
    --teacher-data-ratio 0.2 --teacher-data-files "$FILES" \
    --onpolicy-lambda 0.7 --ckpt-dir "$DIR" \
    --ckpt-every-mins 15 --ckpt-milestone-every 2000 --keep-last 5 \
    --num-workers 0 --trust-remote-code --log-every 20 \
    --total-steps "$TARGET" || true          # exit code NOT trusted
  at=$(step_of "$(latest "$DIR")")
fi

if [ "$at" -lt "$TARGET" ]; then
  echo "=== stopped early at $at (< $TARGET) — not probing ==="
  echo "early stop at $at $(date)" > "$FAIL"; exit 1
fi

# ── Generation probe: the SAME instrument the control was measured with ──
OUT="reports/onpolicy_rollout_probe_${at}_reuse8_n5.txt"
echo "=== probing $at -> $OUT ==="
python -u -m tools.onpolicy_rollout_probe --ckpt-dir "$DIR" \
  --student-device xpu:0 --teacher-device xpu:0 --teacher-id "$TEACHER" \
  --trust-remote-code --no-kv-cache --samples 5 | tee "$OUT" || true
[ -s "$OUT" ] && [ "$(wc -l < "$OUT")" -ge 20 ] || {
  echo "probe produced nothing"; echo "probe failed $(date)" > "$FAIL"; exit 1; }

# ── Code eval: n=8 both penalty settings (capability + degeneracy) ──
for pen in 1.0 1.15; do
  J="reports/code_eval_reuse8_${at}_pen${pen}.json"
  python -u -m tools.code_eval -c "$(latest "$DIR")" --device xpu:0 \
    --samples 8 --temperature 0.4 --repetition-penalty "$pen" --json "$J" \
    2>&1 | grep -E "reached L2|repetition|SALVAGED" || true
  [ -s "$J" ] || { echo "code_eval pen=$pen produced nothing"; \
                   echo "code_eval failed $(date)" > "$FAIL"; exit 1; }
done

echo "reuse8 A/B OK $(date)" > "$OK"
echo "=== DONE -> $OK ==="
echo "Compare: $OUT"
echo "     vs: reports/onpolicy_rollout_probe_66000_lambda07_n5.txt   (reuse=2 control)"
echo "     and reports/code_eval_reuse8_${at}_pen*.json"
echo "     vs: reports/code_eval_pen1.0_64000.json / code_eval_pen1.15_64000.json"
