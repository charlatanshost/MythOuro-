#!/usr/bin/env bash
# TEACHER-EXHAUSTION SERIES — is the MATERIAL what went wrong?
#
#   bash run_exhaustion.sh          # ~30 min, 27 checkpoints, no training
#
# THE HYPOTHESIS (owner, 2026-08-18): the degradation is the data, not the
# recipe — "it got noisy again, this has happened before." The docs back the
# shape. On 2026-07-29 the uniform-mix change produced exactly this signature:
# soft-KL fell 46k->50k, then jumped +0.0587 at 52,000 AND STAYED ELEVATED. Our
# curve has the same form — better through 116,000, then up at 120,000 and 125,181.
#
# WHY THIS TOOL AND NOT best_exit_probe. best_exit streams its eval text from
# remote HF datasets, so its ABSOLUTE numbers are not comparable across sessions:
# on 2026-08-18 step 108,000 measured 0.348 where an archived 108,471 run had said
# 0.2036, 471 steps apart. That killed the cross-session comparison and nearly
# took a real finding with it. kd_exhaustion draws a FIXED held-out sample and
# CACHES it — reports/.heldout_s12345_b8x4_L512.pt is still on disk from
# 2026-07-29, so this series is directly comparable to the 46k->58k one
# (2.2451 -> 2.3216, mean 2.2634, step-noise rms 0.0460).
#
# ⚠ THE DOSE ARITHMETIC, which is the strongest support for the hypothesis.
# data_teacher_v2 + data_teacher_med = 17,132 rows, ~10.9M tokens. At 16 seq x
# 1024 tok x --teacher-data-ratio 0.2 = 3,277 teacher tokens/step, ONE EPOCH IS
# 3,339 STEPS. The pour has covered 72,000 -> 125,181, so the teacher corpus has
# been re-read on the order of a dozen times or more. The chat-mix leg collapsed
# at 10.3 epochs of a comparable-size corpus and recovered at 1.35. Nothing about
# the continuation corpus makes it immune to the same effect.
# (The shard files themselves are unchanged since 2026-07-24/07-29, so this is
# over-exposure to fixed data, NOT new bad data arriving.)
#
# WHAT THE SHAPE WILL SAY:
#   STEP JUMP that stays elevated  -> a distribution change, same as 52,000. Find
#       which 2,000-step window owns it and look at what the stream did there.
#   SMOOTH RISE from early on      -> genuine exhaustion/over-epoching. The fix is
#       fresh teacher material or a lower ratio, not a lower lambda.
#   FLAT across the whole pour     -> the student-teacher gap did NOT widen, and
#       the CE degradation is something else entirely (decoder, calibration, the
#       eval itself). That would be the most surprising outcome and the most
#       important one.
#
# ⚠ THE UNRESOLVED CONFOUND, from the 2026-07-29 entry: at lambda 0.7 roughly 70%
# of steps are on-policy, so soft-KL on held-out TEXT is not measuring the same
# distribution the model is actually trained on. That caveat was never cleared. It
# does not invalidate the SHAPE — a step change is a step change — but it does
# mean the absolute level should not be read as "the" student-teacher gap.
set -uo pipefail
trap 'pkill -INT -P $$ 2>/dev/null; true' INT TERM
cd "$(dirname "$0")"
source ../venv-xpu/bin/activate     # `python` does not exist outside the venv
export SYCL_CACHE_PERSISTENT=1 PYTORCH_ALLOC_CONF=expandable_segments:True TRITON_DEFAULT_BACKEND=intel

LOG="logs/exhaustion_$(date +%Y%m%d_%H%M).log"
mkdir -p logs reports

if pgrep -f "training[.](distill|sft|train_depth_policy)" >/dev/null; then
  echo "a trainer is running — it needs the card; stop it first"; exit 1
fi
if pgrep -f "tools[.](code_eval|math_eval|best_exit_probe|onpolicy_rollout_probe|kd_exhaustion)" >/dev/null; then
  echo "another probe is running; let it finish"; exit 1
fi

if [ ! -f reports/.heldout_s12345_b8x4_L512.pt ]; then
  echo "⚠ the 2026-07-29 held-out cache is MISSING — a fresh sample will be drawn."
  echo "  The series will still be internally consistent but NOT comparable to the"
  echo "  46k-58k numbers. Ctrl-C now if that matters."
fi

echo "=== EXHAUSTION SERIES: checkpoints_newmix, every 2000 (72,000 -> 124,000) ==="
echo "=== comparable baseline (2026-07-29): mean 2.2634, step-noise rms 0.0460 ==="
echo "=== log: $LOG ==="

python -u -m tools.kd_exhaustion \
  --ckpt-dir checkpoints_newmix --device xpu:0 \
  --teacher-id ByteDance/Ouro-2.6B-Thinking --trust-remote-code \
  --every 2000 --batches 8 --batch-size 4 --seq-len 512 \
  2>&1 | tee "$LOG"

echo
echo "Read the 116,000 / 118,000 / 120,000 rows first — best_exit put the break"
echo "in that window, and 118,000 is the checkpoint that splits it."
