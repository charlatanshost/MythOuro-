#!/usr/bin/env bash
# GROWN 24 -> 48 EXPERTS, first pour on the promoted model.
# ~19h TOTAL at the MEASURED 11.5 s/step (steps 2550->2600, 2026-08-29). This is
# a MULTI-NIGHT leg. Two earlier estimates were wrong in both directions -- 6.5
# s/step (reduced-config profile) and 28.1 s/step (first 14 steps, polluted by
# warmup and HF 503 retries). Quote a rate only from a >=50-step steady window.
#
#   bash run_grown48.sh          # Ctrl-C ONCE and WAIT
#
# Runs in the ONE configuration that has ever survived at 48 experts (see the
# env block below) and SUPERVISES itself: a crash resumes from the last
# checkpoint rather than losing the night. Three no-progress attempts = real
# failure, and it stops.
#
# WHY GROWTH, AND WHY NOW. Every intervention this month has been a TRADE, not a
# gain: +code corpus bought L4 (5->30/320) and sold L3+ (76.6->54.1); more steps
# sold it back (L4 26->19, L3+ 54.7->77.5); the pour bought chat and sold raw;
# SFT bought teacher-forced fit and sold generation. At 278M with 2.58B tokens
# seen the model looks capacity-saturated — it reallocates rather than retains.
# Growth is the only lever that changes that constraint.
#
# ⚠ AND THE OBVIOUS ALTERNATIVE WAS RULED OUT BY ARITHMETIC. "Leg 3 hurt because
# the corpus was re-read" is FALSE: 92M tokens = 28,076 steps/epoch, and the
# damage appeared at 12,000 cumulative steps = 0.43 EPOCHS. The corpus was never
# exhausted, so more corpus does not obviously fix this.
#
# WHY THIS IS NOT v5. The 2026-06-06 "growth is tapped out" verdict was RETRACTED
# on 2026-06-29: v5's ceiling was token dilution + exposure-bias collapse, not a
# parameter ceiling. The guardrail is "expand only on an un-collapsed base, with
# tokens scaling in step". v5 violated both — it grew onto a collapsed model with
# 20-40M tokens and followed with ~2.4K SFT steps. Here the base is 0/80
# char-degenerate with the acronym salad gone, and 2.58B tokens are in.
#
# ⚠ THE GROWTH PATH ONLY WORKED THROUGH SFT UNTIL TODAY. tools/grow_checkpoint.py
# gates the 24 new experts behind router_bias -100.0, decaying over 500 steps.
# sft.py has applied that schedule since growth was built; distill.py had ZERO
# references to it (fixed 2026-08-27, commit 123e460). Without that patch this
# run would produce a 397M model whose new experts NEVER enter top-k — bigger,
# slower, identical to the source. Watch for the "GROWN checkpoint" log line on
# startup; if it is absent, STOP: the metadata is not being read.
#
# SOURCE: checkpoints_base/step_0157000.pt — the preserved stand-in for 157,238,
# which was the best checkpoint on L4 (26/320) and prose (0.150, zero salad)
# before leg 3 undid both. 157,238 itself was rotated away by --keep-last 5.
# ⚠ 157,000 is UNEVALUATED; it is presumed equivalent at 238 steps' distance.
#
# ⚠ NO --resume FLAG EXISTS. distill.py resumes by globbing --ckpt-dir via
# list_ckpts(), which requires the zero-padded `step_{0000000}.pt` convention —
# tools/grow_checkpoint.py's suggested `promoted_step_0.pt` is NOT matched and
# would be silently ignored, starting a 397M model from RANDOM INIT. The promoted
# file is therefore renamed to step_0000000.pt inside the ckpt dir.
#
# --start-loops 4, NOT the default 2. The curriculum ramps against total_steps
# from the run's own step counter, and a promoted checkpoint restarts at step 0 —
# so the default would run the first ~1,000 steps at HALF the trained depth on a
# model that is function-preserving at depth 4. (This exact mistake made the
# 2026-08-26 profile measure the wrong model.)
#
# ⚠ PARAM COUNT CORRECTED 2026-08-29: this is a 397M model, NOT 460M. The old
# figure came from summing the state dict, which stores the TIED embedding twice
# (`embed.weight` and `head.weight` are the same 49152x1280 = 62.9M tensor).
# Trainable params are 396,879,731 — the number distill.py logs on startup.
#
# TOKEN GUARDRAIL, stated honestly: 397M at the current 9.3 tok/param wants 3.69B
# tokens; 2.58B are in. This leg does not close that gap — it tests whether the
# promotion HOLDS and whether the trades weaken. The gap is ~8 days on the 1100.
#
# STOPPING: Ctrl-C ONCE then WAIT (~3.3GB+ checkpoint). Do not press twice. XPU
# often deadlocks in teardown after the save — kill -9 then confirm with
# `xpu-smi dump -d 0 -m 18 -n 1`.
set -uo pipefail
STOP=0
trap 'STOP=1; pkill -INT -P $$ 2>/dev/null; true' INT TERM
cd "$(dirname "$0")"
source ../venv-xpu/bin/activate
# ── ENV = THE ONE CONFIGURATION THAT HAS EVER RUN (2026-08-28 16:40) ─────────
# The 48-expert trainer crashes with a GPU page fault (NotPresent/PDE/Write)
# under every config tried EXCEPT one; the 2x2 over {cache} x {config} has one
# green cell and this leg replicates it VERBATIM, including the diagnostic env
# vars present in the passing run. Do not "clean up" without re-testing: the
# fault class is nondeterministic and config-sensitive.
export TRITON_DEFAULT_BACKEND=intel
unset SYCL_CACHE_PERSISTENT PYTORCH_ALLOC_CONF
export PYTHONFAULTHANDLER=1 PYTHONUNBUFFERED=1
export SYCL_QUEUE_THREAD_POOL_SIZE=1
export ZE_SERIALIZE=2

SRC=checkpoints_grown48/step_0000000.pt
DIR=checkpoints_grown48
TEACHER=ByteDance/Ouro-2.6B-Thinking
FILES='data_teacher_code/shard_*.jsonl,data_teacher_math/shard_*.jsonl'
STEPS=6000
LOG="logs/grown48_$(date +%Y%m%d_%H%M).log"
OK=reports/grown48_DONE; FAIL=reports/grown48_FAILED
mkdir -p logs reports "$DIR"
rm -f "$OK" "$FAIL"
trap '[ -f "$OK" ] || echo "incomplete exit=$? $(date)" > "$FAIL"' EXIT

step_of() { basename "$1" | sed 's/step_0*//; s/\.pt//; s/promoted_step_//'; }
latest()  { ls -t "$1"/step_*.pt 2>/dev/null | head -1; }

if pgrep -f "python -u -m training\.(distill|sft)" >/dev/null; then
  echo "a trainer is already running"; exit 1; fi
[ -f "$SRC" ] || { echo "promoted checkpoint missing — run tools/grow_checkpoint.py"; exit 1; }

rows=$(cat data_teacher_code/shard_*.jsonl data_teacher_math/shard_*.jsonl 2>/dev/null | wc -l)
[ "${rows:-0}" -ge 250000 ] || { echo "corpus short: ${rows:-0} rows"; exit 1; }
echo "=== corpus: $rows rows ==="

at=$(latest "$DIR"); at=${at:+$(step_of "$at")}; at=${at:-0}
TARGET=$((at + STEPS))
echo "=== GROWN 48-expert leg: $at -> $TARGET  (397M params) ==="
echo "=== EXPECT a 'GROWN checkpoint — 24 -> 48 experts' line below. If absent, STOP. ==="
echo "=== log: $LOG ==="

# ── SUPERVISED RESTART LOOP ─────────────────────────────────────────────────
# The 48-expert fault is NONDETERMINISTIC and load-dependent: the same command
# crashed all day 2026-08-28 and passed once. A single-shot launch therefore
# gambles the whole night on one draw. This loop re-launches from the last
# checkpoint instead (distill.py resumes by globbing step_*.pt), so a crash
# costs at most --ckpt-every-mins of work rather than the night.
#
# It gives up only when THREE consecutive attempts make ZERO forward progress —
# that distinguishes "unlucky draw" from "cannot start at all", so a hard block
# still surfaces as a real failure instead of spinning until morning.
#
# Ctrl-C sets STOP=1 and is NOT treated as a crash: distill.py exits 0 after a
# clean shutdown save, which would otherwise look like a restartable stall.
MAX_TRIES=10
tries=0; stall=0
while :; do
  before=$(latest "$DIR"); before=${before:+$(step_of "$before")}; before=${before:-0}

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
    --ckpt-every-mins 15 --ckpt-milestone-every 500 --keep-last 5 \
    --num-workers 0 --trust-remote-code --log-every 50 \
    --total-steps "$TARGET" \
    > >(tee -a "$LOG") 2> >(tee -a "$LOG.err" >&2)
  rc=$?

  after=$(latest "$DIR"); after=${after:+$(step_of "$after")}; after=${after:-0}
  [ "$STOP" = "1" ] && { echo "=== interrupted by user at step $after ==="; break; }
  [ "${after:-0}" -ge "$TARGET" ] && { echo "=== reached $after ==="; break; }

  tries=$((tries+1))
  if [ "${after:-0}" -le "${before:-0}" ]; then stall=$((stall+1)); else stall=0; fi
  echo "=== CRASH rc=$rc  step $before -> $after  (try $tries/$MAX_TRIES, stall $stall/3) $(date) ==="
  tail -20 "$LOG.err" 2>/dev/null | sed 's/^/    | /'

  if [ "$stall" -ge 3 ]; then
    echo "=== 3 consecutive attempts made no progress — the leg cannot start. Stopping. ==="
    break
  fi
  if [ "$tries" -ge "$MAX_TRIES" ]; then
    echo "=== $MAX_TRIES attempts exhausted at step $after. Stopping. ==="
    break
  fi
  echo "=== restarting from step $after in 60s ==="
  sleep 60
done
# ⚠ stderr goes to its OWN file, unbuffered by tee's stdout path. The first
# attempt (2026-08-28 00:33) died after the first forward pass with NOTHING in
# the log — a hard abort can kill the process before a shared `2>&1 | tee`
# flushes. GPU-level faults on this card print to stderr and have been lost this
# way before (bench_rollout, 2026-08-24).

at=$(step_of "$(latest "$DIR")")
echo "grown48 reached $at $(date)" > "$OK"
cat <<EOF

READ AT --samples 32. Baselines are the 278M model at its best:
  157,238 (pre-leg-3):  L3+ 54.7  committed 42%  L4 26/320  prose 0.150
  163,238 (post):       L3+ 77.5  committed 64%  L4 19/320  prose 0.242

THE QUESTION IS WHETHER THE TRADES WEAKEN. At 278M every intervention moved one
metric up and another down. If 397M moves L3+ AND L4 AND prose together, the
capacity hypothesis is supported and growth is the lever. If it trades the same
way, the problem is the recipe, not the size — and that is worth knowing before
committing months to a 1B run.

  bash run_g2_readout.sh    # control vs final at n=320, prose, and the G2 verdict

  # ^ replaces the old two-liner. tools.code_eval takes ONE checkpoint (-c is
  #   required=True, no nargs), so \`-c \$DIR/step_0*.pt\` expanded to 22 paths
  #   and died on unrecognised arguments. run_g2_readout.sh also evaluates
  #   step_0000000 as a within-session control, since promotion is bit-exact.
EOF
