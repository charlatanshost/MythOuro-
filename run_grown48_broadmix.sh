#!/usr/bin/env bash
# GROWN 397M (24->48 experts) — BROADENED-CORPUS LONG POUR. 6,000 steps/invocation.
#
#   bash run_grown48_broadmix.sh      # Ctrl-C ONCE and WAIT. Re-run to add 6,000 more.
#
# Same model, same checkpoint dir, same crash-avoidance config as run_grown48.sh.
# The ONLY difference is the corpus: this adds data_teacher_v2 (general/prose) and
# data_teacher_med (the mission corpus) to code+math.
#
# WHY BROADEN. run_grown48.sh pours code+math ONLY. The record says that mix is a
# TRADE, not a gain: the code corpus bought L4 (5->30/320) and SOLD L3+
# (76.6->54.1) and prose. Pouring it for the ~22 nights needed to close the token
# gap would amplify exactly that. v2+med is what run_newmix_pour.sh used for the
# pour that successfully reached 140,000.
#
# THE MIX, by TOKENS (not rows — medical rows are long, so row share understates it):
#   code+math   92.00M   88.2%
#   v2           8.65M    8.3%
#   med          3.61M    3.5%   <- 0% in run_grown48.sh; this is the mission corpus
#   TOTAL      104.26M  = 31,816 steps/epoch at 3,277 teacher tok/step (ratio 0.2)
#
# ⚠ WHY NO CHAT DATA, THOUGH data_teacher_chat/ EXISTS AND IS LARGER THAN BOTH.
# Deliberate. Chat-mix has been run TWICE and failed twice (roadmap rung 8):
#   * capability loss is DOSE-driven — 10.3 epochs cost -25pp raw code L3+,
#     1.35 epochs cost -6.2pp (a recovering transient).
#   * "never answers" is NOT dose-driven and has NO known fix — `<think>` opened
#     in 79/80 chat-framed samples at 1.35 epochs vs 80/80 at 10.3. A 7.6x dose
#     cut moved it by ONE sample.
# And the pour produced chat capability FOR FREE: chat-framed code L3+ went
# 6.2% -> 37.5% between 116,000 and 120,000 with ZERO chat data, after three SFT
# attempts and two chat-mix legs produced nothing. Adding chat here would risk a
# documented regression to buy something the pour already gives. Do not add it
# without a new result that overturns rung 8.
#
# ⚠⚠ HARD DOSE CEILING — THIS SCRIPT CANNOT CLOSE THE TOKEN GAP ALONE.
#   safe dose (from the chat-mix post-mortem)  ~1.35 epochs = ~43,000 steps
#   closing the 1.11B-token gap                 67,749 steps = 2.13 epochs
# The gap-closing pour is ABOVE the dose that measurably cost 6.2pp. So:
#   - up to ~40,000 cumulative steps: fine, 1.26 epochs, no re-read damage
#   - beyond that: EXPAND THE CORPUS FIRST (harvest), do not just keep pouring.
# Re-reading a corpus 2.4x is the "token dilution" half of the v5 post-mortem.
#
# ⚠ MEDICAL IS 3.5% OF THE TEACHER MIX, AND THE TEACHER IS 20% OF BATCHES — so
# ~0.7% of all tokens. That is a real prose/domain injection, NOT enough to build
# medical capability. Medical capability needs HARVEST, not remixing. Do not
# oversample med by repeating its glob: at the ratio needed to matter it would be
# ~8 epochs of med, which is the dose that permanently damaged the chat legs.
#
# ⚠ CONFOUND TO CARRY INTO ANY READOUT: every 278M baseline ran --rollout-batch 32.
# This runs 8 (crash avoidance, see the env block). Comparisons to 157,238/163,238
# are NOT clean size comparisons — two variables moved.
#
# Everything else — the 48-expert crash-avoidance config, the supervised restart
# loop, the growth_metadata handling — is inherited from run_grown48.sh unchanged.
# Read that script's header for why each knob is load-bearing.
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
FILES='data_teacher_code/shard_*.jsonl,data_teacher_math/shard_*.jsonl,data_teacher_v2/shard_*.jsonl,data_teacher_med/shard_*.jsonl'
STEPS=6000
LOG="logs/grown48_broadmix_$(date +%Y%m%d_%H%M).log"
OK=reports/grown48_broadmix_DONE; FAIL=reports/grown48_broadmix_FAILED
mkdir -p logs reports "$DIR"
rm -f "$OK" "$FAIL"
trap '[ -f "$OK" ] || echo "incomplete exit=$? $(date)" > "$FAIL"' EXIT

step_of() { basename "$1" | sed 's/step_0*//; s/\.pt//; s/promoted_step_//'; }
latest()  { ls -t "$1"/step_*.pt 2>/dev/null | head -1; }

if pgrep -f "python -u -m training\.(distill|sft)" >/dev/null; then
  echo "a trainer is already running"; exit 1; fi
[ -f "$SRC" ] || { echo "promoted checkpoint missing — run tools/grow_checkpoint.py"; exit 1; }

rows=$(cat data_teacher_code/shard_*.jsonl data_teacher_math/shard_*.jsonl \
            data_teacher_v2/shard_*.jsonl data_teacher_med/shard_*.jsonl 2>/dev/null | wc -l)
[ "${rows:-0}" -ge 360000 ] || { echo "corpus short: ${rows:-0} rows — expected ~367k across 4 dirs"; exit 1; }
echo "=== corpus: $rows rows across code+math+v2+med ==="
# ⚠ CONFIRM THE BLEND RESOLVED. MixedDataset logs "teacher corpus = N files — <dir>: n".
# If that line names fewer than 4 directories, the glob silently missed one and
# this leg is NOT the broadened mix. Stop and fix it.
echo "=== EXPECT a 'teacher corpus = ... 4 directories' line naming code, math, med, v2 ==="

at=$(latest "$DIR"); at=${at:+$(step_of "$at")}; at=${at:-0}
TARGET=$((at + STEPS))
echo "=== GROWN 397M broadmix leg: $at -> $TARGET  (code+math+v2+med) ==="
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
echo "grown48_broadmix reached $at $(date)" > "$OK"
cat <<EOF

READ AT --samples 32. Baselines are the 278M model at its best:
  157,238 (pre-leg-3):  L3+ 54.7  committed 42%  L4 26/320  prose 0.150
  163,238 (post):       L3+ 77.5  committed 64%  L4 19/320  prose 0.242

THE QUESTION IS WHETHER THE TRADES WEAKEN. At 278M every intervention moved one
metric up and another down. If 397M moves L3+ AND L4 AND prose together, the
capacity hypothesis is supported and growth is the lever. If it trades the same
way, the problem is the recipe, not the size — and that is worth knowing before
committing months to a 1B run.

  python -u -m tools.code_eval -c $DIR/step_0*.pt --device xpu:0 \\
    --samples 32 --temperature 0.4 --seed 1234 --repetition-penalty 1.15 \\
    --json reports/code_grown48.json
  DIR=$DIR bash run_anneal_readout.sh    # prose, 6 seeds, salad detector
EOF
