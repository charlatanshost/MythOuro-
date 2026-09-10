#!/usr/bin/env bash
# RE-HARVEST WITH THINKING ENABLED — the experiment that separates two
# explanations for why instruction data damages this model.
#
#   bash run_harvest_think.sh              # PILOT, ~35-45 min. DO THIS FIRST.
#   FULL=1 bash run_harvest_think.sh       # full run — only after reading the pilot
#
# THE QUESTION. Leg 2 (2026-09-10) showed 0.98 epochs of the clean instruction
# corpus cost −8.8pp bare code L3+, tripled L0, and made chat framing WORSE
# (degeneracy 69.1% → 95.3%, closure 8.4% → 3.1%). Two explanations survive:
#
#   (a) the EMPTY `<think></think>` SHAPE. 99.5% of that corpus demonstrates
#       "think nothing, then answer", while exit_pdf spent 4,200 steps teaching
#       the opposite. Under chat framing the two objectives collide.
#   (b) instruction data at dose simply hurts a 278M student, whatever it
#       contains — rung 8's original reading.
#
# A thinking-enabled harvest separates them. If ~1 epoch of trace-bearing
# instruction data does NOT degrade, the shape was the problem. If it degrades
# identically, the axis is closed at this scale and the answer is a bigger
# student, not a better corpus.
#
# ⚠⚠ THE 2026-08-14 DECISION PREDICTED THIS WOULD FAIL, AND IT MIGHT.
# `--no-think` was chosen for two reasons, quoted from the flag's own help:
#
#     "at --max-new 1536, 50% of responses never finished reasoning and were
#      rejected as unterminated, at ~9 tok/s. Beyond throughput, a 278M student
#      cannot execute 1500-token CoT — TRAINING ON TRACES TEACHES RAMBLING."
#
# That is a real prediction from someone who had measured it. The counter-
# argument is only this: the model was trained on ZERO traces and rambles ANYWAY
# (3.1% closure, 95.3% degenerate). So the no-think choice did not buy what it
# was meant to buy — but that does not mean traces are safe. It means untested.
#
# THE MIDDLE PATH, which 08-14 did not consider: keep traces, but only SHORT
# ones. `tools/filter_think_corpus.py` drops any row whose reasoning exceeds
# --max-think-tokens (default 200), so every retained row demonstrates a trace
# the student can FINISH inside the 512-token window it is evaluated in.
#
# ⚠ WHY THE PILOT IS NOT OPTIONAL. The feasibility of that bound is unmeasured.
# Running the filter over the EXISTING corpus (2026-09-10) found that of the 87
# rows that accidentally carry real reasoning, **70 are already longer than 200
# tokens**. If Ouro's traces are uniformly 500-1500 tokens, a 200-token bound
# yields almost nothing and the full harvest is 20-30 hours of wasted card time.
# The pilot answers that for ~40 minutes. It is the same discipline 08-14 used:
# "verified 0/18 compliance on a 2-minute probe before committing a night."
#
# THROUGHPUT, measured, for sizing the full run:
#   --no-think            24 accepted tok/s, acceptance 70.5%
#   thinking @ max_new 1536  ~9 accepted tok/s, 50% unterminated
# At ~9 tok/s a 1.0M-token corpus is ~31 hours BEFORE the length filter rejects
# anything. That is multiple nights, so read the pilot before committing.
set -uo pipefail
trap 'pkill -INT -P $$ 2>/dev/null; true' INT TERM
cd "$(dirname "$0")"
source ../venv-xpu/bin/activate
export TRITON_DEFAULT_BACKEND=intel
unset SYCL_CACHE_PERSISTENT PYTORCH_ALLOC_CONF
export PYTHONFAULTHANDLER=1 PYTHONUNBUFFERED=1
export SYCL_QUEUE_THREAD_POOL_SIZE=1
export ZE_SERIALIZE=2

OUT=data_teacher_chat_think
if [ "${FULL:-0}" = "1" ]; then
  TARGET="${TARGET:-1200000}"; TAG=full
else
  TARGET="${TARGET:-20000}";   TAG=pilot; OUT="${OUT}_pilot"
fi
LOG="logs/harvest_think_${TAG}_$(date +%Y%m%d_%H%M).log"
mkdir -p logs "$OUT"

if pgrep -f "python -u -m (training\.|tools\.)" >/dev/null; then
  echo "something is already on the card — stop it first"; exit 1; fi
FREE=$(df --output=avail -BG . | tail -1 | tr -dc '0-9')
[ "$FREE" -lt 5 ] && { echo "only ${FREE}G free"; exit 1; }

echo "=== $TAG harvest: thinking ENABLED, target ${TARGET} accepted tokens ==="
echo "=== out: $OUT   log: $LOG ==="
echo "=== NOTE: --no-think is deliberately ABSENT. That is the whole experiment. ==="

python -u -m tools.gen_teacher_corpus \
  --device xpu:0 --teacher-id ByteDance/Ouro-2.6B-Thinking --trust-remote-code \
  --out-dir "$OUT" --target-tokens "$TARGET" \
  --chat-template --prompt-len 256 \
  --max-new 1024 --min-new 32 \
  --batch 18 --prealloc-cache --telemetry \
  2>&1 | tee -a "$LOG"

echo
echo "=== WHAT THE PILOT HAS TO TELL YOU, in order ==="
echo "  1. Do the traces CLOSE? If most rows are unterminated at --max-new 1024,"
echo "     raise it or accept a low yield — the teacher needs room to finish."
echo "  2. How LONG are the think blocks? This decides whether the whole idea"
echo "     is viable. Run the filter in report mode:"
echo "       python3 tools/filter_think_corpus.py $OUT --max-think-tokens 200"
echo "     If KEPT is under ~20%, a 200-token bound is not reachable and the"
echo "     options are a larger bound, a 'think briefly' prompt change, or"
echo "     abandoning the axis. DO NOT launch FULL=1 before reading this."
echo "  3. READ THE TRACES. Counts have pointed the wrong way three times this"
echo "     month:  head -1 $OUT/shard_0000.jsonl | python3 -m json.tool"
echo
echo "  Sizing the full run from what the pilot measures:"
echo "    hours = 1.0e6 / (accepted_tok_per_s * kept_fraction) / 3600"
