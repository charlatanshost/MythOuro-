#!/usr/bin/env bash
# INSTRUCTION HARVEST — teacher answers questions about real passages.
#
#   bash run_harvest_chat.sh          # Ctrl-C whenever (see STOPPING, below)
#
# WHY THIS CORPUS EXISTS. Every teacher row before 2026-08-11 was a CONTINUATION:
# 48 tokens of real text, teacher writes the next 768. So the student had NEVER
# seen an instruction -> response pair from its teacher — its whole training
# history is raw web text plus teacher continuations of raw web text. The ONLY
# instruction-shaped data it ever met was SFT, applied all at once in a ChatML
# format it had never seen, and SFT collapsed it at 3k steps, at 36.2k steps, and
# at 32x batch. This routes instruction-following through on-policy distillation
# instead — the one channel proven on this model (code L3+ 51.2% -> 75.0%).
#
# The instruction wraps a REAL corpus snippet: the corpus supplies the CONTENT,
# the teacher supplies the FORM. Grounding also cuts fabrication — free
# continuation is what produced the medical harvest's "PAWL study" and 30-mg
# ibuprofen (real dosing 200-400mg).
#
# ⚠ THE FIRST ATTEMPT (2026-08-13) PRODUCED 1.22M UNUSABLE TOKENS. Structure was
# perfect — 0 malformed rows, 0 hallucinated turns, 0 missing terminators — and
# the summaries were genuinely grounded. It was still garbage, because EVERY
# response was pinned at the 512-token cap (median 512, max 513: nothing finished
# naturally), 67% had an unclosed <think>, and the record appended <|im_end|>
# whether the model emitted it or not. That last one is the killer: it stamps a
# completion marker on a sentence cut in half, teaching "stop mid-thought after N
# tokens" as a valid ending. All three are fixed (d17242e) — rows that hit
# --max-new without terminating are now REJECTED, not relabelled.
#
# SETTINGS, and why they are not the defaults:
#   --no-think      Prefills a CLOSED <think></think> so the teacher answers
#                   DIRECTLY. At --max-new 1536 WITHOUT it, 50% of responses never
#                   finished reasoning (rejected as unterminated) at ~9 tok/s. It
#                   is also the better corpus: a 278M student cannot execute
#                   1500-token CoT, and training on traces teaches rambling.
#                   ⚠ VERIFY THE TEACHER COMPLIES — if it opens a second <think>,
#                   the flag does nothing. That check is a 2-minute run.
#   --max-new 1024  MEASURED, not guessed: with --no-think the teacher's answers
#                   run median ~1400 chars (~470 tok), max ~2455 (~820 tok).
#                   512 truncated a third of them. 1024 covers the median
#                   comfortably and most of the tail.
#   --min-new 32    Direct answers are legitimately short.
#   --batch 18      lane = 256+1024+8 = 1288 tok; 18 x 1288 = 23,184 lane-tokens
#                   ~42.5 GB against the measured 24,720 = 45.3 GB envelope.
#
# EXPECT ~40-60 accepted tok/s, NOT the 118 of the continuation harvest, and
# that gap is structural rather than a bug. Continuations all run to the full
# --max-new and every token is kept. Instruction answers vary in length, and
# batched decode cannot stop until the LONGEST row in the batch finishes — so
# short answers idle in their lanes. Continuous batching (lane eviction and
# refill) is the real fix and is unbuilt; harvest_speedup_plan.md demoted it
# after measuring EOS waste at 7.8%, which was correct for continuations and
# is wrong here. Plan 2-3 nights for 5M tokens, or lower --target-tokens.
#   --prompt-len 256  ~30 tokens of ChatML framing, ~226 of passage. NOT a
#                   throughput lever — measured 118 tok/s at 256, which beat the
#                   best continuation harvest (116.7).
#
# STOPPING — read this once, it has cost hours twice:
#   * Ctrl-C ONCE, then WAIT. The handler is COOPERATIVE: it flushes buffered
#     rows and exits at the next safe point. Nothing appears immediately.
#   * Do NOT press it twice. A second signal forces KeyboardInterrupt and skips
#     the flush — up to ROWS_PER_SHARD-1 accepted rows (~700k tokens, hours of
#     GPU) are discarded. That has happened before.
#   * The XPU/SYCL runtime often DEADLOCKS IN TEARDOWN after the work is done
#     (max1100_field_notes.md). If the process lingers with the GPU idle, the
#     harvest already finished: `kill -9`, then confirm memory actually returned
#     with `xpu-smi dump -d 0 -m 18 -n 1`. A half-dead process holds its full
#     allocation and the NEXT job OOMs on a card that looks free.
set -uo pipefail
cd "$(dirname "$0")"
source ../venv-xpu/bin/activate
export SYCL_CACHE_PERSISTENT=1 PYTORCH_ALLOC_CONF=expandable_segments:True TRITON_DEFAULT_BACKEND=intel

TEACHER=ByteDance/Ouro-2.6B-Thinking
DIR=data_teacher_chat
TARGET=5000000
LOG="logs/harvest_chat_$(date +%Y%m%d_%H%M).log"
OK=reports/harvest_chat_DONE; FAIL=reports/harvest_chat_FAILED
mkdir -p logs reports
rm -f "$OK" "$FAIL"
trap '[ -f "$OK" ] || echo "incomplete exit=$? $(date)" > "$FAIL"' EXIT

# Two harvests on one out-dir interleave shard indices and corrupt the manifest.
if pgrep -f "tools[.]gen_teacher_corpus" >/dev/null; then
  echo "a harvest is already running; refusing to start a second"; exit 1
fi
# The teacher needs the whole card; a trainer would OOM it (or be OOM'd).
if pgrep -f "training[.](distill|sft)" >/dev/null; then
  echo "a trainer is running — the teacher needs the card to itself"; exit 1
fi

echo "=== INSTRUCTION HARVEST -> $DIR (target $TARGET tok) ==="
echo "=== log: $LOG ==="
echo "=== expect within ~2s: 'chat-template preflight OK: 256-token prompts' ==="
echo "=== if that line does NOT appear, stop: config is wrong, no GPU time spent ==="

python -u -m tools.gen_teacher_corpus \
  --device xpu:0 --teacher-id "$TEACHER" --trust-remote-code \
  --chat-template --no-think --prompt-len 256 --max-new 1024 --min-new 32 \
  --batch 18 --prealloc-cache \
  --target-tokens "$TARGET" --out-dir "$DIR" --stream-seed 1 \
  2>&1 | tee "$LOG" || true          # exit code NOT trusted (XPU teardown)

rows=$(cat "$DIR"/shard_*.jsonl 2>/dev/null | wc -l)
echo "harvest reached $rows rows $(date)" > "$OK"
echo "=== stopped: $rows rows in $DIR ==="
cat <<EOF

WATCH THE REJECT COUNTERS — they are the instrument that says whether --max-new
is big enough, and they were invisible before 2026-08-13:

    unterminated=N    hit --max-new without emitting <|im_end|>  -> RAISE --max-new
    unclosed_think=N  opened <think>, never closed               -> RAISE --max-new
    too_short=N       shorter than --min-new                     -> fine in moderation

SPOT-READ BEFORE TRAINING ON THIS. Structural checks are NOT sufficient — the
first harvest passed every one of them and was still unusable:

  python3 - <<'PY'
  import json, statistics as st
  rows=[json.loads(l) for l in open("$DIR/shard_0000.jsonl")]
  L=[len(r["text"].split("<|im_start|>assistant\\n",1)[1]) for r in rows]
  print(f"{len(rows)} rows | response chars: median {st.median(L):.0f} max {max(L)}")
  print("  ^ if the median sits AT the cap, nothing finished naturally — stop")
  for r in rows[:2]:
      print("="*70); print(r["text"][:900])
  PY

Then: does the assistant ANSWER the passage, or continue it? Any second
<|im_start|>user turn? Does <think> close before the answer?
EOF
