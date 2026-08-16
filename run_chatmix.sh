#!/usr/bin/env bash
# CHAT-MIX LEG — does INSTRUCTION-shaped teacher data change this model at all?
#
#   bash run_chatmix.sh          # Ctrl-C whenever (see STOPPING, below)
#
# THE QUESTION. Every teacher row this model has ever trained on was a
# CONTINUATION. The only instruction-shaped data it ever met was SFT, which
# collapsed it at 3k steps, at 36.2k steps, and at 32x batch. On 2026-08-14 we
# harvested the first instruction corpus — 2,545 verified exchanges,
# `data_teacher_chat_clean/` — and routed it through the channel that actually
# works on this model: on-policy distillation, which took code L3+ 51.2% -> 75.0%.
#
# THE DESIGN, and why it is a clean A/B. All --teacher-data-files globs collapse
# into ONE stream weighted by --teacher-data-ratio. Passing chat ALONGSIDE
# v2+med would make it ~10% of that stream = ~2% of training: far too dilute to
# see anything. So this holds the ratio at the pour's 0.2 and swaps the stream's
# CONTENT — continuation teacher data out, instruction teacher data in. Same
# knob, same strength, different shape. The pour (checkpoints_newmix, 0.2 with
# v2+med) is the control, already run.
#
# ⚠ REPETITION IS REAL AND ACKNOWLEDGED. 3,000 steps x 16 seq x 1024 tok x 0.2
# draws ~9.8M tokens from a ~0.96M-token corpus — about TEN epochs. That is a lot
# for one corpus and it caps what this can prove: a positive result here means
# "instruction data moves output shape", NOT "this is the right dose". If it
# works, the answer is more harvest nights, not more epochs.
#
# WHAT TO JUDGE IT ON — and this is the part that matters. At this dose, do NOT
# expect code_eval or math_eval to move; the base is 5.0% math L3+ / 0.0% L4 and
# 3k steps will not change capability. The question is whether OUTPUT SHAPE moves
# toward instruction-following: does it emit <|im_end|> at sensible points, does
# it ANSWER a prompt rather than continue it. Read the text (docs: aggregates
# have pointed the wrong way four times).
#
# ABORT CONDITIONS:
#   * `ce` climbing steadily rather than settling => the chat format is fighting
#     the base. Kill it; checkpoints_newmix is untouched.
#   * repetition/degeneracy in the 500-step probe => the SFT attractor is back
#     under a different name.
#
# STOPPING (see also max1100_field_notes.md):
#   * Ctrl-C ONCE, then WAIT. The handler is cooperative: it flushes at the next
#     safe point and writes a ~3.3GB checkpoint. 30-60s of a terminal that looks
#     frozen. It is not.
#   * Do NOT press it twice — a second signal skips the graceful save.
#   * The XPU/SYCL runtime often deadlocks in TEARDOWN after the checkpoint is
#     written. If it lingers with the GPU idle it has finished: `kill -9 <pid>`,
#     then confirm with `xpu-smi dump -d 0 -m 18 -n 1` that memory returned. A
#     half-dead process holds its allocation and the next job OOMs on a "free"
#     card.
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

SRC=checkpoints_newmix/step_0108471.pt
DIR=checkpoints_chatmix
TEACHER=ByteDance/Ouro-2.6B-Thinking
FILES='data_teacher_chat_clean/shard_*.jsonl'
START=108471
STEPS=3000
TARGET=$((START + STEPS))
LOG="logs/chatmix_$(date +%Y%m%d_%H%M).log"
OK=reports/chatmix_DONE; FAIL=reports/chatmix_FAILED
mkdir -p logs reports "$DIR"
rm -f "$OK" "$FAIL"
trap '[ -f "$OK" ] || echo "incomplete exit=$? $(date)" > "$FAIL"' EXIT

step_of() { basename "$1" | sed 's/step_0*//; s/\.pt//'; }
latest()  { ls -t "$1"/step_*.pt 2>/dev/null | head -1; }

if pgrep -f "training[.](distill|sft|train_depth_policy)" >/dev/null; then
  echo "a trainer is already running; refusing to start a second"; exit 1
fi
if pgrep -f "tools[.]gen_teacher_corpus" >/dev/null; then
  echo "a harvest is running — it needs the card; refusing to start"; exit 1
fi

# Fail before the teacher loads if the corpus is missing or suspiciously small.
rows=$(cat data_teacher_chat_clean/shard_*.jsonl 2>/dev/null | wc -l)
if [ "${rows:-0}" -lt 1000 ]; then
  echo "data_teacher_chat_clean has only ${rows:-0} rows — expected ~2,545."
  echo "seed corpus missing or truncated; refusing to start"; exit 1
fi
echo "=== chat corpus: $rows rows ==="

[ -n "$(latest "$DIR")" ] || cp "$SRC" "$DIR/" || {
  echo "cannot seed $DIR from $SRC"; echo "seed failed $(date)" > "$FAIL"; exit 1; }

at=$(step_of "$(latest "$DIR")")
echo "=== CHAT-MIX LEG: $at -> $TARGET  (teacher stream = INSTRUCTION data @0.2) ==="
echo "=== control: checkpoints_newmix, same 0.2 ratio with CONTINUATION data ==="
echo "=== log: $LOG ==="

python -u -m training.distill \
  --student-variant mythouro_distill_tiny \
  --student-device xpu:0 --teacher-device xpu:0 --teacher-id "$TEACHER" \
  --seq-len 1024 --micro-batch 8 --grad-accum 2 \
  --warmup-steps 200 --lr 1e-4 --min-lr 3e-5 \
  --depth-reg-coeff 0.3 --divergence rev_kl \
  --use-sandwich-norm --use-depth-aware-init \
  --teacher-mix-alpha 0.5 --rollout-len 64 --rollout-batch 32 \
  --rollout-reuse 8 \
  --teacher-data-ratio 0.2 --teacher-data-files "$FILES" \
  --onpolicy-lambda 0.7 \
  --ckpt-dir "$DIR" \
  --ckpt-every-mins 15 --ckpt-milestone-every 500 --keep-last 5 \
  --num-workers 0 --trust-remote-code --log-every 50 \
  --total-steps "$TARGET" 2>&1 | tee "$LOG" || true   # exit code NOT trusted (XPU teardown)

at=$(step_of "$(latest "$DIR")")
echo "chatmix reached $at $(date)" > "$OK"
echo "=== stopped at step $at ==="
cat <<EOF

READ THE TEXT FIRST. Aggregates have pointed the wrong way four times on this
project; the question here is SHAPE, not score.

  python -u -m tools.code_eval -c $DIR/step_0*$at.pt --device xpu:0 \\
    --samples 8 --temperature 0.4 --seed 1234 --repetition-penalty 1.15 \\
    --chat-template --json reports/code_chatmix_$at.json

  python -u -m tools.math_eval -c $DIR/step_0*$at.pt --device xpu:0 \\
    --samples 8 --seed 1234 --chat-template \\
    --json reports/math_chatmix_$at.json

WHAT A POSITIVE LOOKS LIKE (any of these, at this dose):
  * emits <|im_end|> at a sensible stopping point instead of running on
  * ANSWERS the prompt rather than continuing it
  * math copied_from_prompt falls from the base's 30/80 without rel_err rising

WHAT IT WILL NOT DO: move code L3+ off 75.0% or math L4 off 0.0%. 3,000 steps on
a base at this capability does not change capability. If the shape moves, the
next step is MORE HARVEST NIGHTS — not more epochs on 0.96M tokens.

BASELINES: code L3+ 75.0% @pen1.15 | math L3+ 5.0%, copied 30/80, rel_err 0.4286.
EOF
