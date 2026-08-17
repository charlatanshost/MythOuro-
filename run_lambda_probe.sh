#!/usr/bin/env bash
# λ PROBE — is the ON-POLICY TEACHER what stops the model from ever answering?
#
#   bash run_lambda_probe.sh          # Ctrl-C whenever (see STOPPING, below)
#
# THE QUESTION. The 2026-08-17 chat-mix retry split one failure into two:
#
#   CAPABILITY LOSS is dose-driven and now understood. Cutting 10.3 -> 1.35
#   epochs cut raw code loss from -25pp to -6.2pp, roughly proportional.
#
#   "NEVER ANSWERS" is NOT dose-driven. <think> appeared in 79/80 chat-framed
#   samples at 1.35 epochs vs 80/80 at 10.3 — a 7.6x dose cut moved it by ONE
#   sample. The model writes fluent, coherent, non-degenerate prose about the
#   problem and never emits code. adj_degenerate 0/80: it is not broken, it
#   never finishes.
#
# THE HYPOTHESIS. Every corpus row carries an EMPTY <think></think> then an
# answer, so training on it should teach "open, close immediately, answer".
# Meanwhile the same leg runs --onpolicy-lambda 0.7 against Ouro-2.6B-THINKING,
# which reasons at length by default. The corpus teaches the MARKER; the
# on-policy KL teaches what goes INSIDE it, and they pull opposite ways. The
# supporting evidence is that the BASE already opens <think> in 51/80 chat-framed
# samples with NO chat training at all — exactly what prior on-policy
# distillation against a Thinking teacher would produce.
#
# THE TEST. Identical to the chat-mix leg in every respect except
# --onpolicy-lambda 0.7 -> 0.2. Same base, same corpus, same seed, same steps.
# One variable.
#
# ⚠ WHAT THIS RISKS, and it is not small. ON-POLICY IS WHAT CURED THE REPETITION
# ATTRACTOR (tracker 2026-06-29: "on-policy converted a tokens-PROOF attractor
# into a tokens-RESPONSIVE undertrained model — the thesis flip the project was
# chasing"). Lowering λ moves back toward the regime that produced `is is is`.
# So the ladder is NOT the only readout here: adj_repeat_frac and lrs_frac decide
# whether a better chat number is real or bought with degeneracy. A leg that
# answers more and repeats more is a BAD trade and must be read as one.
#
# 1,000 STEPS, not 3,000. This is a mechanism probe, not a capability run — if
# the <think> rate is going to move, it will move early, and a short leg keeps
# the repetition risk bounded. ~80 minutes.
#
# STOPPING (see also max1100_field_notes.md):
#   * Ctrl-C ONCE, then WAIT — the handler is cooperative and writes a ~3.3GB
#     checkpoint. 30-60s of an apparently frozen terminal.
#   * Do NOT press it twice; the second signal skips the graceful save.
#   * XPU/SYCL often deadlocks in TEARDOWN after the checkpoint is written. If it
#     lingers with the GPU idle it has finished: kill -9, then confirm memory
#     returned with `xpu-smi dump -d 0 -m 18 -n 1`.
set -uo pipefail
# ── SIGNAL FORWARDING (2026-08-15) ────────────────────────────────────────────
# Without this the wrapper dies on Ctrl-C while the python child survives,
# orphaned, still holding the GPU — observed in production, see the run scripts'
# shared note. The children handle SIGINT cooperatively, so they need the signal
# delivered and then time, not a dead parent.
trap 'pkill -INT -P $$ 2>/dev/null; true' INT TERM
cd "$(dirname "$0")"
source ../venv-xpu/bin/activate
export SYCL_CACHE_PERSISTENT=1 PYTORCH_ALLOC_CONF=expandable_segments:True TRITON_DEFAULT_BACKEND=intel

SRC=checkpoints_newmix/step_0108471.pt
DIR=checkpoints_lambdaprobe
TEACHER=ByteDance/Ouro-2.6B-Thinking
FILES='data_teacher_chat/shard_*.jsonl'
LAMBDA=0.2                       # the ONE variable. chat-mix used 0.7.
STEPS=1000
LOG="logs/lambdaprobe_$(date +%Y%m%d_%H%M).log"
OK=reports/lambdaprobe_DONE; FAIL=reports/lambdaprobe_FAILED
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

rows=$(cat data_teacher_chat/shard_*.jsonl 2>/dev/null | wc -l)
if [ "${rows:-0}" -lt 20000 ]; then
  echo "data_teacher_chat has only ${rows:-0} rows — expected ~26,000; refusing"; exit 1
fi
echo "=== chat corpus: $rows rows ==="

[ -n "$(latest "$DIR")" ] || cp "$SRC" "$DIR/" || {
  echo "cannot seed $DIR from $SRC"; echo "seed failed $(date)" > "$FAIL"; exit 1; }

at=$(step_of "$(latest "$DIR")")
TARGET=$((at + STEPS))
if [ "$at" -ge "$TARGET" ]; then
  echo "already at step $at >= target $TARGET — nothing to do"; exit 1
fi
echo "=== λ PROBE: $at -> $TARGET   (--onpolicy-lambda $LAMBDA, chat-mix used 0.7) ==="
echo "=== compare against chatmix2: RAW 68.8% | CHAT 0.0% | <think> 79/80 ==="
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
  --onpolicy-lambda "$LAMBDA" \
  --ckpt-dir "$DIR" \
  --ckpt-every-mins 15 --ckpt-milestone-every 250 --keep-last 5 \
  --num-workers 0 --trust-remote-code --log-every 50 \
  --total-steps "$TARGET" 2>&1 | tee "$LOG" || true   # exit code NOT trusted (XPU teardown)

at=$(step_of "$(latest "$DIR")")
echo "lambdaprobe reached $at $(date)" > "$OK"
echo "=== stopped at step $at ==="
cat <<EOF

TWO READOUTS, AND THE SECOND IS THE VETO.

  # 1. did the <think> rate fall? (the hypothesis)
  python -u -m tools.code_eval -c $DIR/step_0*$at.pt --device xpu:0 \\
    --samples 8 --temperature 0.4 --seed 1234 --repetition-penalty 1.15 \\
    --chat-template --extract --json reports/code_lambdaprobe_${at}_CHAT.json

  # 2. did repetition come back? (the risk)
  python -u -m tools.code_eval -c $DIR/step_0*$at.pt --device xpu:0 \\
    --samples 8 --temperature 0.4 --seed 1234 --repetition-penalty 1.15 \\
    --json reports/code_lambdaprobe_${at}_RAW.json

READ IT LIKE THIS:
  <think> rate FALLS and adj_degenerate stays 0  -> hypothesis CONFIRMED, the
      on-policy teacher was fighting the corpus. Next: a full leg at low lambda.
  <think> rate FALLS but adj_degenerate/lrs_frac RISE -> the repetition attractor
      is returning. That is the 2026-06-29 regression and it is NOT a win.
  <think> rate UNCHANGED (~79/80) -> lambda is not the mechanism. The behaviour
      lives somewhere else and this whole line is wrong.

BASELINES: chatmix2 @lambda 0.7 -> RAW 68.8% | CHAT 0.0% | <think> 79/80 |
adj_degenerate 0/80 | base RAW 75.0% CHAT 6.2% <think> 51/80.
EOF
