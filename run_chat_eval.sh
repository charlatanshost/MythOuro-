#!/usr/bin/env bash
# CHAT-FRAMED eval — the ONLY framing that tests rung 8's unfixed failure.
#
#   bash run_chat_eval.sh checkpoints_instruct/step_0003000.pt instruct_3000
#
# WHY THIS EXISTS, separately from run_eval.sh. `run_eval.sh` pins
# framing=bare / chat=False, because every archived n=320 baseline used that and
# the comparisons stop working otherwise. But rung 8's failure was never a bare
# -framing failure: "`<think>` opened in 79/80 CHAT-FRAMED samples at 1.35
# epochs vs 80/80 at 10.3". Bare framing cannot see it at all.
#
# ⚠ `--extract` IS MANDATORY HERE, NOT OPTIONAL. Without it `--chat-template` is
# a 0% FLOOR for every model — the base itself scores 0.0%, as do all nine
# chat-framed evals in project history, from 2,000 to 111,471 steps. That is the
# FRAMING being measured, not the checkpoint (tools/code_eval.py:315). A 0%
# reported without --extract means nothing and has already wasted nine runs.
#
# ⚠ The L3+ number this prints is NOT comparable to the bare-framing baselines
# in run_eval.sh or to any archived report. Different framing, different
# extraction. Compare chat-to-chat only. The number that matters here is the
# THINK-TAG TABLE below it, which is the actual gate.
#
# ⚠⚠ MAX_NEW IS THE WHOLE BALLGAME HERE — read this before quoting a result.
# code_eval defaults to `--max-new 96`. A chat-framed reasoning model spends all
# 96 tokens INSIDE `<think>` and is cut off mid-sentence, so it can never emit
# `</think>`. "Opens and never closes" is then a property of the BUDGET, not of
# the model.
#
# This is not hypothetical. 2026-09-07, instruct @3,000 at max_new 96:
#   opened <think> 320/320, closed 1/320 — and completion length was
#   min 234 / median 419 / max 476 chars. NOT ONE completion reached a natural
#   stop; every single one was truncated.
# The archived rung-8 evals show the SAME signature (median 376-414, max
# 448-515), which is why "a 7.6x dose cut moved it by ONE sample": dose cannot
# fix a budget.
#
# So MAXNEW defaults to 512 here. Set MAXNEW=96 only to deliberately reproduce
# the historical artifact.
set -uo pipefail
cd "$(dirname "$0")"
source ../venv-xpu/bin/activate

CKPT="${1:?usage: bash run_chat_eval.sh <checkpoint.pt> [tag]}"
[ -f "$CKPT" ] || { echo "no such checkpoint: $CKPT"; exit 1; }
TAG="${2:-$(basename "$(dirname "$CKPT")")_$(basename "$CKPT" .pt | sed 's/step_0*//')}"
MAXNEW="${MAXNEW:-512}"
TOPP="${TOPP:-0}"      # 0 = off = historical full-vocab sampling
OUT="reports/chat_${TAG}.json"
mkdir -p reports logs

if pgrep -f "python -u -m training\.(distill|sft)" >/dev/null; then
  echo "a trainer is running — the eval will contend for the card. Stop it first."; exit 1; fi

echo "=== chat-framed eval: $CKPT  ->  $OUT  (max_new=$MAXNEW) ==="
python -u -m tools.code_eval -c "$CKPT" --device xpu:0 \
  --samples 32 --temperature 0.4 --seed 1234 --repetition-penalty 1.15 \
  --chat-template --extract --max-new "$MAXNEW" --top-p "$TOPP" \
  --json "$OUT" 2>&1 | tee -a "logs/chat_${TAG}.log"

# Single source of truth for the readout — the inline copy that used to
# live here shipped a truncation rule that stayed silent on the very run
# it was written for (2026-09-07). One implementation, tested.
python3 tools/chat_gate_readout.py "$OUT"
