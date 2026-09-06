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
set -uo pipefail
cd "$(dirname "$0")"
source ../venv-xpu/bin/activate

CKPT="${1:?usage: bash run_chat_eval.sh <checkpoint.pt> [tag]}"
[ -f "$CKPT" ] || { echo "no such checkpoint: $CKPT"; exit 1; }
TAG="${2:-$(basename "$(dirname "$CKPT")")_$(basename "$CKPT" .pt | sed 's/step_0*//')}"
OUT="reports/chat_${TAG}.json"
mkdir -p reports logs

if pgrep -f "python -u -m training\.(distill|sft)" >/dev/null; then
  echo "a trainer is running — the eval will contend for the card. Stop it first."; exit 1; fi

echo "=== chat-framed eval: $CKPT  ->  $OUT ==="
python -u -m tools.code_eval -c "$CKPT" --device xpu:0 \
  --samples 32 --temperature 0.4 --seed 1234 --repetition-penalty 1.15 \
  --chat-template --extract \
  --json "$OUT" 2>&1 | tee -a "logs/chat_${TAG}.log"

python - "$OUT" <<'PY'
import json, sys, os, re
d = json.load(open(sys.argv[1]))
S = [s for t in d["tasks"] for s in t["samples"]]
n = len(S)

opened = closed = answered = empty_think = 0
for s in S:
    c = s.get("completion", "") or ""
    o, cl = c.count("<think>"), c.count("</think>")
    if o: opened += 1
    if o and cl: closed += 1
    if o and cl:
        # is there real content after the LAST closer?
        if c.rsplit("</think>", 1)[-1].strip():
            answered += 1
        # did it think, or just emit the empty-block shape the corpus teaches?
        if not c.split("<think>", 1)[1].split("</think>", 1)[0].strip():
            empty_think += 1

print("\n" + "=" * 66)
print("  THINK-TAG BEHAVIOUR — this is the gate, not the L3+ number")
print("=" * 66)
def row(lbl, k):
    print(f"  {lbl:38s} {k:4d}/{n:<4d}  {k/n if n else 0:6.1%}")
row("opened <think>", opened)
row("...and CLOSED it", closed)
row("...and produced an answer after it", answered)
row("...with an EMPTY think block", empty_think)

print("\n  REFERENCE — rung 8's unfixed failure (data_teacher_chat, 26,130 rows):")
print("    <think> opened and NEVER closed in 79/80 chat-framed samples at 1.35")
print("    epochs, 80/80 at 10.3. A 7.6x dose cut moved it by ONE sample.")
print("\n  READ IT THIS WAY:")
if opened and closed / max(opened, 1) > 0.9 and answered / max(opened, 1) > 0.9:
    print("    * It CLOSES and ANSWERS. Rung 8's failure did not reproduce on the")
    print("      clean corpus -> that failure was the bad HARVEST, not instruction")
    print("      data. This is the result the run was built to get.")
elif opened and closed / max(opened, 1) < 0.5:
    print("    * Still opens-and-never-closes -> the failure is a property of")
    print("      instruction data itself. Roadmap rung 8 closes; chat_clean was")
    print("      the last cheap shot at it.")
else:
    print("    * Mixed. Read the completions in the json before calling it.")
if opened and empty_think / max(opened, 1) > 0.5:
    print("    * ⚠ But most blocks are EMPTY — it learned the corpus's no-think")
    print("      SHAPE (99.5% of rows), which is not the same as learning to")
    print("      answer. Check halt depth in run_prose_readout.sh before calling")
    print("      this a win: baseline 2.88, below ~2.70 means depth was spent.")
print("\n  ⚠ ONE checkpoint is ONE DRAW (L3+ sd 13.6 pp). And READ THE")
print("    COMPLETIONS in the json — the tag counts are a screen, not the call.")
PY
