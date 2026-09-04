#!/usr/bin/env bash
# INSTRUCTION DATA — the model has never seen an instruction→response pair in a pour.
#
#   bash run_instruct.sh          # ~8.3h, 3,000 steps
#
# WHY NOW. Reading all six probe seeds at α=0.0 AND α=0.25 across the lineage
# (2026-09-04) showed a consistent pattern across THREE independent interventions
# — growth, exit_pdf, expert dropout: **fluency up, factual grounding down.**
# The 157,000 base is still the only checkpoint with correct fibonacci base cases
# and the only one naming the quadratic formula; 163,238 is the only one naming
# the discriminant. It matches the code ladder exactly — L0 collapsed 12.6% → 2.5%
# while L4 never moved. More math tokens are not an obvious fix for grounding.
#
# ⚠ CHAT DATA FAILED TWICE. Roadmap rung 8, and the failures were different:
#   * capability loss IS dose-driven — 10.3 epochs cost −25pp raw code L3+,
#     1.35 epochs cost −6.2pp (a recovering transient).
#   * "never answers" is NOT dose-driven and had NO known fix — `<think>` opened
#     in 79/80 chat-framed samples at 1.35 epochs vs 80/80 at 10.3. A 7.6x dose
#     cut moved it by ONE sample.
#
# ⚠⚠ BUT BOTH LEGS USED `data_teacher_chat/` — THE 26,130-ROW HARVEST. This uses
# `data_teacher_chat_clean/`, which is a different corpus and has NEVER been
# poured. From the 2026-08-14 entry: 2,545 rows, ~1.0M tokens, and verified —
# 0/2,545 malformed, 0 unclosed `<think>`, 0 reopened `<think>` (168 dropped from
# the raw 2,713), content checked BY READING. The first harvest, by contrast,
# produced "1.22M UNUSABLE tokens that passed every structural check".
# So "never answers" may be a property of that bad harvest rather than of
# instruction data. That is the hypothesis this run tests.
#
# DOSE: 4x oversample = 2.70% of the mix = ~0.27 epochs over 3,000 steps.
# Deliberately 5x below the 1.35-epoch dose that cost 6.2pp, and ~40x below the
# 10.3 that did not recover. Oversampling works because the loader keeps
# duplicate globs: `sorted(f for p in pattern.split(",") for f in glob(p))`.
#
# ⚠ chat_clean carried `chat` and `seed_len` columns the other corpora lack.
# preflight REFUSED the mix until it was normalised — without that assertion the
# entire instruction corpus would have been silently dropped and the run would
# have looked clean. (Same failure that halved the corpus on 2026-08-31.)
#
# THE GATE, pre-registered:
#   `<think>` still opens-and-never-closes  -> rung 8's unfixed failure is a
#       property of instruction data itself, not of the bad harvest. Stop; the
#       axis is closed and chat_clean was the last cheap shot at it.
#   answers appear AND code/prose hold      -> instruction data works at this
#       dose on a verified corpus, and grounding has a lever that is not more math.
#   code/prose regress                      -> dose is still too high even at
#       0.27 epochs; drop to natural 0.69% before abandoning.
set -uo pipefail
STOP=0
trap 'STOP=1; pkill -INT -P $$ 2>/dev/null; true' INT TERM
cd "$(dirname "$0")"
source ../venv-xpu/bin/activate
export TRITON_DEFAULT_BACKEND=intel
unset SYCL_CACHE_PERSISTENT PYTORCH_ALLOC_CONF
export PYTHONFAULTHANDLER=1 PYTHONUNBUFFERED=1
export SYCL_QUEUE_THREAD_POOL_SIZE=1
export ZE_SERIALIZE=2

DIR=checkpoints_instruct
SRC=checkpoints_exitpdf/step_0007200.pt      # best current model; ONE variable changes: the corpus
TEACHER=ByteDance/Ouro-2.6B-Thinking
CC='data_teacher_chat_clean/shard_*.jsonl'
FILES="data_teacher_code/shard_*.jsonl,data_teacher_math/shard_*.jsonl,data_teacher_v2/shard_*.jsonl,data_teacher_med/shard_*.jsonl,$CC,$CC,$CC,$CC"
STEPS="${STEPS:-3000}"
LOG="logs/instruct_$(date +%Y%m%d_%H%M).log"
mkdir -p logs reports "$DIR"

if pgrep -f "python -u -m training\.(distill|sft)" >/dev/null; then
  echo "a trainer is already running"; exit 1; fi
[ -f "$SRC" ] || { echo "missing $SRC"; exit 1; }
if [ ! -f "$DIR/step_0000000.pt" ]; then
  cp "$SRC" "$DIR/step_0000000.pt"
  python - <<'PY'
import torch, os
p="checkpoints_instruct/step_0000000.pt"
ck=torch.load(p,map_location="cpu",weights_only=False)
ck["step"]=0                      # fresh lineage — see the 09-01 zero-step failure
torch.save(ck,p+".tmp"); os.replace(p+".tmp",p); print("  seeded at step 0")
PY
fi

at=$(basename "$(ls -t $DIR/step_*.pt | head -1)" | sed 's/step_0*//; s/\.pt//'); at=${at:-0}
echo "=== instruction leg: $at -> $((at+STEPS))  (chat_clean 4x = 2.70%, ~0.27 epochs) ==="
echo "=== EXPECT 5 corpus directories below. Fewer means a glob missed. ==="
echo "=== log: $LOG ==="

python -u -m training.distill \
  --student-variant mythouro_distill_tiny \
  --student-device xpu:0 --teacher-device xpu:0 --teacher-id "$TEACHER" \
  --seq-len 1024 --micro-batch 2 --grad-accum 8 \
  --warmup-steps 500 --lr 1e-4 --min-lr 3e-5 --start-loops 4 \
  --loop-loss-weighting exit_pdf --depth-reg-coeff 0.1 \
  --divergence rev_kl \
  --use-sandwich-norm --use-depth-aware-init \
  --teacher-mix-alpha 0.45 --rollout-len 64 --rollout-batch 8 --rollout-reuse 8 \
  --teacher-data-ratio 0.2 --teacher-data-files "$FILES" \
  --onpolicy-lambda 0.7 \
  --ckpt-dir "$DIR" \
  --ckpt-every-mins 15 --ckpt-milestone-every 500 --keep-last 8 \
  --num-workers 0 --trust-remote-code --log-every 50 \
  --total-steps $((at+STEPS)) \
  > >(tee -a "$LOG") 2> >(tee -a "$LOG.err" >&2)

echo
echo "=== READ IN THIS ORDER ==="
echo "  1. Does it ANSWER?  (rung 8's unfixed failure — <think> opens, never closes)"
echo "     bash run_eval.sh $DIR/step_0003000.pt instruct_3000"
echo "  2. Prose, >=3 checkpoints:"
echo "     bash run_prose_readout.sh $(ls -t $DIR/step_0*.pt 2>/dev/null | head -3 | tr '\n' ' ')"
echo "  3. Read ALL SIX seeds at α=0.0 AND α=0.25, and check specifically:"
echo "       fibonacci base cases  /  the quadratic formula  /  diabetes symptoms"
echo "     Those are where the new lineage has been losing to the 157,000 base."
echo "  bars: exit_pdf 0.106/0.527  base 0.137/0.520  grown48 0.104/0.571  mathcode 0.159/0.482"
