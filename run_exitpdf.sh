#!/usr/bin/env bash
# EXIT_PDF — is our accuracy wall the OBJECTIVE rather than the parameter count?
#
#   bash run_exitpdf.sh          # ~8h, 3,000 steps
#
# THE CASE, from docs/looped_lm_landscape.md §0.1 (owner-supplied literature):
#   * recurrent-depth-ttc: iterative-target supervision extrapolates 24x beyond
#     trained depth; FINAL-ONLY SUPERVISION CAUSES ACCURACY WALLS.
#   * Ouro (arXiv 2510.25741) — OUR TEACHER — trains L = Σ_t p(t|x)·L^(t).
#   * RLTT (arXiv 2602.10520): distributing credit across the latent trajectory
#     beats terminal-only by +5.8% (1.4B) / +10.9% (2.6B), measured on Ouro.
#
# MythOuro supervises FINAL-ONLY (h_K) — not by choice, but because the
# ACT-weighted sum let the optimiser pin λ₀≈1 and collapse depth. And we show the
# predicted signature: halt depth pinned at EXACTLY 2.00/4 on every sample of
# every domain, the --n-loops sweep flat at 4/6/8, L4 never clearing ~9%.
#
# The landscape doc's own reading: "depth is not dead, it is walled, the wall is
# the objective, and rung 5 (grow depth) treats the symptom while rung 3 treats
# the cause." We spent 2026-08-27..09-01 growing WIDTH instead.
#
# WHY RUNG 3 IS NOT ACTUALLY CLOSED. It tested `uniform` ONLY — the harshest arm,
# which hands loop 0 (never an output state) 25% of the gradient. It inverted the
# depth trajectory (loop 3: 0.980 -> 0.075), which is what forcing loop 0 to
# match loop 3 should do. `exit_pdf` — documented in-flag as "the model's own halt
# distribution, i.e. Ouro's p(t|x) exactly" — HAS NEVER BEEN RUN.
#
# ⚠ THE COUPLED KNOB, AND WHY THIS IS A TWO-SIDED EXPERIMENT.
# run_loopweighted.sh skipped exit_pdf deliberately: --depth-reg-coeff 0.3 is a
# KL from the halt distribution toward UNIFORM, so exit_pdf would be dragged
# toward uniform anyway. And with halt PINNED at 2.00, exit_pdf concentrates
# nearly all weight on loop 2 — i.e. loop-2-only, not distributed supervision.
# So depth-reg must come down for exit_pdf to mean anything. But depth-reg is
# exactly what prevents the ACT loop-collapse that made us supervise h_K in the
# first place. The two pull against each other.
#
# 0.3 -> 0.1 here: 0.1 is the documented v3-onward default (failure_modes.md),
# not a guess. Do NOT go to 0 in the same run — that changes both knobs at once.
#
# THE GATE, pre-registered by the landscape doc itself:
#   halt distribution MOVES off 2.00 and L4/L3+ hold or rise
#       -> the wall was the objective. Rung 3 reopens, depth becomes the axis,
#          and the whole capacity argument needs revisiting.
#   halt COLLAPSES toward loop 0
#       -> vindicates Silent Thinking's final-only position; depth-reg was doing
#          necessary work; revert and stop asking.
#   halt stays pinned at 2.00
#       -> exit_pdf is degenerate under our halt head; the halt head is the
#          blocker, not the weighting. That points at the hardcoded-halt-rule
#          fallback (recurrent-depth-ttc) rather than more objective work.
set -uo pipefail
STOP=0
trap 'STOP=1; pkill -INT -P $$ 2>/dev/null; true' INT TERM
cd "$(dirname "$0")"
source ../venv-xpu/bin/activate
export TRITON_DEFAULT_BACKEND=intel
unset SYCL_CACHE_PERSISTENT PYTORCH_ALLOC_CONF
export PYTHONFAULTHANDLER=1 PYTHONUNBUFFERED=1

# STEPS override so the night can be de-risked with a short pilot first:
#   STEPS=1200 bash run_exitpdf.sh    # ~1.7h — does the halt distribution MOVE?
#   bash run_exitpdf.sh               # continues by 3,000 from wherever it is
STEPS="${STEPS:-3000}"
DIR=checkpoints_exitpdf
SRC=checkpoints_pruned24/step_0000000.pt
TEACHER=ByteDance/Ouro-2.6B-Thinking
FILES='data_teacher_code/shard_*.jsonl,data_teacher_math/shard_*.jsonl,data_teacher_v2/shard_*.jsonl,data_teacher_med/shard_*.jsonl'
LOG="logs/exitpdf_$(date +%Y%m%d_%H%M).log"
mkdir -p logs reports "$DIR"

if pgrep -f "python -u -m training\.(distill|sft)" >/dev/null; then
  echo "a trainer is already running"; exit 1; fi
[ -f "$SRC" ] || { echo "missing $SRC — run tools/prune_masked_experts.py"; exit 1; }
[ -f "$DIR/step_0000000.pt" ] || cp "$SRC" "$DIR/step_0000000.pt"

at=$(basename "$(ls -t $DIR/step_*.pt | head -1)" | sed 's/step_0*//; s/\.pt//'); at=${at:-0}
echo "=== exit_pdf: $at -> $((at+STEPS))  (24 experts, depth-reg 0.3 -> 0.1) ==="
echo "=== WATCH THE HALT DISTRIBUTION. That is the experiment. ==="
echo "=== log: $LOG ==="

python -u -m training.distill \
  --student-variant mythouro_distill_tiny \
  --student-device xpu:0 --teacher-device xpu:0 --teacher-id "$TEACHER" \
  --seq-len 1024 --micro-batch 8 --grad-accum 2 \
  --warmup-steps 500 --lr 1e-4 --min-lr 3e-5 --start-loops 4 \
  --loop-loss-weighting exit_pdf \
  --depth-reg-coeff 0.1 --divergence rev_kl \
  --use-sandwich-norm --use-depth-aware-init \
  --teacher-mix-alpha 0.45 --rollout-len 64 --rollout-batch 32 --rollout-reuse 8 \
  --teacher-data-ratio 0.2 --teacher-data-files "$FILES" \
  --onpolicy-lambda 0.7 \
  --ckpt-dir "$DIR" \
  --ckpt-every-mins 15 --ckpt-milestone-every 500 --keep-last 8 \
  --num-workers 0 --trust-remote-code --log-every 50 \
  --total-steps $((at+STEPS)) \
  > >(tee -a "$LOG") 2> >(tee -a "$LOG.err" >&2)

echo
echo "=== HALT DEPTH over the run — the gate ==="
grep -ohE "n_loops [0-9]+ \| op [0-9]+/[0-9]+" "$LOG.err" | tail -3 | sed 's/^/  /'
grep -ohE "loop_efficiency[= ][0-9.]+|halt [0-9.]+/[0-9]+" "$LOG.err" | tail -8 | sed 's/^/  /'
echo
echo "  Then read BOTH instruments over >=3 checkpoints:"
echo "    bash run_eval.sh $DIR/step_0003000.pt exitpdf_3000"
echo "    bash run_prose_readout.sh $DIR/step_0003000.pt $DIR/step_0002500.pt $DIR/step_0002000.pt"
