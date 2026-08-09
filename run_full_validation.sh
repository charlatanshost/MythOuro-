#!/usr/bin/env bash
# FULL VALIDATION BATTERY — every probe we have, one checkpoint, one sitting.
#
#   bash run_full_validation.sh [checkpoint]     # default: newest in checkpoints_newmix
#
# WHY ALL OF THEM. Most sessions run two or three probes and reason from those.
# Four times in the 07-31..08-09 window a conclusion had to be reversed because
# the coverage was partial — count-before-text, one alpha of four, one remembered
# sample vs a full set, a spike called from two points. This runs the complete
# set on one checkpoint so the picture is whole and the comparisons are matched.
#
# NINE INSTRUMENTS, and what each is actually for:
#
#   1 collapse_metrics GREEDY   the HARSH diagnostic. Greedy decode exposes
#                               attractors and the untrained tail that sampling
#                               hides. Pairs with (2): greedy-vs-sampled is how
#                               this project tells an ARTIFACT from a real weakness.
#   2 collapse_metrics SAMPLED  ~real use. Same instrument, T=0.8.
#   3 onpolicy_rollout_probe    the alpha ladder + prompt->output text. THE
#                               reading instrument; --json keeps all n samples
#                               (the .txt keeps only #1, which has misled before).
#   4 math_eval                 capability. copied_from_prompt is the mechanism
#                               metric (echo vs compute); L4 is the hard floor.
#   5 code_eval                 capability, both penalty settings: 1.0 =
#                               degeneracy number, 1.15 = capability number.
#   6 best_exit_probe           is a depth policy learnable, and how much headroom.
#   7 knowledge_probe           does the student CARRY correct domain knowledge,
#                               separate from whether it can generate it.
#   8 knowledge_likelihood      does it KNOW the fact, decoupled from producing it.
#                               (7) and (8) together separate "doesn't know" from
#                               "knows but can't say" — the medical question.
#   9 kd_exhaustion             is there signal LEFT in Ouro? The teacher-parity
#                               gauge that gates Rung 3 of the teacher curriculum.
#  10 per_loop_calibration      UncertaintyHead ECE per loop. Especially worth
#                               re-running now: the depth policy retrained that
#                               head on 2026-08-09, and the P0.5 audit measured
#                               loop 0 at ECE 0.17-0.22 BEFORE that.
#
# Non-fatal: one tool failing must not cost the other eight. Everything is
# logged to a file — results have been lost to scrollback twice.
set -uo pipefail
cd "$(dirname "$0")"
source ../venv-xpu/bin/activate
export SYCL_CACHE_PERSISTENT=1 PYTORCH_ALLOC_CONF=expandable_segments:True TRITON_DEFAULT_BACKEND=intel

CKPT="${1:-$(ls -t checkpoints_newmix/step_*.pt | head -1)}"
DIR="$(dirname "$CKPT")"
STEP="$(basename "$CKPT" | sed 's/step_0*//; s/\.pt//')"
TEACHER=ByteDance/Ouro-2.6B-Thinking
LOG="logs/validation_${STEP}_$(date +%Y%m%d_%H%M).log"
OK=reports/validation_${STEP}_DONE
mkdir -p logs reports; rm -f "$OK"

if pgrep -f "training[.](distill|train_depth_policy)" >/dev/null; then
  echo "a trainer is running — stop it first (these will contend for the card)"; exit 1
fi

echo "FULL VALIDATION @ step $STEP  ($CKPT)" | tee "$LOG"
echo "log: $LOG" | tee -a "$LOG"
PASS=0; FAILED=""

run () {                      # run <name> <artifact-or-EXITCODE> <command...>
  # VERIFY BY ARTIFACT, NOT EXIT CODE. XPU teardown raises
  # "PyGILState_Release: thread state must be current" AFTER the work finishes
  # (field notes workaround #4), so a completed tool routinely exits non-zero.
  # The first battery run scored 2 tools FAILED that had written their JSON and
  # printed their verdict — the same mistake the training scripts already avoid.
  local name="$1" want="$2"; shift 2
  echo -e "\n########## $name ##########" | tee -a "$LOG"
  "$@" >>"$LOG" 2>&1; local rc=$?
  local ok=1
  if [ "$want" = "EXITCODE" ]; then
    [ $rc -eq 0 ] || ok=0
  else
    [ -s "$want" ] || ok=0          # artifact exists and is non-empty
  fi
  if [ $ok -eq 1 ]; then
    PASS=$((PASS+1)); echo "  OK (rc=$rc)" | tee -a "$LOG"
  else
    FAILED="$FAILED $name"; echo "  *** FAILED (rc=$rc, no artifact) ***" | tee -a "$LOG"
  fi
}

run "1 collapse GREEDY" EXITCODE python -u -m tools.collapse_metrics \
    --checkpoint "$CKPT" --device xpu:0 --n-loops 4 --generate --max-new 96
run "2 collapse SAMPLED" EXITCODE python -u -m tools.collapse_metrics \
    --checkpoint "$CKPT" --device xpu:0 --n-loops 4 --generate --max-new 96 --temperature 0.8

run "3 rollout probe (alpha ladder + TEXT)" "reports/probe_${STEP}.json" python -u -m tools.onpolicy_rollout_probe \
    --ckpt-dir "$DIR" --student-device xpu:0 --teacher-device xpu:0 \
    --teacher-id "$TEACHER" --trust-remote-code --no-kv-cache --samples 5 \
    --json "reports/probe_${STEP}.json"

run "4 math_eval" "reports/math_eval_${STEP}.json" python -u -m tools.math_eval -c "$CKPT" --device xpu:0 \
    --samples 8 --seed 1234 --json "reports/math_eval_${STEP}.json"

run "5a code_eval pen=1.0" "reports/code_eval_${STEP}_pen1.0.json" python -u -m tools.code_eval -c "$CKPT" --device xpu:0 \
    --samples 8 --temperature 0.4 --seed 1234 --repetition-penalty 1.0 \
    --json "reports/code_eval_${STEP}_pen1.0.json"
run "5b code_eval pen=1.15" "reports/code_eval_${STEP}_pen1.15.json" python -u -m tools.code_eval -c "$CKPT" --device xpu:0 \
    --samples 8 --temperature 0.4 --seed 1234 --repetition-penalty 1.15 \
    --json "reports/code_eval_${STEP}_pen1.15.json"

run "6 best_exit_probe" "reports/best_exit_${STEP}.json" python -u -m tools.best_exit_probe -c "$CKPT" --device xpu:0 \
    --n-loops 8 --chunks 12 --seq-len 256 --json "reports/best_exit_${STEP}.json"

run "7 knowledge_probe" EXITCODE python -u -m tools.knowledge_probe --ckpt-dir "$DIR" \
    --student-device xpu:0 --teacher-device xpu:0 --teacher-id "$TEACHER" \
    --trust-remote-code
run "8 knowledge_likelihood" EXITCODE python -u -m tools.knowledge_likelihood_probe \
    --ckpt-dir "$DIR" --student-device xpu:0 --n-loops 4

run "9 kd_exhaustion" EXITCODE python -u -m tools.kd_exhaustion --ckpt-dir "$DIR" \
    --device xpu:0 --teacher-id "$TEACHER" --trust-remote-code --batches 8

run "10 per_loop_calibration" "reports/per_loop_ece_${STEP}.json" python -u -m tools.per_loop_calibration \
    --checkpoint "$CKPT" --device xpu:0 --max-samples 20 --seq-len 256 \
    --out "reports/per_loop_ece_${STEP}.json"

echo -e "\n=========================================" | tee -a "$LOG"
echo "PASSED $PASS/11" | tee -a "$LOG"
[ -n "$FAILED" ] && echo "FAILED:$FAILED" | tee -a "$LOG"
echo "validation @$STEP: $PASS/10 passed$FAILED $(date)" > "$OK"
echo "full log: $LOG" | tee -a "$LOG"
cat <<EOF | tee -a "$LOG"

READ IN THIS ORDER:
  1. the TEXT from (3) — reports/probe_${STEP}.json, all 5 samples per alpha.
     Compare with tools/text_diff against the 4-point history.
  2. GREEDY vs SAMPLED (1 vs 2) — a failure in greedy that vanishes under
     sampling is a decode artifact; one that survives both is real.
  3. math copied_from_prompt (41.2% -> 16.2% -> 32.5% so far) and code
     per-sample L3+ at BOTH penalties.
  4. kd_exhaustion (9) — if the teacher signal is flattening, that is the
     Rung-3 gate in docs/teacher_data_curriculum.md, not a training problem.
EOF
