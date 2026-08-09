#!/usr/bin/env bash
# THE POUR — first training on the corrected data mix, at 2.3x throughput.
#
#   bash run_newmix_pour.sh          # Ctrl-C whenever; resuming is free
#
# WHAT CHANGED, and why this run is the test of it. Three separate data defects
# were found and fixed on 2026-07-31, none of which the model has trained on yet:
#
#   medical  PROMOTED from harvest-only into the mix (adaef8d). It had NEVER read
#            a real PubMed abstract — every medical token it ever saw was Ouro's
#            continuation from a seed, which fabricates ("the PAWL study",
#            30-mg ibuprofen; real dosing is 200-400mg).
#   math     open-web-math is DISCOURSE: 76% LaTeX, 71% equations, but only 2%
#            ever mark an answer. Split 0.26 -> 0.13 raw + 0.13 OpenMathInstruct-2
#            (problem -> worked solution -> \boxed answer).
#   code     codeparrot-clean is whole FILES. Split 0.27 -> 0.135 raw + 0.135
#            OpenCodeInstruct (instruction -> unit-tested body).
#
# Shape splits, NOT a rebalance: math and code keep their total share, so if the
# evals move we know it was shape rather than dose.
#
# BASELINES ALREADY MEASURED at step 70,500, all seeded (1234) and reproducible:
#   math per-sample L4   0.0%  (copied-from-prompt 41.2%, median rel_err 0.400)
#   code per-sample L3+  51.2% bare / 48.8% file-framed
#   text  reports/text_diff_history_alpha00.md
#
# WHAT IS ALREADY RULED OUT, so nobody re-litigates it mid-run: decoding (a
# repetition penalty removes looping and L4 does not move), framing (file context
# matches training shape, L4 does not move), and recurrent depth (forced 8 passes
# changes nothing; the model halts at the 3rd pass of 4). What remains is DATA or
# CAPACITY — and this run tests data, which is the cheaper of the two.
#
# NEW DIRECTORY on purpose. checkpoints_reuse8 stays frozen as the pre-change
# line so the comparison is clean, and step_0070500.pt remains the fallback.
#
# TARGET 100,000 deliberately overshoots (~30h at 4.81 s/step). A chain that
# FINISHES early leaves the card idle, which is the real constraint on a
# dual-boot rig. Stop it whenever; checkpoints flush and re-running resumes.
set -uo pipefail
cd "$(dirname "$0")"
source ../venv-xpu/bin/activate
export SYCL_CACHE_PERSISTENT=1 PYTORCH_ALLOC_CONF=expandable_segments:True TRITON_DEFAULT_BACKEND=intel

SRC=checkpoints_reuse8/step_0070500.pt
DIR=checkpoints_newmix
TEACHER=ByteDance/Ouro-2.6B-Thinking
FILES='data_teacher_v2/shard_*.jsonl,data_teacher_med/shard_*.jsonl'
TARGET=140000
OK=reports/newmix_pour_DONE; FAIL=reports/newmix_pour_FAILED
rm -f "$OK" "$FAIL"
trap '[ -f "$OK" ] || echo "incomplete exit=$? $(date)" > "$FAIL"' EXIT

step_of() { basename "$1" | sed 's/step_0*//; s/\.pt//'; }
latest()  { ls -t "$1"/step_*.pt 2>/dev/null | head -1; }

# Refuse to start if a trainer is already alive — two processes writing one ckpt
# dir corrupts the run and the symptom appears hours later.
if pgrep -f "training[.]distill" >/dev/null; then
  echo "a trainer is already running; refusing to start a second"; exit 1
fi

mkdir -p "$DIR"
[ -n "$(latest "$DIR")" ] || cp "$SRC" "$DIR/" || {
  echo "cannot seed $DIR from $SRC"; echo "seed failed $(date)" > "$FAIL"; exit 1; }

at=$(step_of "$(latest "$DIR")")
echo "=== POUR on the corrected mix: $at -> $TARGET  (Ctrl-C is safe) ==="
echo "=== mix: general .27 | math .13 + math_instruct .13 | code .135 + code_instruct .135 | medical .20 ==="

python -u -m training.distill \
  --student-variant mythouro_distill_tiny \
  --student-device xpu:0 --teacher-device xpu:0 --teacher-id "$TEACHER" \
  --seq-len 1024 --micro-batch 8 --grad-accum 2 \
  --warmup-steps 500 --lr 1e-4 --min-lr 3e-5 \
  --depth-reg-coeff 0.3 --divergence rev_kl \
  --use-sandwich-norm --use-depth-aware-init \
  --teacher-mix-alpha 0.5 --rollout-len 64 --rollout-batch 32 \
  --rollout-reuse 8 \
  --teacher-data-ratio 0.2 --teacher-data-files "$FILES" \
  --onpolicy-lambda 0.7 --ckpt-dir "$DIR" \
  --ckpt-every-mins 15 --ckpt-milestone-every 2000 --keep-last 5 \
  --num-workers 0 --trust-remote-code --log-every 50 \
  --total-steps "$TARGET" || true              # exit code NOT trusted (XPU teardown)

at=$(step_of "$(latest "$DIR")")
echo "pour reached $at $(date)" > "$OK"
echo "=== stopped at step $at ==="
cat <<'EOF'

WHEN YOU COME BACK — same seed and settings as the baselines, so it is directly
comparable:

  python -u -m tools.math_eval -c checkpoints_newmix/step_00XXXXX.pt \
    --device xpu:0 --samples 8 --seed 1234 --json reports/math_eval_XXXXX.json
  python -u -m tools.code_eval -c checkpoints_newmix/step_00XXXXX.pt \
    --device xpu:0 --samples 8 --temperature 0.4 --seed 1234 \
    --json reports/code_eval_XXXXX.json
  python -u -m tools.onpolicy_rollout_probe --ckpt-dir checkpoints_newmix \
    --student-device xpu:0 --teacher-device xpu:0 --teacher-id ByteDance/Ouro-2.6B-Thinking \
    --trust-remote-code --no-kv-cache --samples 5 \
    --json reports/probe_XXXXX.json | tee reports/probe_XXXXX.txt

Use --json on the probe: the .txt keeps only sample #1 of n.
Compare against math 0.0% L4 / code 51.2% L3+ at step 70,500.
EOF
