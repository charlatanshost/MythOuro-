#!/usr/bin/env bash
# OPT-B BENCHMARK — turn a projection into a measurement, in stages.
#
#   bash run_optb_bench.sh            # STAGE 1: apply B, measure top-K mass. STOPS.
#
# ⚠ 2026-09-11, first attempt, two bugs in this script and its tool:
#   * stage 1 died on usage text — --report-only still required --out. K=32 was
#     then chosen blind, the exact thing staging exists to prevent. Fixed in the
#     tool: --out is optional under --report-only.
#   * stage 2 hung in the teacher load: 17 min at 100% CPU on one core, nothing
#     written. The tool used a bare AutoModelForCausalLM; the trainer's
#     load_distillation_teacher (pad_token_id fix, eager attention on XPU, move
#     after load) brings the same teacher up in ~13 s. Fixed: one loader.
#   K=32 bash run_optb_bench.sh       # STAGE 2+3: build a small cache, then 200 steps
#
# WHY THIS MATTERS MORE THAN IT LOOKS. 2026-09-10 we costed a bigger student —
# dim 1792 + prelude/coda 4 + top-k 6 = 633M total / 460M activated, 2.55x the
# current 180.6M activated. Naively that is 2.55x slower: 12.4 -> ~32 s/step, or
# ~26h per 3,000-step leg. Three nights per experiment, which is not a project.
#
# But 78.4% of a step is the TEACHER forward, and that does not scale with
# student size. If B removes it:
#
#     today          12.4 s/step   = teacher ~9.7 + backward 1.7 + student 0.8
#     +B, 278M       ~3.4 s/step   teacher gone from the offline path
#     +B, 2.55x      ~8.3 s/step   only backward + student forward scale
#
# ⇒ a 2.55x bigger student with B would be FASTER per step than what we train
# today. That is the entire argument for scaling up on this rig, and it rests on
# a projection that has never been run. This measures it.
#
# ⚠ A IS ALREADY APPLIED (3 OPT-A markers in distill.py, .preA backup present),
# so the 12.4 s/step measured on instruct legs 1 and 2 is the **+A** number.
# The README's 11.5 / 5.77 / 3.30 table was measured on the 48-EXPERT config at
# different micro-batch and grad-accum — do NOT compare against those absolutes.
# The transferable claim is the ratio: teacher_fwd is 78.4% of a step and B
# removes 92.8% of the teacher's token-work (the offline path).
#
# ⚠⚠ B IS AN APPROXIMATION, AND SPEED IS NOT THE GATE. It computes the
# divergence on a coarsened event space (K symbols + one lumped tail), which by
# the data-processing inequality is a LOWER BOUND on the true KL. A trains
# against the same objective; B trains against a softened one. `test_optB.py`
# shows it reproduces `distillation_loss` exactly at K=V and is monotone in K.
# So the speed number below is necessary but NOT sufficient — the real decision
# needs a quality comparison at matched steps, which is a separate leg.
#
# ⚠ B ALSO FIXES THE OFFLINE CORPUS. A cache is finite where streaming
# fineweb-edu is not, and re-reading it past ~1.35 epochs is the dose that cost
# 6.2pp of code L3+ in the chat-mix post-mortem — the same dose ceiling leg 2
# just re-confirmed on clean data. Size any real cache against the leg length.
set -uo pipefail
trap 'pkill -INT -P $$ 2>/dev/null; true' INT TERM
cd "$(dirname "$0")"
source ../venv-xpu/bin/activate
export TRITON_DEFAULT_BACKEND=intel
unset SYCL_CACHE_PERSISTENT PYTORCH_ALLOC_CONF
export PYTHONFAULTHANDLER=1 PYTHONUNBUFFERED=1
export SYCL_QUEUE_THREAD_POOL_SIZE=1
export ZE_SERIALIZE=2

TEACHER=ByteDance/Ouro-2.6B-Thinking
FILES="data_teacher_code/shard_*.jsonl,data_teacher_math/shard_*.jsonl,data_teacher_v2/shard_*.jsonl,data_teacher_med/shard_*.jsonl"
mkdir -p logs reports

if pgrep -f "python -u -m training\.(distill|sft)" >/dev/null; then
  echo "a trainer is running — stop it first"; exit 1; fi

# ---- apply B if it is not already in the tree -------------------------------
if ! grep -q "sparse_kd\|teacher-logit-cache" training/distill.py; then
  echo "=== applying OPT-B (A is already present; backups are .preA/.preB) ==="
  bash optim/apply.sh B || { echo "apply failed — tree unchanged"; exit 1; }
else
  echo "=== OPT-B already applied ==="
fi

# ---- STAGE 1: choose K by measurement, not taste ----------------------------
if [ -z "${K:-}" ]; then
  echo
  echo "=== STAGE 1: how much probability mass does top-K capture? ==="
  python -u -m tools.precompute_teacher_logits --report-only \
    --files "$FILES" --teacher-id "$TEACHER" --trust-remote-code \
    --device xpu:0 --seq-len 1024 --sample-rows 64 \
    2>&1 | tee -a logs/optb_report.log
  echo
  echo "=== READ THE MASS FIGURE, THEN RE-RUN WITH K ==="
  echo "  Below ~99% captured, the lumped tail carries real signal and K should"
  echo "  go up. Storage is 4K+4 bytes/token: K=16 -> 68 B/tok, K=32 -> 132,"
  echo "  K=64 -> 260. The 200-step bench below needs only ~3.3M tokens, so even"
  echo "  K=64 is well under 1 GB — pick for QUALITY here, not for disk."
  echo
  echo "    K=32 bash run_optb_bench.sh"
  exit 0
fi

# ---- STAGE 2: build a cache just big enough for the bench -------------------
CACHE="data_teacher_logits_k${K}_bench"
STEPS="${STEPS:-200}"
# 200 steps x micro_batch 2 x grad_accum 8 x seq 1024 = 3.28M tokens; take 2x
# headroom so the loader never wraps and re-reads during the measurement.
MAXTOK="${MAXTOK:-7000000}"
# ⚠ manifest.json is the completion marker, not the directory. 2026-09-11 a
# stage-2 run hung in the teacher load and was killed with the dir created but
# empty; a "non-empty dir" test would also be fooled by a partial run that
# wrote some shards and died.
if [ ! -f "$CACHE/manifest.json" ]; then
  rm -rf "$CACHE"
  echo "=== STAGE 2: precompute top-$K teacher logits (~${MAXTOK} tokens) ==="
  echo "=== this is a GPU job — forward-only at batch 8, but not free ==="
  python -u -m tools.precompute_teacher_logits \
    --files "$FILES" --teacher-id "$TEACHER" --trust-remote-code \
    --device xpu:0 --seq-len 1024 --top-k "$K" --batch 8 \
    --max-tokens "$MAXTOK" --out "$CACHE" \
    2>&1 | tee -a "logs/optb_precompute_k${K}.log"
  [ -f "$CACHE/manifest.json" ] || { echo "precompute did not finish — no manifest"; exit 1; }
else
  echo "=== STAGE 2: reusing existing $CACHE (manifest present) ==="
fi
du -sh "$CACHE" | sed 's/^/  cache size: /'

# ---- STAGE 3: measure s/step against the +A baseline ------------------------
DIR=checkpoints_optb_bench
SRC=checkpoints_exitpdf/step_0007200.pt
LOG="logs/optb_bench_k${K}_$(date +%Y%m%d_%H%M).log"
mkdir -p "$DIR"
[ -f "$DIR/step_0000000.pt" ] || { cp "$SRC" "$DIR/step_0000000.pt"
  python - <<'PY'
import torch, os
p="checkpoints_optb_bench/step_0000000.pt"
ck=torch.load(p,map_location="cpu",weights_only=False); ck["step"]=0
torch.save(ck,p+".tmp"); os.replace(p+".tmp",p); print("  seeded at step 0")
PY
}

echo
echo "=== STAGE 3: $STEPS steps with the cache. BASELINE TO BEAT: 12.4 s/step (+A) ==="
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
  --teacher-logit-cache "$CACHE" \
  --onpolicy-lambda 0.7 \
  --ckpt-dir "$DIR" --ckpt-every-mins 999 --ckpt-milestone-every 100000 \
  --num-workers 0 --trust-remote-code --log-every 25 \
  --total-steps "$STEPS" \
  > >(tee -a "$LOG") 2> >(tee -a "$LOG.err" >&2)

echo
echo "=== RESULT ==="
python3 - "$LOG.err" <<'PY'
import re, sys
from datetime import datetime as dt
rows=[]
for ln in open(sys.argv[1], errors="ignore"):
    m=re.search(r"^(\S+ \S+?)\.\d+ .*step\s+(\d+)/", ln)
    if m:
        try: rows.append((dt.strptime(m.group(1), "%Y-%m-%d %H:%M:%S"), int(m.group(2))))
        except ValueError: pass
if len(rows) < 3:
    print("  not enough step lines to time — read the log"); sys.exit()
# skip the first interval: warmup, allocator growth and the first cache read
(t0,s0),(t1,s1) = rows[1], rows[-1]
sec=(t1-t0).total_seconds()/max(s1-s0,1)
print(f"  measured: {sec:.2f} s/step over steps {s0}->{s1} (first interval dropped)")
BASE = 12.34   # measured by THIS parser on logs/instruct2_*.log.err, steps
               # 100->3000. Leg 1 independently gave 12.43 over 2,950 steps.
print(f"  baseline: {BASE} s/step (+A, same config, same parser)")
print(f"  speedup:  {BASE/sec:.2f}x")
print()
print(f"  a 3,000-step leg: {sec*3000/3600:.1f} h   (today: 10.4 h)")
print(f"  projected for a 2.55x-activated student: {(sec-0.3)*2.55+0.3:.1f} s/step"
      f"  -> {((sec-0.3)*2.55+0.3)*3000/3600:.1f} h per leg")
print()
print("  ⚠ SPEED IS NOT THE DECISION. B softens the objective (coarsened event")
print("    space, a lower bound on the true KL). Before adopting it, run a")
print("    quality leg at matched steps from the same checkpoint and compare")
print("    L3+/L4 and the prose readout against an A-only leg. If B is even")
print("    slightly worse, A alone is the safe win — the README's own gate.")
PY
