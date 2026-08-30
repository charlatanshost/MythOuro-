#!/usr/bin/env bash
# VERIFY THE CE FINDING — is best_exit_probe reproducible on one checkpoint?
#
#   bash run_verify_ce.sh          # ~10 min, no training
#
# WHY. Every "the pour degraded the model" claim of 2026-08-18 rests on
# best_exit_probe, and that tool's anchor already failed once: step 108,000
# measured 0.348 where an archived run at 108,471 — 471 steps away — had said
# 0.2036. It draws its eval text from STREAMING HF datasets, so if the stream is
# not identical across processes, the whole curve (0.348 / 0.280 / 0.286 / 0.494 /
# 0.561) is partly measuring different TEXT rather than different MODELS.
#
# Meanwhile kd_exhaustion — fixed CACHED held-out sample, the comparable
# instrument — says the opposite: soft-KL to the teacher FELL across the same
# span (drift -0.0881, 19/25 sign-flips, no step change at 116k-120k). Two
# instruments, opposite verdicts. This decides which one to believe.
#
# STEP 1 re-measures step 125,181, ALREADY measured at mean 0.5608 today at 17:45.
#   reproduces within ~0.01  -> the tool is deterministic, the CE curve is REAL,
#       and the 116k->120k step change is a genuine event needing explanation.
#   differs materially       -> the curve was streamed-data noise. Discard it, and
#       with it "the pour is degrading the model". The surviving evidence is
#       raw-frame code L3+ (75 -> 60 -> 55), which the chat-conversion story
#       explains without anything being broken.
#
# STEP 2 tests the conversion story directly. Chat-framed code L3+ is 6.2% at
# 108,471 and 37.5% at 125,181 — a 6x rise with NO chat data in the run. If
# 116,000 sits between them, the model is steadily becoming a chat model, which
# is what distilling against Ouro-2.6B-THINKING should do. Same flags as both
# existing points, so it drops straight into the series.
set -uo pipefail
trap 'pkill -INT -P $$ 2>/dev/null; true' INT TERM
cd "$(dirname "$0")"
source ../venv-xpu/bin/activate
export SYCL_CACHE_PERSISTENT=1 PYTORCH_ALLOC_CONF=expandable_segments:True TRITON_DEFAULT_BACKEND=intel

LOG="logs/verify_ce_$(date +%Y%m%d_%H%M).log"
mkdir -p logs reports

if pgrep -f "training[.](distill|sft|train_depth_policy)" >/dev/null; then
  echo "a trainer is running — it needs the card; stop it first"; exit 1
fi
if pgrep -f "tools[.](code_eval|math_eval|best_exit_probe|onpolicy_rollout_probe|kd_exhaustion)" >/dev/null; then
  echo "another probe is running; let it finish"; exit 1
fi

{
echo "=== 1/2  REPEAT best_exit on step 125,181 (first run today: mean 0.5608) ==="
python -u -m tools.best_exit_probe -c checkpoints_newmix/step_0125181.pt \
  --device xpu:0 --n-loops 8 --trained-loops 4 --by-domain \
  --json reports/best_exit_125181_repeat.json

echo; echo "=== 2/2  chat-framed code at 116,000 (108,471 = 6.2%, 125,181 = 37.5%) ==="
python -u -m tools.code_eval -c checkpoints_newmix/step_0116000.pt --device xpu:0 \
  --samples 8 --temperature 0.4 --seed 1234 --repetition-penalty 1.15 \
  --chat-template --extract --json reports/code_eval_116000_CHAT.json
} 2>&1 | tee "$LOG"

echo
echo "=== REPRODUCIBILITY VERDICT ==="
python3 - <<'PY'
import json
a=json.load(open("reports/best_exit_125181.json"))["by_domain"]
b=json.load(open("reports/best_exit_125181_repeat.json"))["by_domain"]
A={e["domain"]:e["ce_trained_depth"] for e in a}
B={e["domain"]:e["ce_trained_depth"] for e in b}
print(f"  {'domain':14}{'run 1':>9}{'run 2':>9}{'diff':>9}")
for d in A:
    print(f"  {d:14}{A[d]:9.4f}{B[d]:9.4f}{B[d]-A[d]:+9.4f}")
ma,mb=sum(A.values())/len(A),sum(B.values())/len(B)
print(f"  {'MEAN':14}{ma:9.4f}{mb:9.4f}{mb-ma:+9.4f}")
print()
if abs(mb-ma)<0.02:
    print("  REPRODUCIBLE -> the CE curve is real. The 116k->120k step change is an")
    print("  event that needs explaining, and 'the pour degraded the model' stands.")
else:
    print("  NOT REPRODUCIBLE -> the tool's eval text varies between runs. The whole")
    print("  0.348/0.280/0.286/0.494/0.561 curve is unusable, and every conclusion")
    print("  drawn from it today is void. kd_exhaustion becomes the only CE-shaped")
    print("  evidence, and it says the student-teacher gap NARROWED.")
PY
echo
echo "Then read chat-framed 116,000 against 6.2% (108,471) and 37.5% (125,181)."
