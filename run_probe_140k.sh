#!/usr/bin/env bash
# END-OF-POUR PROBE — the two readings that decide what happens next.
#
#   bash run_probe_140k.sh          # ~15 min, no training
#
# TWO QUESTIONS, AND THEY MUST BE READ TOGETHER.
#
# 1. IS THE PHASE TRANSITION STILL RUNNING? Between ~116,000 and 120,000 the pour
#    crossed from a continuation model into a chat model. Chat-framed code L3+ has
#    gone 6.2% -> 10.0% -> 37.5% while raw-framed fell 75.0% -> 60.0% -> 55.0%, on
#    the same weights. STILL CLIMBING means more steps is the cheapest win on the
#    board — 140,000 is an arbitrary round number (TARGET=140000, no rationale in
#    any doc), not a principled stop. FLATTENING means the conversion has run its
#    course and the instruction corpus finally has a base worth using.
#    ⚠ Report BOTH frames. Either alone inverts the verdict.
#
# 2. SHOULD alpha ANNEAL 0.5 -> 0.45? The 2026-06-30 trigger was: capability
#    present at high alpha but NOT internalized into alpha=0.0, with alpha=0.0
#    flat. By that rule the condition is already met — alpha=0.0 top_share over
#    35,000 steps: 0.161 (90,351) / 0.201 (100,000) / 0.170 (108,471) / 0.155
#    (125,181). Flat. The 0.6->0.5 anneal moved it 0.18 -> 0.12 in 216 STEPS.
#    onpolicy_plan.md already names the next value (0.45) and shelved it only
#    because the decision then was "HOLD 0.5, pour TOKENS". Those tokens are poured.
#
# ⚠ WHY THEY ARE READ TOGETHER. The rollout probe measures CONTINUATION-frame
# generation at alpha=0.0 — the frame the model is LEAVING. So a flat alpha=0.0
# may be frame drift rather than a true plateau. If chat is still climbing steeply,
# that argues for taking the anneal AND the extra steps in one leg (which is what
# the last anneal did: changed alpha mid-run at 7,242, probed 216 steps later)
# rather than treating flat alpha=0.0 as a stall that needs its own experiment.
#
# BASELINES — all same seed/settings, so these drop straight into the series:
#   raw code pen 1.15   75.0 (108,471) / 60.0 (120,000) / 55.0 (125,181)
#   chat code (extract)  6.2 (108,471) / 10.0 (116,000) / 37.5 (125,181)
#   alpha=0.0 top_share 0.161 / 0.201 / 0.170 / 0.155  (90,351 -> 125,181)
set -uo pipefail
trap 'pkill -INT -P $$ 2>/dev/null; true' INT TERM
cd "$(dirname "$0")"
source ../venv-xpu/bin/activate     # `python` does not exist outside the venv
export SYCL_CACHE_PERSISTENT=1 PYTORCH_ALLOC_CONF=expandable_segments:True TRITON_DEFAULT_BACKEND=intel

DIR=checkpoints_newmix
TEACHER=ByteDance/Ouro-2.6B-Thinking
LOG="logs/probe_140k_$(date +%Y%m%d_%H%M).log"
mkdir -p logs reports

if pgrep -f "training[.](distill|sft|train_depth_policy)" >/dev/null; then
  echo "a trainer is running — it needs the card; stop it first"; exit 1
fi
if pgrep -f "tools[.](code_eval|math_eval|best_exit_probe|onpolicy_rollout_probe|kd_exhaustion)" >/dev/null; then
  echo "another probe is running; let it finish"; exit 1
fi

CKPT=$(ls -t "$DIR"/step_*.pt | head -1)
S=$(basename "$CKPT" | sed 's/step_0*//; s/\.pt//')

{
echo "=== END-OF-POUR PROBE @ step $S   $(date) ==="

echo; echo "--- 1/3  RAW-framed code (the frame being LEFT)  vs 75.0 / 60.0 / 55.0 ---"
python -u -m tools.code_eval -c "$CKPT" --device xpu:0 \
  --samples 8 --temperature 0.4 --seed 1234 --repetition-penalty 1.15 \
  --json "reports/code_eval_${S}_pen115.json"

echo; echo "--- 2/3  CHAT-framed code (the frame being ENTERED)  vs 6.2 / 10.0 / 37.5 ---"
python -u -m tools.code_eval -c "$CKPT" --device xpu:0 \
  --samples 8 --temperature 0.4 --seed 1234 --repetition-penalty 1.15 \
  --chat-template --extract --json "reports/code_eval_${S}_CHAT.json"

echo; echo "--- 3/3  rollout probe: alpha=0.0 is the ANNEAL TRIGGER (teacher loads here) ---"
python -u -m tools.onpolicy_rollout_probe --ckpt-dir "$DIR" \
  --student-device xpu:0 --teacher-device xpu:0 --teacher-id "$TEACHER" \
  --trust-remote-code --no-kv-cache --samples 5 \
  --json "reports/probe_${S}.json" | tee "reports/probe_${S}.txt"
} 2>&1 | tee "$LOG"

echo
echo "================ BOTH SERIES ================"
python3 - "$S" <<'PY'
import json,glob,re,statistics as st,sys
S=sys.argv[1]
def l3(p):
    try: return 100*json.load(open(p))["diagnostics"]["per_sample_l3plus"]
    except Exception: return None
raw={108471:75.0,120000:60.0,125181:55.0}; chat={108471:6.2,116000:10.0,125181:37.5}
r=l3(f"reports/code_eval_{S}_pen115.json"); c=l3(f"reports/code_eval_{S}_CHAT.json")
if r: raw[int(S)]=round(r,1)
if c: chat[int(S)]=round(c,1)
print("\n  code L3+ by frame")
print(f"  {'step':>9}{'RAW':>9}{'CHAT':>9}")
for k in sorted(set(raw)|set(chat)):
    print(f"  {k:>9}{raw.get(k,''):>9}{chat.get(k,''):>9}")

print("\n  alpha=0.0 (pure student) — the anneal trigger")
print(f"  {'step':>9}{'top_share':>11}{'distinct1':>11}")
for f in sorted(glob.glob("reports/probe_*.json"), key=lambda p:int(re.search(r"probe_(\d+)",p).group(1))):
    st_,d1=[],[]
    d=json.load(open(f))
    if "seeds" not in d: continue
    for _,b in d["seeds"].items():
        z=b.get("0.0")
        if z: st_+=z["top_share"]; d1+=z["distinct1"]
    if st_: print(f"  {int(re.search(r'probe_(\\d+)',f).group(1)):>9}{st.mean(st_):11.3f}{st.mean(d1):11.3f}")
PY
cat <<'EOF'

DECIDE:
  CHAT still climbing steeply -> extend the pour past 140,000, and take the
      alpha 0.5 -> 0.45 anneal IN THE SAME LEG (precedent: alpha changed mid-run
      at 7,242, probed 216 steps later). Expect loss to RISE — documented as
      expected/good, not a regression. Watch fragile seeds for re-collapse; last
      time the bacterial/LaTeX seed de-fragilized 0.47 -> 0.18, which is what
      made the call safe.
  CHAT flattening -> the conversion is done. The 26,130-row instruction corpus
      in data_teacher_chat/ now has a base worth using, and re-running chat-mix
      on a converted base is a genuinely different experiment from the two legs
      that failed on 108,471.
  alpha=0.0 STILL FLAT either way -> the anneal is warranted on its own terms.
      It is flat across 35,000 steps against a 216-step precedent.
EOF
