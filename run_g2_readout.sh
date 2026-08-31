#!/usr/bin/env bash
# G2 READOUT — does capacity break the trade pattern? (docs/roadmap.md → G2)
#
#   bash run_g2_readout.sh
#
# THE QUESTION. At 278M every intervention moved one metric up and another down:
# +code corpus bought L4 (5->30/320) and sold L3+ (76.6->54.1); more steps sold
# it back (L4 26->19, L3+ 54.7->77.5). If 397M moves L3+ AND L4 AND prose
# TOGETHER, capacity was the constraint. If it trades the same way, the problem
# is the recipe, not the size — worth knowing before committing months.
#
# The pass/fail bar was pre-registered in docs/roadmap.md BEFORE these numbers
# existed. Do not renegotiate it after seeing them.
#
# WHY step_0000000 IS ALSO EVALUATED. Promotion is bit-exact, so step 0 of the
# grown model computes the SAME function as the 278M base. Measuring it in this
# session gives an internal control at identical settings — immune to the
# cross-session comparability problem the tracker documents for eval tools that
# stream their prompts. If step 0 disagrees with the archived 157,238 numbers,
# trust the within-session pair and say so.
#
# ⚠ CONFOUND, recorded before the measurement: the G0 baselines ran
# --rollout-batch 32; this leg ran 8 (48-expert crash avoidance). Two variables
# moved. State it in whatever conclusion is drawn.
#
# ⚠ tools.code_eval takes ONE checkpoint: `-c` is required=True with no nargs.
# The glob `-c $DIR/step_0*.pt` printed by run_grown48.sh's tail expands to 22
# paths and dies on unrecognised arguments. Hence one invocation per checkpoint.
set -uo pipefail
cd "$(dirname "$0")"
source ../venv-xpu/bin/activate

DIR=checkpoints_grown48
BASE=$DIR/step_0000000.pt          # bit-exact promoted 278M — the internal control
FINAL=$(ls -t $DIR/step_*.pt | head -1)
EV="--device xpu:0 --samples 32 --temperature 0.4 --seed 1234 --repetition-penalty 1.15"
# ^ 10 tasks x 32 samples = n=320, matching the archived G0 baselines exactly
#   (T=0.4 pen=1.15 seed=1234 max_new=96 framing=bare chat=False — verified).
mkdir -p reports logs
LOG="logs/g2_readout_$(date +%Y%m%d_%H%M).log"

if pgrep -f "python -u -m training\.(distill|sft)" >/dev/null; then
  echo "a trainer is running — the eval will contend for the card. Stop it first."; exit 1; fi

echo "=== G2 readout: control $BASE  vs  final $FINAL ==="
echo "=== log: $LOG ==="

python -u -m tools.code_eval -c "$BASE"  $EV --json reports/code_g2_control.json 2>&1 | tee -a "$LOG"
python -u -m tools.code_eval -c "$FINAL" $EV --json reports/code_g2_final.json   2>&1 | tee -a "$LOG"
DIR=$DIR bash run_anneal_readout.sh 2>&1 | tee -a "$LOG"

python3 - <<'PY' 2>&1 | tee -a "$LOG"
import json, os
def read(p):
    d=json.load(open(p))
    n=sum(len(t["samples"]) for t in d["tasks"])
    l4=sum(1 for t in d["tasks"] for s in t["samples"] if s.get("rung",0)>=4)
    g=d["diagnostics"]
    return dict(step=d["step"], n=n, l3p=g.get("per_sample_l3plus"),
                ci=g.get("per_sample_l3plus_ci95"), l4=l4,
                degen=g.get("adj_degenerate"))
rows=[("G0 157,238 (archived)","reports/code_mathcode_157238.json"),
      ("G0 163,238 (archived)","reports/code_mathcode_163238.json"),
      ("397M step 0 (control)","reports/code_g2_control.json"),
      ("397M final","reports/code_g2_final.json")]
print("\n" + "="*74)
print("  G2 READOUT".center(74))
print("="*74)
print(f"  {'checkpoint':24s} {'n':>4}  {'L3+':>7}  {'L4':>9}  {'degen':>5}")
got={}
for label,p in rows:
    if not os.path.exists(p): print(f"  {label:24s}  (missing {p})"); continue
    r=read(p); got[label]=r
    print(f"  {label:24s} {r['n']:>4}  {r['l3p']*100:6.1f}%  {r['l4']:>4}/{r['n']:<4} {r['degen']:>5}")
f=got.get("397M final")
if f:
    L3_BAR, L4_BAR = 0.775, 26          # pre-registered: 163,238's L3+ and 157,238's L4
    up3, up4 = f["l3p"] >= L3_BAR, f["l4"] > L4_BAR
    print("\n  pre-registered bar: L3+ >= 77.5%  AND  L4 > 26/320  AND  prose <= 0.150")
    print(f"    L3+ {f['l3p']*100:.1f}%  {'PASS' if up3 else 'below bar'}")
    print(f"    L4  {f['l4']}/{f['n']}    {'PASS' if up4 else 'below bar'}")
    print(f"    prose -> read the salad score printed by run_anneal_readout.sh above")
    print("\n  VERDICT (needs the prose number too):")
    if up3 and up4:   print("    both code metrics cleared TOGETHER — if prose held, this is a G2 PASS.")
    elif up3 or up4:  print("    one up, one down — the TRADE PATTERN PERSISTS. G2 PARTIAL:")
    else:             print("    neither cleared — G2 FAIL: recipe, not size.")
    print("\n  ⚠ rollout-batch confound: baselines ran 32, this leg ran 8.")
print("="*74)
PY
