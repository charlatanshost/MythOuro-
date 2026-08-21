#!/usr/bin/env bash
# ANNEAL READOUT — α=0.0 is the verdict, not the loss.
#
#   bash run_anneal_readout.sh        # ~5 min
#
# The 0.6→0.5 anneal was judged on exactly this: α=0.0 top_share fell 0.18 → 0.12
# in ~216 steps, 4/6 seeds improved, and the fragile bacterial/LaTeX seed
# DE-fragilized (0.47 → 0.18) rather than re-collapsing. That is the shape to
# look for. This leg ran 3,500 steps at α=0.45, so it has had 16x the exposure
# the last verdict was called on.
#
# BASELINE @140,000 (α=0.5): top_share 0.154 | distinct1 0.477 | halt 2.00/4
# and the flat history it sits in: 0.161 / 0.201 / 0.170 / 0.155 / 0.154
# across 90,351 → 140,000 — 50,000 steps of no movement.
set -uo pipefail
trap 'pkill -INT -P $$ 2>/dev/null; true' INT TERM
cd "$(dirname "$0")"
source ../venv-xpu/bin/activate
export SYCL_CACHE_PERSISTENT=1 PYTORCH_ALLOC_CONF=expandable_segments:True TRITON_DEFAULT_BACKEND=intel
mkdir -p logs reports

if pgrep -f "training[.](distill|sft)" >/dev/null; then
  echo "a trainer is running — it needs the card"; exit 1
fi

DIR="${DIR:-checkpoints_anneal045}"   # override: DIR=checkpoints_anneal040 bash ...
S=$(basename "$(ls -t $DIR/step_*.pt | head -1)" | sed 's/step_0*//; s/\.pt//')
TAG=$(basename "$DIR" | sed "s/^checkpoints_//")   # report prefix follows DIR
echo "=== anneal readout @ step $S (α=0.45) vs 140,000 (α=0.5) ==="

python -u -m tools.onpolicy_rollout_probe --ckpt-dir "$DIR" \
  --student-device xpu:0 --teacher-device xpu:0 \
  --teacher-id ByteDance/Ouro-2.6B-Thinking \
  --trust-remote-code --no-kv-cache --samples 5 \
  --json "reports/probe_${TAG}_${S}.json" \
  2>&1 | tee "reports/probe_${TAG}_${S}.txt"

echo
echo "=== α=0.0 PER SEED — re-collapse is the abort signal ==="
python3 - "$S" "$TAG" <<'PY'
import json,sys,statistics as st
S,TAG=sys.argv[1],sys.argv[2]
new=json.load(open(f"reports/probe_{TAG}_{S}.json"))
old=json.load(open("reports/probe_140000.json"))
def per_seed(d):
    out={}
    for seed,b in d["seeds"].items():
        z=b.get("0.0")
        if z: out[seed]=(st.mean(z["top_share"]), st.mean(z["distinct1"]))
    return out
a,b=per_seed(old),per_seed(new)
print(f"  {'seed':34}{'@140k':>9}{'@'+S:>9}{'diff':>9}")
worse=[]
for k in a:
    if k not in b: continue
    print(f"  {k[:32]:34}{a[k][0]:9.3f}{b[k][0]:9.3f}{b[k][0]-a[k][0]:+9.3f}")
    if b[k][0] > 0.40: worse.append(k)
ma=st.mean([v[0] for v in a.values()]); mb=st.mean([v[0] for v in b.values()])
da=st.mean([v[1] for v in a.values()]); db=st.mean([v[1] for v in b.values()])
print(f"  {'MEAN top_share':34}{ma:9.3f}{mb:9.3f}{mb-ma:+9.3f}")
print(f"  {'MEAN distinct1':34}{da:9.3f}{db:9.3f}{db-da:+9.3f}")
# ── ACRONYM-SALAD DETECTOR (added 2026-08-21) ────────────────────────────────
# top_share does NOT catch this. At α=0.40 the bacterial seed emitted
# "C.C.C.S.A.H.F.S.C.F.T.H.C.E.F.T.R.G.T.C.E.D.G.F.T.S.C" at top_share 0.135 —
# nowhere near the 0.40 abort threshold. The degeneracy is a run of single
# letters, which is lexically DIVERSE, so the proxy reads it as healthy prose.
# Count it directly instead.
import re as _re
def _salad(t):
    return len(_re.findall(r"(?:\b[A-Z]\.){4,}", t)) + \
           len(_re.findall(r"(?:\b[A-Z]\b[ .]){5,}", t))
_hits=sum(_salad(x) for kk,v in new["seeds"].items()
          if not any(z in kk for z in ("fibonacci","quadratic"))
          for x in v["0.0"]["texts"])
print(f"  acronym-salad hits across prose samples: {_hits}   (α=0.45 @143,500 = 0, α=0.40 @149,500 = 4)")
print()
if _hits > 0:
    print(f"  ⚠ ACRONYM SALAD RETURNED ({_hits} hits) — this is re-collapse even if")
    print("    top_share looks fine. Read the text before continuing.")
if worse: print(f"  ⚠ RE-COLLAPSE on {len(worse)} seed(s) >0.40 — the abort signal. Fall back to step_0140000.pt.")
elif mb < ma - 0.02: print("  ✅ WORKING — α=0.0 improved with no re-collapse. Same shape as 0.6→0.5. Anneal further.")
else: print("  ⏸ FLAT — α is not the lever at this scale. The exposure-bias gap needs rung 6 (on-policy SFT), not a smaller α.")
PY
echo
echo "READ THE TEXT TOO — the .txt has α=0.0 samples. Metrics have pointed the wrong way before."
