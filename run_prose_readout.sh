#!/usr/bin/env bash
# PROSE READOUT across >=3 checkpoints — the axis growth was actually about.
#
#   bash run_prose_readout.sh <ckpt.pt> [ckpt.pt ...]
#
# WHY THIS EXISTS. The growth programme was motivated by a PROSE regression:
# over 157,238 -> 160,000 -> 163,238 the α=0.0 metrics moved monotonically the
# wrong way (top_share 0.150 -> 0.197 -> 0.242, distinct1 0.484 -> 0.456 ->
# 0.427) and the acronym salad returned in the text. Three checkpoints, both
# metrics, plus the outputs — that evidence survives the protocol critique.
#
# ⚠ AND THEN NOTHING GROWN WAS EVER MEASURED ON IT. `onpolicy_rollout_probe`
# hardcoded `mythouro_distill_tiny` (24 experts), so every grown checkpoint died
# with a size mismatch on router.weight [48,1280] into [24,1280].
# `run_anneal_readout.sh` never passed --student-variant, and its wrapper
# swallowed the traceback. The whole week of growth work was evaluated on CODE
# only, on the axis that did NOT prompt the question. Fixed 2026-09-01: the probe
# now builds its config from the checkpoint.
#
# ⚠ ONE CHECKPOINT IS ONE DRAW. L4 measured 30, 15, 1 across checkpoints ~1,000
# steps apart (relative sd 0.77); L3+ has 13.6 pp checkpoint sd. Prose metrics
# have never had their variance characterised at all — which is part of what this
# script is for. Pass three or more checkpoints and read the MEAN.
set -uo pipefail
cd "$(dirname "$0")"
source ../venv-xpu/bin/activate
[ $# -ge 1 ] || { echo "usage: bash run_prose_readout.sh <ckpt.pt> [ckpt.pt ...]"; exit 1; }
if pgrep -f "python -u -m training\.(distill|sft)" >/dev/null; then
  echo "a trainer is running — stop it first"; exit 1; fi
mkdir -p reports logs /tmp/prose_probe

OUTS=()
for CK in "$@"; do
  [ -f "$CK" ] || { echo "missing: $CK"; exit 1; }
  TAG=$(basename "$CK" .pt | sed 's/^step_0*//')
  D=/tmp/prose_probe/$TAG; rm -rf "$D"; mkdir -p "$D"
  # the probe reads a DIRECTORY and takes the newest checkpoint in it
  cp "$CK" "$D/step_$(printf %07d "$(echo "$TAG" | grep -oE '^[0-9]+' || echo 0)").pt"
  OUT="reports/prose_${TAG}.json"
  echo "=== prose probe: $CK ==="
  python -u -m tools.onpolicy_rollout_probe --ckpt-dir "$D" \
    --student-device xpu:0 --teacher-device xpu:0 \
    --teacher-id ByteDance/Ouro-2.6B-Thinking \
    --trust-remote-code --no-kv-cache --samples 5 \
    --json "$OUT" 2>&1 | tee -a "logs/prose_${TAG}.log"
  OUTS+=("$OUT")
  rm -rf "$D"
done

python - "${OUTS[@]}" <<'PY'
import json, sys, re, statistics as st
SALAD = re.compile(r"(?:\b[A-Z]{2,6}\b[\s,\-]*){4,}")   # runs of bare acronyms
rows=[]
for p in sys.argv[1:]:
    d=json.load(open(p))
    ts,ds,hits,n = [],[],0,0
    for seed,by_alpha in d["seeds"].items():
        a=by_alpha.get("0.0")
        if not a: continue
        ts.append(a["top_share"]); ds.append(a["distinct1"])
        for t in a.get("texts",[]):
            n+=1
            if SALAD.search(t): hits+=1
    rows.append((d.get("step","?"), st.mean(ts), st.mean(ds), hits, n))
print("\n"+"="*64)
print("  PROSE READOUT (α=0.0) — lower top_share, higher distinct1 is better")
print("="*64)
print(f"  {'step':>10} {'top_share':>10} {'distinct1':>10} {'salad':>10}")
for s,t,dd,h,n in rows:
    print(f"  {s:>10} {t:10.3f} {dd:10.3f} {f'{h}/{n}':>10}")
if len(rows)>=2:
    print(f"\n  MEAN over {len(rows)} checkpoints:")
    print(f"    top_share {st.mean([r[1] for r in rows]):.3f} "
          f"(sd {st.pstdev([r[1] for r in rows]):.3f})")
    print(f"    distinct1 {st.mean([r[2] for r in rows]):.3f} "
          f"(sd {st.pstdev([r[2] for r in rows]):.3f})")
    print(f"    salad     {sum(r[3] for r in rows)}/{sum(r[4] for r in rows)}")
print("\n  REFERENCE — the regression that motivated growth (pre-growth 278M):")
print("    157,238  top_share 0.150  distinct1 0.484  salad 0")
print("    160,000  top_share 0.197  distinct1 0.456  salad 0")
print("    163,238  top_share 0.242  distinct1 0.427  salad 1")
print("\n  ⚠ READ THE TEXTS in the json, not just these numbers. The salad regex")
print("    is a screen, not the judgement — 2026-08-22: 'the salad is only")
print("    visible in the text'.")
PY
