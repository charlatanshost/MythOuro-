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
from collections import Counter as _C
def _looping(t):
    """A sample loops if any non-trivial line repeats >=3x.

    Added 2026-09-03 after the owner asked whether the TEXTS had been read.
    They had not, and reading them found a "Question 15" x7 loop that
    distinct1 had averaged into invisibility across 6 seeds. Counting turns
    out to separate far better than distinct1 does: the regressed mathcode
    checkpoint runs 5/30 looping against 0-1/30 everywhere else, where
    distinct1 only manages 0.482 vs 0.571 on a metric with 0.016 sd.
    """
    lines=[l.strip() for l in t.split("\n") if len(l.strip())>3]
    return bool(lines) and max(_C(lines).values()) >= 3
def _stutter(t):
    """Token-level stutter — a 4+ char run doubled inside one word
    ('immunoconductduct', 'ant-antantantant')."""
    return bool(re.search(r"\b\w*?(\w{4,})\1\w*\b", t))
rows=[]
for p in sys.argv[1:]:
    d=json.load(open(p))
    ts,ds,hits,n = [],[],0,0
    loops=stut=0
    for seed,by_alpha in d["seeds"].items():
        a=by_alpha.get("0.0")
        if not a: continue
        # ⚠ top_share / distinct1 are PER-SAMPLE LISTS, not scalars. Treating
        # them as scalars makes statistics.mean() raise
        # "can't convert type 'list' to numerator/denominator" (2026-09-02) —
        # after the probe has already done its work, so it only kills the
        # summary table. Flatten, then average.
        _ts = a["top_share"]; _ds = a["distinct1"]
        ts.extend(_ts if isinstance(_ts, list) else [_ts])
        ds.extend(_ds if isinstance(_ds, list) else [_ds])
        for t in a.get("texts",[]):
            n+=1
            if SALAD.search(t): hits+=1
            if _looping(t): loops+=1
            if _stutter(t): stut+=1
    rows.append((d.get("step","?"), st.mean(ts), st.mean(ds), hits, n, loops, stut))
print("\n"+"="*64)
print("  PROSE READOUT (α=0.0) — lower top_share, higher distinct1 is better")
print("="*64)
print(f"  {'step':>10} {'top_share':>10} {'distinct1':>10} {'salad':>8} {'LOOPING':>9} {'stutter':>9}")
for s,t,dd,h,n,lo,su in rows:
    print(f"  {s:>10} {t:10.3f} {dd:10.3f} {f'{h}/{n}':>8} {f'{lo}/{n}':>9} {f'{su}/{n}':>9}")
if len(rows)>=2:
    print(f"\n  MEAN over {len(rows)} checkpoints:")
    print(f"    top_share {st.mean([r[1] for r in rows]):.3f} "
          f"(sd {st.pstdev([r[1] for r in rows]):.3f})")
    print(f"    distinct1 {st.mean([r[2] for r in rows]):.3f} "
          f"(sd {st.pstdev([r[2] for r in rows]):.3f})")
    print(f"    salad     {sum(r[3] for r in rows)}/{sum(r[4] for r in rows)}")
    print(f"    LOOPING   {sum(r[5] for r in rows)}/{sum(r[4] for r in rows)}"
          f"   <- separates better than distinct1; regressed ckpt runs 5/30")
    print(f"    stutter   {sum(r[6] for r in rows)}/{sum(r[4] for r in rows)}")
print("\n  REFERENCE — the regression that motivated growth (pre-growth 278M):")
print("    157,238  top_share 0.150  distinct1 0.484  salad 0")
print("    160,000  top_share 0.197  distinct1 0.456  salad 0")
print("    163,238  top_share 0.242  distinct1 0.427  salad 1")
print("\n  ⚠ READ THE TEXTS in the json, not just these numbers. The salad regex")
print("    is a screen, not the judgement — 2026-08-22: 'the salad is only")
print("    visible in the text'.")
PY
