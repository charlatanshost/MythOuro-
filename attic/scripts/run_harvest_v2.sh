#!/usr/bin/env bash
# Teacher-corpus harvest, validated fast config (b30 + prealloc + on-device
# sampling, ~101 accepted tok/s). Resumes shard numbering automatically.
#
# CORPUS_TARGET is the TOTAL corpus size we want on disk, not this session's
# quota. `--target-tokens` is per-session (accepted_tok resets to 0 each launch),
# so a bare relaunch after an interruption harvests the FULL number again on top
# of what already exists — the 2026-07-23 power-outage restart would have
# overshot 12M by 4.3M. We read the manifest's cumulative total and ask only for
# the remainder, which makes this script safe to re-run any number of times.
set -euo pipefail
cd "$(dirname "$0")"

CORPUS_TARGET=${CORPUS_TARGET:-12000000}
OUT_DIR=${OUT_DIR:-data_teacher_v2}
# DEVICE selects the XPU. Single card: leave default. Two-tile 1550: launch two
# copies, DEVICE=xpu:0 and DEVICE=xpu:1, each with its OWN OUT_DIR (concurrent
# writers on one dir collide on shard numbering — see the two-tile runbook in
# docs/hardware_options.md), then merge.
DEVICE=${DEVICE:-xpu:0}

read -r HAVE REMAIN < <(python3 - "$OUT_DIR" "$CORPUS_TARGET" <<'PY'
import json, sys
from pathlib import Path
out, target = Path(sys.argv[1]), int(sys.argv[2])
m = {}
p = out / "MANIFEST.json"
if p.exists():
    try:
        m = json.loads(p.read_text())
    except Exception:
        m = {}
# sessions-aware total; falls back to the flat pre-2026-07-23 key.
have = m.get("total_accepted_tokens")
if have is None:
    have = sum(s.get("accepted_tokens", 0) for s in m.get("sessions", [])) \
        or m.get("accepted_tokens", 0)
print(have, max(0, target - have))
PY
)
echo "corpus target ${CORPUS_TARGET} | on disk ${HAVE} | harvesting ${REMAIN}"
if [ "$REMAIN" -le 0 ]; then
  echo "target already met — nothing to do."; exit 0
fi

source ../venv-xpu/bin/activate
export SYCL_CACHE_PERSISTENT=1 PYTORCH_ALLOC_CONF=expandable_segments:True TRITON_DEFAULT_BACKEND=intel

# --seed-mix: drawing the target ratios does NOT yield those ratios in the
# ACCEPTED corpus — code accepts at ~half general's rate and its samples are
# shorter, so a code seed returns 396 usable tokens vs a general seed's 536
# (measured 2026-07-23). These weights over-draw code+math to compensate.
# TARGET CHANGED 2026-07-27 to a UNIFORM accepted corpus (was 40/40/20), matching
# the uniform _MIX_RATIOS — code was the dose-limited laggard (@52k probe), so
# more code, in both the real mix AND the teacher corpus. Recompute if the
# target / filters / measured acceptance rates change.
exec python -u -m tools.gen_teacher_corpus --device "$DEVICE" --trust-remote-code \
  --out-dir "$OUT_DIR" --target-tokens "$REMAIN" --batch 30 --prealloc-cache \
  --seed-mix 'general=0.2823,math=0.3361,code=0.3816' --telemetry
