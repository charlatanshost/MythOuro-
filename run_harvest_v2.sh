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

# --seed-mix: drawing 40/40/20 does NOT yield a 40/40/20 ACCEPTED corpus.
# Measured 2026-07-23 on the first 4.28M of v2: acceptance is 82.5/58.8/51.6%
# and mean length 650/766/768, so a code seed returns 396 usable tokens vs a
# general seed's 536 — the corpus landed at 45.3/38.0/16.7 by token. These
# weights over-draw code+math so the FULL 12M corpus lands on 40/40/20,
# compensating for the drift already banked. Recompute them if the target,
# the filters, or the measured acceptance rates change.
exec python -u -m tools.gen_teacher_corpus --device xpu:0 --trust-remote-code \
  --out-dir "$OUT_DIR" --target-tokens "$REMAIN" --batch 30 --prealloc-cache \
  --seed-mix 'general=0.3211,math=0.4237,code=0.2552' --telemetry
