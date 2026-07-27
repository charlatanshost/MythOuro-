#!/usr/bin/env bash
# Quick rig snapshot for remote check-ins: what's the card doing right now,
# what step, has the probe landed, how big is the corpus. Read-only.
#   bash status.sh
cd "$(dirname "$0")"
TARGET=${TARGET:-52000}
{
  echo "time:   $(date '+%a %H:%M %Z')"
  echo "phase:  $(pgrep -f training.distill >/dev/null && echo TRAINING \
    || (pgrep -f onpolicy_rollout_probe >/dev/null && echo 'PROBING (rollout)' \
    || (pgrep -f collapse_metrics >/dev/null && echo 'PROBING (sweep)' \
    || (pgrep -f gen_teacher_corpus >/dev/null && echo HARVESTING \
    || echo 'IDLE / DONE'))))"
  echo "step:   $(ls -t checkpoints_onpolicy_fixed/step_*.pt 2>/dev/null | head -1 | grep -oE '[0-9]+' | sed 's/^0*//')"
  echo "gpu:    $(xpu-smi dump -d 0 -m 18 -n 1 2>/dev/null | tail -1 | awk -F, '{print $3}') MiB"
  ls "reports/leg_${TARGET}_DONE"   >/dev/null 2>&1 && echo "probe:  ✅ DONE — reports/*_$(sed 's/step=//;s/ .*//' reports/leg_${TARGET}_DONE 2>/dev/null)_* ready"
  ls "reports/leg_${TARGET}_FAILED" >/dev/null 2>&1 && echo "probe:  ❌ FAILED — cat reports/leg_${TARGET}_FAILED"
  echo "corpus: $(python3 -c "import json;print(f\"{json.load(open('data_teacher_v2/MANIFEST.json'))['total_accepted_tokens']:,}\")" 2>/dev/null) accepted tok"
}
