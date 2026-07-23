#!/usr/bin/env bash
# Stop harvest/training cleanly + zombie check (field-notes gotcha #7).
pkill -f "tools.gen_teacher_corpus" 2>/dev/null
pkill -f "training.distill" 2>/dev/null
sleep 5
if pgrep -f "tools.gen_teacher_corpus|training.distill" | grep -qv $$; then
  echo "still shutting down; re-run this script if memory below isn't ~0"
fi
xpu-smi dump -d 0 -m 18 -n 1 | tail -1
echo "^ memory should read ~22 MiB. If large: pgrep -af python; kill -9 <pid>"
