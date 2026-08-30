#!/usr/bin/env bash
# Guarded apply/revert for the throughput optimisations. See README.md.
#
#   bash optim/apply.sh A        # teacher-logit reuse cache (equivalent)
#   bash optim/apply.sh B        # + precomputed offline logits (approximate)
#   bash optim/apply.sh revert   # restore the .preA/.preB backups
#
# REFUSES while a trainer is running. distill.py is already compiled into a live
# process so an edit would not change it mid-run, but the next resume would pick
# up a half-applied tree — and an 8.5h leg is not worth that risk.
set -uo pipefail
cd "$(dirname "$0")/.."
if pgrep -f "python -u -m training\.(distill|sft)" >/dev/null; then
  echo "a trainer is running — refusing to patch. Stop it first."; exit 1
fi
case "${1:-}" in
  A) python3 optim/patchA_apply.py . && python3 -m pytest optim/test_optA.py \
       optim/test_optA_regression.py -q ;;
  B) grep -q "OPT-A" training/distill.py || { echo "apply A first"; exit 1; }
     python3 optim/patchB_apply.py . && python3 -m pytest optim/test_optB.py \
       optim/test_optB_loader.py -q ;;
  revert)
     for f in training/distill.py mythouro/rollout.py; do
       for s in preB preA; do
         [ -f "$f.$s" ] && { cp "$f.$s" "$f"; echo "  restored $f from .$s"; break; }
       done
     done ;;
  *) sed -n '2,10p' "$0"; exit 1 ;;
esac
