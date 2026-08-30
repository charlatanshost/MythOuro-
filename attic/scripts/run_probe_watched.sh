#!/usr/bin/env bash
# Run the 48-expert probe while POLLING GPU MEMORY, so the crash comes with a
# number instead of an inference.
#
# Every fault today reads "Segmentation fault from GPU ... type: 0 (NotPresent),
# level: 1 (PDE), access: 1 (Write)" — a write to an unmapped page. The minimal
# repro produced that identical signature purely by over-allocating (35.85 GB),
# which says this card PAGE-FAULTS where another would raise a clean OOM.
# If that is the story here, memory should be near the 48 GB ceiling at the
# moment of death. If it dies at 20 GB, memory is definitively not the cause and
# this line of investigation closes.
set -uo pipefail
cd "$(dirname "$0")"
mkdir -p logs
MEM="logs/probe_mem_$(date +%Y%m%d_%H%M).csv"
( while true; do
    printf "%s,%s\n" "$(date +%H:%M:%S)" \
      "$(timeout 5 xpu-smi dump -d 0 -m 18 -n 1 2>/dev/null | tail -1 | awk -F, '{print $3}')"
    sleep 2
  done ) > "$MEM" &
POLL=$!
trap 'kill $POLL 2>/dev/null' EXIT

bash run_grown48_probe.sh || true

kill $POLL 2>/dev/null
echo
echo "=== GPU memory (MiB) over the run ==="
sort -t, -k2 -rn "$MEM" | head -3 | awk -F, '{printf "  peak  %s  %s MiB (%.1f GB)\n",$1,$2,$2/1024}'
tail -3 "$MEM" | awk -F, '{printf "  last  %s  %s MiB (%.1f GB)\n",$1,$2,$2/1024}'
echo "  card total: 48 GB"
