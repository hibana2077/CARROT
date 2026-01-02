#!/bin/bash
set -euo pipefail

# Submit all generated sweep scripts in this directory.
# Assumes you run this on the cluster login node where `qsub` is available.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

shopt -s nullglob
scripts=(CS2000_a*_m*.sh)

if [ ${#scripts[@]} -eq 0 ]; then
  echo "No scripts found matching CS2000_a*_m*.sh in $SCRIPT_DIR" >&2
  exit 1
fi

for s in "${scripts[@]}"; do
  echo "qsub $s"
  qsub "$s"
done
