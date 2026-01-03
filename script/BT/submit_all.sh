#!/bin/bash
set -euo pipefail

s 0
e 29

for f in script/BT/BT$(printf "%03d" $(seq $s $e)); do
  echo "qsub $f"
  qsub "$f"
done
