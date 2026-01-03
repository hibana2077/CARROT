#!/bin/bash
set -euo pipefail

s=0
e=29

for i in $(seq $s $e); do
  qsub BT$(printf "%03d" $i).sh
done