#!/bin/bash
set -euo pipefail

s=0
e=29

for i in $(seq $s $e); do
  qsub BTC$(printf "%03d" $i).sh
done