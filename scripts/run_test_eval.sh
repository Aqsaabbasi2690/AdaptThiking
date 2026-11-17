#!/bin/bash
# usage: bash scripts/run_test_eval.sh <model_dir> <name>
cd ~/AdaptThink
M=$1; N=$2; T=outputs/test_eval.times
for ds in gsm8k math; do
  f=outputs/natural/test_${N}_$ds.jsonl
  [ -f $f ] && { echo "skip $N $ds" >> $T; continue; }
  echo "$N $ds start $(date +%T)" >> $T
  python analysis/gen_natural.py --model $M --dataset $ds --n 4 --out $f > outputs/natural/test_${N}_$ds.log 2>&1 || { echo "$N $ds FAILED" >> $T; continue; }
  echo "$N $ds done $(date +%T)" >> $T
done
