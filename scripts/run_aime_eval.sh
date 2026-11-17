#!/bin/bash
# AIME24 file already has 16 copies of each problem -> --n 1. Same pipeline for the baseline and tuned models.
cd ~/AdaptThink
T=outputs/test_eval.times
for spec in "$HOME/models/AdaptThink-1.5B-delta0.05:base" "outputs/sft/main_r16_lr1e-4_e2:main" "outputs/sft/A3_lr3e5:A3"; do
  M=${spec%%:*}; N=${spec##*:}; f=outputs/natural/test_${N}_aime.jsonl
  [ -f $f ] && { echo "skip $N aime" >> $T; continue; }
  echo "$N aime start $(date +%T)" >> $T
  python analysis/gen_natural.py --model $M --dataset aime --n 1 --out $f > outputs/natural/test_${N}_aime.log 2>&1 || { echo "$N aime FAILED" >> $T; continue; }
  echo "$N aime done $(date +%T)" >> $T
done
