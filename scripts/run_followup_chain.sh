#!/bin/bash
# Post-hoc follow-ups chosen after seeing the pre-registered grid: lower lr, and a seed replicate of A3.
cd ~/AdaptThink
T=outputs/ablation.times
run() {  # name  sft-args
  name=$1; sft=$2
  if [ -f outputs/natural/val_$name.jsonl ]; then echo "skip $name" >> $T; return; fi
  echo "$name start $(date +%T)" >> $T
  python analysis/sft_lora.py --data outputs/natural/sft_data.jsonl --out outputs/sft/$name $sft > outputs/sft/$name.log 2>&1 || { echo "$name TRAIN FAILED" >> $T; return; }
  echo "$name train done $(date +%T)" >> $T
  python analysis/gen_natural.py --model outputs/sft/$name --dataset val --n 4 --max_tokens 8192 --out outputs/natural/val_$name.jsonl > outputs/natural/val_$name.log 2>&1 || { echo "$name EVAL FAILED" >> $T; return; }
  echo "$name val done $(date +%T)" >> $T
}
run A5_lr1e5 "--lr 1e-5"
run A6_lr3e5_seed1 "--lr 3e-5 --seed 1"
echo "FOLLOWUP DONE $(date +%T)" >> $T
