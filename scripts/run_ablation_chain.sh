#!/bin/bash
# Pre-registered ablation grid (one change at a time vs the main config). Skips finished runs.
cd ~/AdaptThink
T=outputs/ablation.times
run() {  # name  select-args  sft-args  [data]
  name=$1; sel=$2; sft=$3
  d=outputs/natural/sft_$name.jsonl
  if [ -f outputs/natural/val_$name.jsonl ]; then echo "skip $name"; return; fi
  echo "$name start $(date +%T)" >> $T
  if [ -n "$sel" ]; then python analysis/select_sft_data.py $sel --out $d > outputs/natural/sel_$name.log 2>&1; else d=outputs/natural/sft_data.jsonl; fi
  python analysis/sft_lora.py --data $d --out outputs/sft/$name $sft > outputs/sft/$name.log 2>&1 || { echo "$name TRAIN FAILED" >> $T; return; }
  echo "$name train done $(date +%T)" >> $T
  python analysis/gen_natural.py --model outputs/sft/$name --dataset val --n 4 --max_tokens 8192 --out outputs/natural/val_$name.jsonl > outputs/natural/val_$name.log 2>&1 || { echo "$name EVAL FAILED" >> $T; return; }
  echo "$name val done $(date +%T)" >> $T
}
run A1_random   "--select random"        ""
run A2_noimpl   "--no_implicit_filter"   ""
run A3_lr3e5    ""                       "--lr 3e-5"
run A4_ep1      ""                       "--epochs 1"
echo "ALL DONE $(date +%T)" >> $T
