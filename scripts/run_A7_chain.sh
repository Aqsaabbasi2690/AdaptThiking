#!/bin/bash
cd ~/AdaptThink
T=outputs/ablation.times
n=A7_modeshort
echo "$n start $(date +%T)" >> $T
python analysis/sft_lora.py --data outputs/natural/sft_A7.jsonl --lr 3e-5 --out outputs/sft/$n > outputs/sft/$n.log 2>&1 || { echo "$n TRAIN FAILED" >> $T; exit 1; }
echo "$n train done $(date +%T)" >> $T
python analysis/gen_natural.py --model outputs/sft/$n --dataset val --n 4 --max_tokens 8192 --out outputs/natural/val_$n.jsonl > outputs/natural/val_$n.log 2>&1 || { echo "$n EVAL FAILED" >> $T; exit 1; }
echo "$n val done $(date +%T)" >> $T
bash scripts/run_test_eval.sh outputs/sft/$n A7
echo "A7 aime start $(date +%T)" >> outputs/test_eval.times
python analysis/gen_natural.py --model outputs/sft/$n --dataset aime --n 1 --out outputs/natural/test_A7_aime.jsonl > outputs/natural/test_A7_aime.log 2>&1 && echo "A7 aime done $(date +%T)" >> outputs/test_eval.times || echo "A7 aime FAILED" >> outputs/test_eval.times
echo "A7 ALL DONE $(date +%T)" >> $T
