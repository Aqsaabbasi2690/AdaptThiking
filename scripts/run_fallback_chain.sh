#!/bin/bash
# Modification 4 (online implicit-thinking fallback) + MMLU evaluation.
# usage: bash scripts/run_fallback_chain.sh            (all steps, skips finished ones)
#        STEPS="A3fb_math" bash scripts/run_fallback_chain.sh   (one step)
cd ~/AdaptThink
BASE=$HOME/models/AdaptThink-1.5B-delta0.05
ORIG=$HOME/models/DeepSeek-R1-Distill-Qwen-1.5B
A3=outputs/sft/A3_lr3e5
T=outputs/fallback.times
O=outputs/natural
MMLU_LIMIT=${MMLU_LIMIT:-2000}   # random subset (seed 0); set MMLU_LIMIT=0 for all 14042 questions

[ -f data/test/preprocessed_data/mmlu.parquet ] || \
  python src/preprocess_dataset.py --input_path ./data/test/mmlu.json --output_path ./data/test/preprocessed_data/mmlu.parquet

go() {  # name  args...
  name=$1; shift
  if [ -n "$STEPS" ] && [[ " $STEPS " != *" $name "* ]]; then return; fi
  out=$O/$name.jsonl
  [ -f $out ] && { echo "skip $name" >> $T; return; }
  echo "$name start $(date +%T)" >> $T
  python analysis/fallback_online.py "$@" --out $out > $O/$name.log 2>&1 \
    && echo "$name done $(date +%T)" >> $T || echo "$name FAILED (see $O/$name.log)" >> $T
}

# 1) Main experiment: LoRA A3 + online fallback (same n as the existing test runs)
go test_A3fb_gsm8k  --model $A3 --dataset gsm8k --n 4
go test_A3fb_math   --model $A3 --dataset math  --n 4
go test_A3fb_aime   --model $A3 --dataset aime  --n 1
# 2) Control: original AdaptThink d0.05 + online fallback (separates the effect of LoRA from the fallback)
go test_basefb_gsm8k --model $BASE --dataset gsm8k --n 4
go test_basefb_math  --model $BASE --dataset math  --n 4
go test_basefb_aime  --model $BASE --dataset aime  --n 1
# 3) MMLU (out-of-domain, paper Table 4), one sample per question
go mmlu_base  --model $BASE --dataset mmlu --n 1 --limit $MMLU_LIMIT --no_fallback
go mmlu_orig  --model $ORIG --dataset mmlu --n 1 --limit $MMLU_LIMIT --no_fallback
go mmlu_A3    --model $A3   --dataset mmlu --n 1 --limit $MMLU_LIMIT --no_fallback
go mmlu_A3fb  --model $A3   --dataset mmlu --n 1 --limit $MMLU_LIMIT

# 4) Optional variant (only if 1) helps but AIME is still low): also abort NoThinking attempts longer than 4096 tokens
#    STEPS="test_A3fbcap_gsm8k test_A3fbcap_math test_A3fbcap_aime" bash scripts/run_fallback_chain.sh
if [ -n "$STEPS" ]; then
go test_A3fbcap_gsm8k --model $A3 --dataset gsm8k --n 4 --nt_max_tokens 4096
go test_A3fbcap_math  --model $A3 --dataset math  --n 4 --nt_max_tokens 4096
go test_A3fbcap_aime  --model $A3 --dataset aime  --n 1 --nt_max_tokens 4096
fi

python analysis/fallback_table.py >> $T 2>&1
echo "FALLBACK CHAIN DONE $(date +%T)" >> $T
