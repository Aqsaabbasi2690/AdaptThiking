| Dataset | Model | Accuracy % (diff [95% CI]) | Avg length incl. aborted tokens (diff, % [95% CI]) | NoThinking % (diff) | Fallback % | Cut at 16K % (diff) |
|---|---|---|---|---|---|---|
| MMLU | DeepSeek-R1-Distill-Qwen-1.5B (original) | 35.9 (-4.30 [-6.70, -2.00]) | 1589 (+596, +60.1% [+438, +760]) | 0.0 (-14.7) | 0.0 | 3.70 (+1.75) |
| MMLU (2000 questions) | AdaptThink d0.05 (base) | 40.1 | 992 | 14.7 | 0.0 | 1.9 |
| MMLU | LoRA A3 | 40.2 (+0.05 [-2.25, +2.35]) | 1534 (+542, +54.6% [+364, +723]) | 15.1 (+0.5) | 0.0 | 5.70 (+3.75) |
| MMLU | LoRA A3 + online fallback | 42.0 (+1.90 [-0.35, +4.10]) | 1538 (+546, +55.0% [+374, +718]) | 15.1 (+0.4) | 0.1 | 5.75 (+3.80) |
