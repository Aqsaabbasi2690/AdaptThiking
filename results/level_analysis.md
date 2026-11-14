## MATH500 by difficulty level (AdaptThink-1.5B d0.05 vs original, ours vs paper)

Unmatched rows (should be 0): 0; difficulty values seen: [1.0, 2.0, 3.0, 4.0, 5.0]

| Level | n | NoThinking % (ours) | NoThinking % (paper Fig.5) | AdaptThink acc | Original acc | AdaptThink len | Original len |
|---|---|---|---|---|---|---|---|
| 1.0 | 43 | 90.7 | 97.7 | 93.0 | 88.4 | 426 | 2476 |
| 2.0 | 90 | 87.8 | 86.7 | 93.3 | 91.1 | 724 | 2742 |
| 3.0 | 105 | 80.0 | 77.1 | 93.3 | 93.3 | 1254 | 3570 |
| 4.0 | 128 | 78.9 | 78.1 | 82.8 | 78.9 | 1720 | 4591 |
| 5.0 | 134 | 61.2 | 61.9 | 65.7 | 64.2 | 3451 | 7319 |

## Implicit thinking in AdaptThink NoThinking responses (keywords: Wait / Alternatively)

| Dataset | NoThinking responses | with keyword % | avg length (tokens) |
|---|---|---|---|
| GSM8K | 1155 | 2.9 | 436 |
| MATH500 | 385 | 7.3 | 953 |
| AIME24 | 217 | 53.9 | 4031 |
| pooled | 1757 | 10.2 | 993 |

Paper Table 3 (1.5B, AdaptThink NoThinking cases): 7.9% implicit thinking, avg length 826.
(Paper pools its own test cases, so compare with the pooled row only roughly.)
