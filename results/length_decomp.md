| Dataset | Model | NT % | len in NT | len in Thinking | total change | from NT-share shift | from within-mode change |
|---|---|---|---|---|---|---|---|
| GSM8K | base | 86.8 | 365 | 1486 | | | |
| GSM8K | main | 99.0 | 380 | 1926 | -116 | -188 | +72 |
| GSM8K | A3 | 96.9 | 381 | 1623 | -93 | -125 | +32 |
| GSM8K | A7 | 93.1 | 357 | 1747 | -60 | -87 | +28 |
| MATH500 | base | 77.1 | 902 | 4773 | | | |
| MATH500 | main | 89.9 | 1022 | 6758 | -185 | -731 | +547 |
| MATH500 | A3 | 91.2 | 1029 | 6332 | -291 | -745 | +454 |
| MATH500 | A7 | 82.0 | 978 | 5329 | -23 | -209 | +186 |
| AIME24 | base | 43.5 | 4033 | 9196 | | | |
| AIME24 | main | 53.5 | 4384 | 10001 | +45 | -562 | +607 |
| AIME24 | A3 | 57.5 | 4266 | 9779 | -339 | -770 | +430 |
| AIME24 | A7 | 43.5 | 4748 | 9536 | +503 | -0 | +503 |

Caveat: the NoThinking subset is not the same set of problems before and after fine-tuning (easier problems move into it), so 'within-mode' also contains a composition effect.
