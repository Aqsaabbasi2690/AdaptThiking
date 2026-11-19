import json
from pathlib import Path
R = Path(__file__).resolve().parent.parent; N = R / "outputs/natural"
SETS = [("GSM8K", "base_gsm8k.jsonl", "test_{}_gsm8k.jsonl"), ("MATH500", "base_math.jsonl", "test_{}_math.jsonl"), ("AIME24", "test_base_aime.jsonl", "test_{}_aime.jsonl")]
RUNS = ["main", "A3", "A7"]
m = lambda x: sum(x) / len(x) if x else float("nan")
def stats(p):
    rows = [json.loads(l) for l in open(p, encoding="utf-8")]
    nt = [r["length"] for r in rows if r["is_nt"]]; th = [r["length"] for r in rows if not r["is_nt"]]
    return len(nt) / len(rows), m(nt), m(th), m(nt + th)
out = ["| Dataset | Model | NT % | len in NT | len in Thinking | total change | from NT-share shift | from within-mode change |", "|---|---|---|---|---|---|---|---|"]
for ds, bf, tf in SETS:
    if not (N / bf).exists(): continue
    p, a, b, L = stats(N / bf)
    out.append(f"| {ds} | base | {100*p:.1f} | {a:.0f} | {b:.0f} | | | |")
    for r in RUNS:
        f = N / tf.format(r)
        if not f.exists(): continue
        p2, a2, b2, L2 = stats(f)
        mix = (p2 - p) * (a2 - b2); within = p * (a2 - a) + (1 - p) * (b2 - b)
        out.append(f"| {ds} | {r} | {100*p2:.1f} | {a2:.0f} | {b2:.0f} | {L2-L:+.0f} | {mix:+.0f} | {within:+.0f} |")
text = "\n".join(out) + "\n\nCaveat: the NoThinking subset is not the same set of problems before and after fine-tuning (easier problems move into it), so 'within-mode' also contains a composition effect."
print(text); (R / "results/length_decomp.md").write_text(text + "\n", encoding="utf-8")
