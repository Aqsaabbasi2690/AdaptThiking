import json, re
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
M = "AdaptThink-1.5B-delta0.05"
KW = re.compile(r"\b(Wait|Alternatively)\b")
def load(key):
    d = ROOT / "outputs" / M / f"eval_results_{key}_len16384_n1" / "data"
    return json.load(open(sorted(d.glob("*.json"))[0], encoding="utf-8"))
def mean(x): return sum(x) / len(x) if x else float("nan")
print("| Dataset | NoThinking subset | n | acc % | avg len |\n|---|---|---|---|---|")
for name, key in [("GSM8K", "gsm8k"), ("MATH500", "math"), ("AIME24", "aime_16")]:
    rows = load(key)
    nt = [r for r in rows if r["is_nothinking"]]
    has = lambda r: bool(KW.search(r["model_response"].split("</think>", 1)[-1]))
    for label, sub in [("with keyword", [r for r in nt if has(r)]),
                       ("no keyword", [r for r in nt if not has(r)]),
                       ("(ref) Thinking mode", [r for r in rows if not r["is_nothinking"]])]:
        print(f"| {name} | {label} | {len(sub)} | {100*mean([r['acc'] for r in sub]):.1f} | {mean([r['response_length'] for r in sub]):.0f} |")
