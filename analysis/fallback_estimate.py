import json, re, collections
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
O, A = "DeepSeek-R1-Distill-Qwen-1.5B", "AdaptThink-1.5B-delta0.05"
KW = re.compile(r"\b(Wait|Alternatively)\b")
def load(model, key):
    d = ROOT / "outputs" / model / f"eval_results_{key}_len16384_n1" / "data"
    return json.load(open(sorted(d.glob("*.json"))[0], encoding="utf-8"))
def norm(s):
    return re.sub(r"\s+", " ", re.sub(r"<｜User｜>|<｜Assistant｜>|<think>", " ", s)).strip()
def mean(x): return sum(x) / len(x) if x else float("nan")
def flagged(r):
    return r["is_nothinking"] and bool(KW.search(r["model_response"].split("</think>", 1)[-1]))

print("| Dataset | Policy | acc % | avg len | rows replaced |\n|---|---|---|---|---|")
for name, key in [("GSM8K", "gsm8k"), ("MATH500", "math"), ("AIME24", "aime_16")]:
    orig, ada = load(O, key), load(A, key)
    by = collections.defaultdict(list)
    for r in orig:
        by[norm(r["problem"])].append(r)
    acc_opt, len_opt, len_cons, replaced, miss = [], [], [], 0, 0
    for r in ada:
        o = by.get(norm(r["problem"]))
        if flagged(r) and o:
            replaced += 1
            oa, ol = mean([x["acc"] for x in o]), mean([x["response_length"] for x in o])
            acc_opt.append(oa); len_opt.append(ol); len_cons.append(ol + r["response_length"])
        else:
            if flagged(r): miss += 1
            acc_opt.append(r["acc"]); len_opt.append(r["response_length"]); len_cons.append(r["response_length"])
    rows = [
        ("Original (Thinking)", mean([r["acc"] for r in orig]), mean([r["response_length"] for r in orig]), 0),
        ("AdaptThink", mean([r["acc"] for r in ada]), mean([r["response_length"] for r in ada]), 0),
        ("AdaptThink + fallback (optimistic cost)", mean(acc_opt), mean(len_opt), replaced),
        ("AdaptThink + fallback (conservative cost)", mean(acc_opt), mean(len_cons), replaced),
    ]
    for label, a, l, n in rows:
        print(f"| {name} | {label} | {100*a:.1f} | {l:.0f} | {n} |")
    if miss: print(f"<!-- {name}: {miss} flagged rows had no matching original problem -->")
