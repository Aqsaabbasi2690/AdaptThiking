import json, collections
import numpy as np
from pathlib import Path
R = Path(__file__).resolve().parent.parent
N = R / "outputs/natural"
SETS = [("GSM8K", "base_gsm8k.jsonl", "test_{}_gsm8k.jsonl", "idx"),
        ("MATH500", "base_math.jsonl", "test_{}_math.jsonl", "idx"),
        ("AIME24", "test_base_aime.jsonl", "test_{}_aime.jsonl", "problem")]
RUNS = [("LoRA main (lr 1e-4)", "main"), ("LoRA A3 (lr 3e-5)", "A3"), ("LoRA A7 (mode-preserving, lr 3e-5)", "A7")]
M = {"acc": lambda r: 100 * r["acc"], "len": lambda r: r["length"],
     "nt": lambda r: 100 * r["is_nt"], "cut": lambda r: 100 * (r["finish"] == "length")}
def load(p, key):
    d = collections.defaultdict(list)
    for l in open(p, encoding="utf-8"):
        r = json.loads(l); d[r[key]].append(r)
    return d
pp = lambda d, ids, f: np.array([np.mean([f(r) for r in d[i]]) for i in ids])
out = ["| Dataset | Model | Accuracy % (diff [95% CI]) | Avg length (diff, % [95% CI]) | NoThinking % (diff) | Cut at 16K % (diff) |", "|---|---|---|---|---|---|"]
for ds, bf, tf, key in SETS:
    if not (N / bf).exists():
        out.append(f"| {ds} | (baseline file missing) | | | | |"); continue
    base = load(N / bf, key); ids = sorted(base)
    b = {k: pp(base, ids, M[k]).mean() for k in M}
    out.append(f"| {ds} ({len(ids)} problems) | AdaptThink d0.05 (base) | {b['acc']:.1f} | {b['len']:.0f} | {b['nt']:.1f} | {b['cut']:.1f} |")
    for name, tag in RUNS:
        p = N / tf.format(tag)
        if not p.exists():
            out.append(f"| {ds} | {name} | (not finished) | | | |"); continue
        d = load(p, key); common = sorted(set(base) & set(d))
        rng = np.random.default_rng(0); boot = rng.integers(0, len(common), (5000, len(common)))
        res = {}
        for k in M:
            x, y = pp(base, common, M[k]), pp(d, common, M[k]); diff = y - x
            res[k] = (y.mean(), diff.mean(), x.mean(), *np.percentile(diff[boot].mean(1), [2.5, 97.5]))
        a, l, n, c = res["acc"], res["len"], res["nt"], res["cut"]
        out.append(f"| {ds} | {name} | {a[0]:.1f} ({a[1]:+.2f} [{a[3]:+.2f}, {a[4]:+.2f}]) | {l[0]:.0f} ({l[1]:+.0f}, {100*l[1]/l[2]:+.1f}% [{l[3]:+.0f}, {l[4]:+.0f}]) | {n[0]:.1f} ({n[1]:+.1f}) | {c[0]:.2f} ({c[1]:+.2f}) |")
text = "\n".join(out); print(text)
(R / "results").mkdir(exist_ok=True); (R / "results/test_tuned.md").write_text(text + "\n", encoding="utf-8")
