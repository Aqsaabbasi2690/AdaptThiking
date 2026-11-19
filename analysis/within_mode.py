import json, collections
import numpy as np
from pathlib import Path
R = Path(__file__).resolve().parent.parent; N = R / "outputs/natural"
SETS = [("GSM8K", "base_gsm8k.jsonl", "test_{}_gsm8k.jsonl", "idx"), ("MATH500", "base_math.jsonl", "test_{}_math.jsonl", "idx"),
        ("AIME24", "test_base_aime.jsonl", "test_{}_aime.jsonl", "problem")]
def load(p, key):
    d = collections.defaultdict(list)
    for l in open(p, encoding="utf-8"):
        r = json.loads(l); d[r[key]].append(r)
    return d
out = ["| Dataset | Model | Mode | problems with both | mean length change on the same problems [95% CI] |", "|---|---|---|---|---|"]
for ds, bf, tf, key in SETS:
    if not (N / bf).exists(): continue
    base = load(N / bf, key)
    for run in ["main", "A3", "A7"]:
        f = N / tf.format(run)
        if not f.exists(): continue
        t = load(f, key)
        for mode, name in [(True, "NoThinking"), (False, "Thinking")]:
            d = []
            for k in set(base) & set(t):
                b = [r["length"] for r in base[k] if r["is_nt"] == mode]; c = [r["length"] for r in t[k] if r["is_nt"] == mode]
                if b and c: d.append(np.mean(c) - np.mean(b))
            d = np.array(d)
            if len(d) == 0: continue
            lo, hi = np.percentile(d[np.random.default_rng(0).integers(0, len(d), (5000, len(d)))].mean(1), [2.5, 97.5])
            out.append(f"| {ds} | {run} | {name} | {len(d)} | {d.mean():+.0f} [{lo:+.0f}, {hi:+.0f}] |")
text = "\n".join(out) + "\n\nOnly problems with at least one response in this mode both before and after are used, so the problem set is the same on both sides. This removes the composition effect of the decomposition table, but problems that switch mode are excluded."
print(text); (R / "results/within_mode.md").write_text(text + "\n", encoding="utf-8")
