import json, sys, os, collections
import numpy as np
def load(p):
    d = collections.defaultdict(list)
    for l in open(p, encoding="utf-8"):
        r = json.loads(l); d[r["problem"] if os.environ.get("GROUP") == "problem" else r["idx"]].append(r)
    return d
a, b = load(sys.argv[1]), load(sys.argv[2])
ids = sorted(set(a) & set(b))
rng = np.random.default_rng(0)
boot = rng.integers(0, len(ids), (5000, len(ids)))
print(f"problems: {len(ids)}, samples per problem: {len(a[ids[0]])} vs {len(b[ids[0]])}")
for name, f in [("accuracy %", lambda r: 100 * r["acc"]), ("avg length", lambda r: r["length"]),
                ("NoThinking %", lambda r: 100 * r["is_nt"]), ("implicit responses % (of all)", lambda r: 100 * r["implicit"]),
                ("cut at max %", lambda r: 100 * (r["finish"] == "length"))]:
    x = np.array([np.mean([f(r) for r in a[i]]) for i in ids]); y = np.array([np.mean([f(r) for r in b[i]]) for i in ids])
    d = y - x; lo, hi = np.percentile(d[boot].mean(1), [2.5, 97.5])
    print(f"{name:32s} {x.mean():8.2f} -> {y.mean():8.2f}   diff {d.mean():+7.2f}  95% CI [{lo:+.2f}, {hi:+.2f}]")
