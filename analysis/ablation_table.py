import json, collections
import numpy as np
from pathlib import Path
R = Path(__file__).resolve().parent.parent
RUNS = [("main (r16, lr1e-4, 2ep, shortest, filter)", "val_main.jsonl", "main_r16_lr1e-4_e2"),
        ("A1 random selection", "val_A1_random.jsonl", "A1_random"),
        ("A2 no implicit filter", "val_A2_noimpl.jsonl", "A2_noimpl"),
        ("A3 lr 3e-5", "val_A3_lr3e5.jsonl", "A3_lr3e5"),
        ("A4 1 epoch", "val_A4_ep1.jsonl", "A4_ep1"),
        ("A5 lr 1e-5 (post-hoc)", "val_A5_lr1e5.jsonl", "A5_lr1e5"),
        ("A6 lr 3e-5, seed 1 (post-hoc replicate of A3)", "val_A6_lr3e5_seed1.jsonl", "A6_lr3e5_seed1"),
        ("A7 mode-preserving selection, lr 3e-5 (post-hoc)", "val_A7_modeshort.jsonl", "A7_modeshort")]
def load(p):
    d = collections.defaultdict(list)
    for l in open(p, encoding="utf-8"):
        r = json.loads(l); d[r["idx"]].append(r)
    return d
base = load(R / "outputs/natural/val_base.jsonl")
M = [("acc %", lambda r: 100 * r["acc"]), ("len", lambda r: r["length"]),
     ("NT %", lambda r: 100 * r["is_nt"]), ("implicit %", lambda r: 100 * r["implicit"])]
def per_problem(d, ids, f): return np.array([np.mean([f(r) for r in d[i]]) for i in ids])
out = ["| run | acc % (diff [95% CI]) | avg len (diff [95% CI]) | NoThinking % (diff) | implicit % (diff) | final train loss |", "|---|---|---|---|---|---|"]
for name, vf, sd in RUNS:
    p = R / "outputs/natural" / vf
    if not p.exists():
        out.append(f"| {name} | (not finished) | | | | |"); continue
    d = load(p); ids = sorted(set(base) & set(d))
    rng = np.random.default_rng(0); boot = rng.integers(0, len(ids), (5000, len(ids)))
    cells = []
    for k, (mn, f) in enumerate(M):
        x, y = per_problem(base, ids, f), per_problem(d, ids, f); diff = y - x
        lo, hi = np.percentile(diff[boot].mean(1), [2.5, 97.5])
        cells.append(f"{y.mean():.1f} ({diff.mean():+.1f} [{lo:+.1f}, {hi:+.1f}])" if k < 2 else f"{y.mean():.1f} ({diff.mean():+.1f})")
    tl = R / "outputs/sft" / sd / "train_log.json"
    loss = np.mean([e["loss"] for e in json.load(open(tl))["log"][-10:]]) if tl.exists() else float("nan")
    out.append(f"| {name} | " + " | ".join(cells) + f" | {loss:.3f} |")
b = [per_problem(base, sorted(base), f).mean() for _, f in M]
out.append(f"\nBaseline (original delta=0.05) on the same 200 validation problems: acc {b[0]:.1f}%, len {b[1]:.0f}, NoThinking {b[2]:.1f}%, implicit {b[3]:.1f}%.")
text = "\n".join(out); print(text)
(R / "results").mkdir(exist_ok=True); (R / "results/ablation_val.md").write_text(text + "\n", encoding="utf-8")
