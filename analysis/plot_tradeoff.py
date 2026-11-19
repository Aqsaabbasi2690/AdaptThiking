import json
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
R = Path(__file__).resolve().parent.parent; N = R / "outputs/natural"
DELTA = {  # our verl re-evaluation of the released checkpoints (results/delta_sweep.md): label -> (avg len, acc)
 "gsm8k": {"original": (1022, 78.5), "d0": (723, 83.5), "d0.01": (673, 81.8), "d0.02": (630, 83.1), "d0.05": (552, 83.2), "d0.075": (568, 83.9), "d0.1": (386, 82.9)},
 "math":  {"original": (4593, 81.0), "d0": (2489, 83.8), "d0.01": (2631, 83.4), "d0.02": (2215, 82.8), "d0.05": (1796, 83.2), "d0.075": (1670, 81.2), "d0.1": (1220, 80.6)}}
FILES = {"AdaptThink d0.05 (base, standalone n=4)": "base_{}.jsonl", "LoRA main": "test_main_{}.jsonl", "LoRA A3": "test_A3_{}.jsonl", "LoRA A7": "test_A7_{}.jsonl"}
def point(f):
    p = N / f
    if not p.exists(): return None
    rows = [json.loads(l) for l in open(p, encoding="utf-8")]
    return sum(r["length"] for r in rows) / len(rows), 100 * sum(r["acc"] for r in rows) / len(rows)
fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
for a, (ds, title) in zip(ax, [("gsm8k", "GSM8K"), ("math", "MATH500")]):
    d = DELTA[ds]
    a.plot([v[0] for v in d.values()], [v[1] for v in d.values()], "s--", color="gray", label="released checkpoints (verl eval, 1 sample)")
    for k, v in d.items(): a.annotate(k, v, fontsize=6, xytext=(3, 3), textcoords="offset points")
    for (name, f), c in zip(FILES.items(), ["C0", "C3", "C1", "C2"]):
        pt = point(f.format(ds))
        if pt: a.scatter(*pt, color=c, s=60, label=name, zorder=3)
    a.set_title(title); a.set_xlabel("avg response length (tokens)"); a.set_ylabel("accuracy %"); a.legend(fontsize=6)
fig.tight_layout(); fig.savefig(R / "results/tradeoff_lora_vs_delta.png", dpi=140); print("saved results/tradeoff_lora_vs_delta.png")
