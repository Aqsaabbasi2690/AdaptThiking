import json
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
DELTAS = ["0", "0.01", "0.02", "0.05", "0.075", "0.1"]
# Paper Table 2 (AdaptThink-1.5B): (acc, length, NoThinking %)
PAPER = {
    "gsm8k": {"0": (84.6, 718, 70.4), "0.01": (82.4, 638, 55.6), "0.02": (83.1, 628, 75.7),
              "0.05": (83.1, 480, 86.9), "0.075": (83.2, 580, 71.8), "0.1": (82.5, 358, 91.7)},
    "math":  {"0": (86.0, 2511, 50.0), "0.01": (82.4, 2473, 67.0), "0.02": (83.4, 2337, 62.6),
              "0.05": (82.0, 1782, 76.8), "0.075": (80.2, 1621, 84.2), "0.1": (78.2, 1272, 90.4)},
}
PAPER_ORIG = {"gsm8k": (79.0, 978), "math": (80.6, 4887)}
NAMES = {"gsm8k": "GSM8K", "math": "MATH500"}

def get(model, ds):
    p = ROOT / "outputs" / model / f"eval_results_{ds}_len16384_n1" / "overall_metrics.json"
    if not p.exists():
        return None
    m = json.load(open(p))
    g = lambda k: m.get(f"val-aux/{ds}/{k}/mean@1")
    nt = g("is_nothinking")
    return 100 * g("acc"), g("response_length"), 100 * nt

out, ours = [], {}
for ds in ["gsm8k", "math"]:
    o = get("DeepSeek-R1-Distill-Qwen-1.5B", ds)
    out.append(f"### {NAMES[ds]}\n")
    out.append("| delta | Acc (ours) | Acc (paper) | Length (ours) | Length (paper) | NoThinking % (ours) | NoThinking % (paper) |")
    out.append("|---|---|---|---|---|---|---|")
    if o:
        out.append(f"| original | {o[0]:.1f} | {PAPER_ORIG[ds][0]} | {o[1]:.0f} | {PAPER_ORIG[ds][1]} | 0.0 | 0.0 |")
    for d in DELTAS:
        r = get(f"AdaptThink-1.5B-delta{d}", ds)
        ours[(ds, d)] = r
        p = PAPER[ds][d]
        if r is None:
            out.append(f"| {d} | missing | {p[0]} | missing | {p[1]} | missing | {p[2]} |")
        else:
            out.append(f"| {d} | {r[0]:.1f} | {p[0]} | {r[1]:.0f} | {p[1]} | {r[2]:.1f} | {p[2]} |")
    out.append("")
text = "\n".join(out)
print(text)
(ROOT / "results").mkdir(exist_ok=True)
(ROOT / "results" / "delta_sweep.md").write_text(text + "\n", encoding="utf-8")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    xs = [float(d) for d in DELTAS]
    fig, ax = plt.subplots(2, 3, figsize=(13, 7))
    for i, ds in enumerate(["gsm8k", "math"]):
        for j, (idx, title) in enumerate([(2, "NoThinking %"), (0, "Accuracy %"), (1, "Avg length (tokens)")]):
            a = ax[i][j]
            a.plot(xs, [ours[(ds, d)][idx] if ours[(ds, d)] else float("nan") for d in DELTAS], "o-", label="ours")
            a.plot(xs, [PAPER[ds][d][idx] for d in DELTAS], "s--", label="paper")
            a.set_title(f"{NAMES[ds]}: {title}"); a.set_xlabel("delta"); a.legend()
    fig.tight_layout(); fig.savefig(ROOT / "results" / "delta_sweep.png", dpi=140)
    print("Saved results/delta_sweep.png")
except Exception as e:
    print("(plot skipped:", e, ")")
