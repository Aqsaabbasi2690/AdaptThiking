#!/usr/bin/env python
"""Level-wise MATH500 analysis + implicit-thinking ratio from saved eval predictions."""
import json, re, collections
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = {"orig": "DeepSeek-R1-Distill-Qwen-1.5B", "adapt": "AdaptThink-1.5B-delta0.05"}
PAPER_NT = {1: 97.7, 2: 86.7, 3: 77.1, 4: 78.1, 5: 61.9}  # paper Fig. 5 (1.5B): % NoThinking per MATH level
KW = re.compile(r"\b(Wait|Alternatively)\b")


def norm(s):
    s = re.sub(r"<｜User｜>|<｜Assistant｜>|<think>", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def mean(x):
    return sum(x) / len(x) if x else float("nan")


def lvnum(lv):
    m = re.search(r"\d+", str(lv))
    return int(m.group()) if m else None


def load_rows(model, key):
    d = ROOT / "outputs" / model / f"eval_results_{key}_len16384_n1" / "data"
    f = sorted(d.glob("*.json"))[0]
    return json.load(open(f, encoding="utf-8"))


ds = json.load(open(ROOT / "data/test/math.json", encoding="utf-8"))
items = [(norm(d["problem"]), d["difficulty"]) for d in ds]
exact = dict(items)


def level_of(problem):
    n = norm(problem)
    if n in exact:
        return exact[n]
    for p, lv in items:
        if p in n:
            return lv
    return None


out = []

# ---- 1) level-wise analysis on MATH500 ----
levels = collections.defaultdict(lambda: {"orig": [], "adapt": []})
unmatched = 0
for tag in MODELS:
    for r in load_rows(MODELS[tag], "math"):
        lv = level_of(r["problem"])
        if lv is None:
            unmatched += 1
            continue
        levels[lv][tag].append(r)

order = sorted(levels, key=lambda l: (lvnum(l) is None, lvnum(l) or 0, str(l)))
out.append("## MATH500 by difficulty level (AdaptThink-1.5B d0.05 vs original, ours vs paper)\n")
out.append(f"Unmatched rows (should be 0): {unmatched}; difficulty values seen: {list(order)}\n")
out.append("| Level | n | NoThinking % (ours) | NoThinking % (paper Fig.5) | AdaptThink acc | Original acc | AdaptThink len | Original len |")
out.append("|---|---|---|---|---|---|---|---|")
for lv in order:
    a, o = levels[lv]["adapt"], levels[lv]["orig"]
    nt = 100 * mean([1.0 if x["is_nothinking"] else 0.0 for x in a])
    pap = PAPER_NT.get(lvnum(lv), "n/a")
    out.append(
        f"| {lv} | {len(a)} | {nt:.1f} | {pap} | {100*mean([x['acc'] for x in a]):.1f} | "
        f"{100*mean([x['acc'] for x in o]):.1f} | {mean([x['response_length'] for x in a]):.0f} | "
        f"{mean([x['response_length'] for x in o]):.0f} |"
    )

# ---- 2) implicit thinking ratio inside NoThinking responses (AdaptThink) ----
out.append("\n## Implicit thinking in AdaptThink NoThinking responses (keywords: Wait / Alternatively)\n")
out.append("| Dataset | NoThinking responses | with keyword % | avg length (tokens) |")
out.append("|---|---|---|---|")
pooled_hits, pooled_len = [], []
for name, key in [("GSM8K", "gsm8k"), ("MATH500", "math"), ("AIME24", "aime_16")]:
    nt = [r for r in load_rows(MODELS["adapt"], key) if r["is_nothinking"]]
    hits = [bool(KW.search(r["model_response"].split("</think>", 1)[-1])) for r in nt]
    lens = [r["response_length"] for r in nt]
    pooled_hits += hits
    pooled_len += lens
    out.append(f"| {name} | {len(nt)} | {100*mean([float(h) for h in hits]):.1f} | {mean(lens):.0f} |")
out.append(f"| pooled | {len(pooled_hits)} | {100*mean([float(h) for h in pooled_hits]):.1f} | {mean(pooled_len):.0f} |")
out.append("\nPaper Table 3 (1.5B, AdaptThink NoThinking cases): 7.9% implicit thinking, avg length 826.")
out.append("(Paper pools its own test cases, so compare with the pooled row only roughly.)")

text = "\n".join(out)
print(text)
(ROOT / "results").mkdir(exist_ok=True)
(ROOT / "results" / "level_analysis.md").write_text(text + "\n", encoding="utf-8")

# ---- optional plot ----
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = [lvnum(lv) for lv in order]
    nt_ours = [100 * mean([1.0 if x["is_nothinking"] else 0.0 for x in levels[lv]["adapt"]]) for lv in order]
    nt_paper = [PAPER_NT.get(x, float("nan")) for x in xs]
    acc_a = [100 * mean([x["acc"] for x in levels[lv]["adapt"]]) for lv in order]
    acc_o = [100 * mean([x["acc"] for x in levels[lv]["orig"]]) for lv in order]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    ax[0].plot(xs, nt_ours, "o-", label="ours"); ax[0].plot(xs, nt_paper, "s--", label="paper")
    ax[0].set_title("NoThinking % by MATH level"); ax[0].set_xlabel("Level"); ax[0].legend()
    ax[1].plot(xs, acc_a, "o-", label="AdaptThink"); ax[1].plot(xs, acc_o, "s--", label="Original")
    ax[1].set_title("Accuracy by MATH level"); ax[1].set_xlabel("Level"); ax[1].legend()
    fig.tight_layout(); fig.savefig(ROOT / "results" / "level_analysis.png", dpi=150)
    print("\nSaved results/level_analysis.png")
except Exception as e:
    print("\n(plot skipped:", e, ") -> pip install matplotlib to enable")
