"""Tables for Modification 4 (online fallback) and the MMLU evaluation.
Same paired-bootstrap method as analysis/test_table.py, baseline = AdaptThink d0.05 natural sampling.
Writes results/fallback_online.md and results/mmlu.md."""
import json, collections
import numpy as np
from pathlib import Path
R = Path(__file__).resolve().parent.parent
N = R / "outputs/natural"
SETS = [("GSM8K", "base_gsm8k.jsonl", "test_{}_gsm8k.jsonl", "idx"),
        ("MATH500", "base_math.jsonl", "test_{}_math.jsonl", "idx"),
        ("AIME24", "test_base_aime.jsonl", "test_{}_aime.jsonl", "problem")]
RUNS = [("AdaptThink d0.05 + online fallback", "basefb"),
        ("LoRA A3 (lr 3e-5)", "A3"),
        ("LoRA A3 + online fallback", "A3fb"),
        ("LoRA A3 + online fallback + NT cap", "A3fbcap")]
M = {"acc": lambda r: 100 * r["acc"], "len": lambda r: r["length"], "nt": lambda r: 100 * r["is_nt"],
     "fb": lambda r: 100 * r.get("fallback", False), "cut": lambda r: 100 * (r["finish"] == "length")}

def load(p, key):
    d = collections.defaultdict(list)
    for l in open(p, encoding="utf-8"):
        r = json.loads(l); d[r[key]].append(r)
    return d
pp = lambda d, ids, f: np.array([np.mean([f(r) for r in d[i]]) for i in ids])

def compare(base, d):
    common = sorted(set(base) & set(d))
    rng = np.random.default_rng(0); boot = rng.integers(0, len(common), (5000, len(common)))
    res = {}
    for k in M:
        x, y = pp(base, common, M[k]), pp(d, common, M[k]); diff = y - x
        res[k] = (y.mean(), diff.mean(), x.mean(), *np.percentile(diff[boot].mean(1), [2.5, 97.5]))
    return res, len(common)

def row(ds, name, res):
    a, l, n, fb, c = res["acc"], res["len"], res["nt"], res["fb"], res["cut"]
    return (f"| {ds} | {name} | {a[0]:.1f} ({a[1]:+.2f} [{a[3]:+.2f}, {a[4]:+.2f}]) | "
            f"{l[0]:.0f} ({l[1]:+.0f}, {100*l[1]/l[2]:+.1f}% [{l[3]:+.0f}, {l[4]:+.0f}]) | {n[0]:.1f} ({n[1]:+.1f}) | {fb[0]:.1f} | {c[0]:.2f} ({c[1]:+.2f}) |")

HDR = ["| Dataset | Model | Accuracy % (diff [95% CI]) | Avg length incl. aborted tokens (diff, % [95% CI]) | NoThinking % (diff) | Fallback % | Cut at 16K % (diff) |",
       "|---|---|---|---|---|---|---|"]
out = list(HDR)
for ds, bf, tf, key in SETS:
    if not (N / bf).exists():
        out.append(f"| {ds} | (baseline file {bf} missing) | | | | | |"); continue
    base = load(N / bf, key); ids = sorted(base)
    b = {k: pp(base, ids, M[k]).mean() for k in M}
    out.append(f"| {ds} ({len(ids)} problems) | AdaptThink d0.05 (base) | {b['acc']:.1f} | {b['len']:.0f} | {b['nt']:.1f} | 0.0 | {b['cut']:.1f} |")
    for name, tag in RUNS:
        p = N / tf.format(tag)
        if p.exists():
            out.append(row(ds, name, compare(base, load(p, key))[0]))
text = "\n".join(out); print(text)
(R / "results").mkdir(exist_ok=True); (R / "results/fallback_online.md").write_text(text + "\n", encoding="utf-8")

# MMLU (out-of-domain, paper Table 4). Baseline for the diffs = AdaptThink d0.05.
MM = [("DeepSeek-R1-Distill-Qwen-1.5B (original)", "mmlu_orig"), ("AdaptThink d0.05", "mmlu_base"),
      ("LoRA A3", "mmlu_A3"), ("LoRA A3 + online fallback", "mmlu_A3fb")]
if (N / "mmlu_base.jsonl").exists():
    base = load(N / "mmlu_base.jsonl", "idx")
    out = list(HDR)
    for name, tag in MM:
        p = N / f"{tag}.jsonl"
        if not p.exists(): continue
        if tag == "mmlu_base":
            ids = sorted(base); b = {k: pp(base, ids, M[k]).mean() for k in M}
            out.append(f"| MMLU ({len(ids)} questions) | {name} (base) | {b['acc']:.1f} | {b['len']:.0f} | {b['nt']:.1f} | 0.0 | {b['cut']:.1f} |")
        else:
            out.append(row("MMLU", name, compare(base, load(p, "idx"))[0]))
    text = "\n".join(out); print("\n" + text)
    (R / "results/mmlu.md").write_text(text + "\n", encoding="utf-8")
