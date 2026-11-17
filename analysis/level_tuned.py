import json, re, collections
from pathlib import Path
R = Path(__file__).resolve().parent.parent
norm = lambda s: re.sub(r"\s+", " ", s).strip().lower()
lv = {norm(d["problem"]): int(float(d["difficulty"])) for d in json.load(open(R / "data/test/math.json", encoding="utf-8"))}
RUNS = [("AdaptThink d0.05 (base)", "base_math.jsonl"), ("LoRA main", "test_main_math.jsonl"), ("LoRA A3 (lr 3e-5)", "test_A3_math.jsonl")]
m = lambda x: sum(x) / len(x) if x else float("nan")
data, miss = {}, {}
for name, f in RUNS:
    by = collections.defaultdict(list); miss[name] = 0
    for l in open(R / "outputs/natural" / f, encoding="utf-8"):
        r = json.loads(l); k = lv.get(norm(r["problem"]))
        if k is None: miss[name] += 1
        else: by[k].append(r)
    data[name] = by
out = [f"Unmatched rows: {miss}", "", "| Level | n problems | " + " | ".join(f"{n}: NT % / acc % / len" for n, _ in RUNS) + " |", "|---|---|" + "---|" * len(RUNS)]
for k in sorted(data[RUNS[0][0]]):
    cells = [f"{100*m([r['is_nt'] for r in data[n][k]]):.0f} / {100*m([r['acc'] for r in data[n][k]]):.1f} / {m([r['length'] for r in data[n][k]]):.0f}" for n, _ in RUNS]
    out.append(f"| {k} | {len(data[RUNS[0][0]][k]) // 4} | " + " | ".join(cells) + " |")
text = "\n".join(out); print(text)
(R / "results").mkdir(exist_ok=True); (R / "results/level_tuned.md").write_text(text + "\n", encoding="utf-8")
