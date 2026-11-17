"""Build SFT data from sampled train responses. Options make each design choice a separate ablation."""
import argparse, json, random, re, collections, statistics as st
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "outputs/natural"
ap = argparse.ArgumentParser()
ap.add_argument("--select", choices=["shortest", "random", "shortest_mode"], default="shortest")
ap.add_argument("--no_implicit_filter", action="store_true")
ap.add_argument("--mode", choices=["both", "nt", "think"], default="both")
ap.add_argument("--k", type=int, default=4, help="use only the first k samples per problem")
ap.add_argument("--frac", type=float, default=1.0, help="fraction of problems to use")
ap.add_argument("--max_len", type=int, default=6144)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default=str(D / "sft_data.jsonl"))
args = ap.parse_args()
rng = random.Random(args.seed)
norm = lambda s: re.sub(r"\s+", " ", s).strip().lower()
test_probs = set()
for n in ["gsm8k", "math", "aime"]:
    for r in json.load(open(ROOT / f"data/test/{n}.json", encoding="utf-8")):
        test_probs.add(norm(r["problem"]))
rows = []
shards = sorted(p for p in D.glob("train_shard*.jsonl") if Path(str(p) + ".done").exists())
for p in shards:
    rows += [json.loads(l) for l in open(p, encoding="utf-8")]
print(f"shards used: {len(shards)}, samples: {len(rows)}")
by = collections.defaultdict(list)
for r in rows:
    by[r["idx"]].append(r)
m = lambda x: sum(x) / len(x) if x else float("nan")
probs = sorted(by); rng.shuffle(probs); probs = probs[: int(len(probs) * args.frac)]
overlap = any_ok = n_ok = n_clean = 0; chosen = []
for idx in probs:
    rs = [r for r in by[idx] if r["sample"] < args.k]
    if norm(rs[0]["problem"]) in test_probs:
        overlap += 1; continue
    ok = [r for r in rs if r["acc"] == 1.0 and r["finish"] == "stop" and r["length"] <= args.max_len]
    if args.mode == "nt": ok = [r for r in ok if r["is_nt"]]
    if args.mode == "think": ok = [r for r in ok if not r["is_nt"]]
    n_ok += len(ok)
    clean = ok if args.no_implicit_filter else [r for r in ok if not r["implicit"]]
    n_clean += len(clean); any_ok += bool(ok)
    if clean:
        if args.select == "shortest_mode":
            maj_nt = 2 * sum(r["is_nt"] for r in rs) > len(rs)          # model's usual mode for this problem (ties -> Thinking)
            pool = [r for r in clean if r["is_nt"] == maj_nt] or clean   # switch mode only if the usual mode has no correct sample
            chosen.append(min(pool, key=lambda r: (r["length"], r["sample"])))
        else:
            chosen.append(min(clean, key=lambda r: (r["length"], r["sample"])) if args.select == "shortest" else rng.choice(clean))
print(f"problems considered: {len(probs)}, overlapping with test sets (dropped): {overlap}")
print(f"problems with >=1 usable sample: {any_ok}; usable samples: {n_ok}, dropped by implicit filter: {n_ok - n_clean}")
print(f"chosen training examples: {len(chosen)}")
if chosen:
    nt = [r for r in chosen if r["is_nt"]]
    print(f"  NoThinking share: {100*len(nt)/len(chosen):.1f}% (all samples: {100*m([r['is_nt'] for r in rows]):.1f}%)")
    print(f"  avg length chosen: {m([r['length'] for r in chosen]):.0f} (all samples: {m([r['length'] for r in rows]):.0f})")
    print(f"  total response tokens: {sum(r['length'] for r in chosen)}")
    q = st.quantiles([r["length"] for r in chosen], n=100)
    print(f"  length p50={int(q[49])}, p90={int(q[89])}, p99={int(q[98])}")
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
with open(args.out, "w", encoding="utf-8") as f:
    for r in chosen:
        f.write(json.dumps(dict(idx=r["idx"], problem=r["problem"], token_ids=r["token_ids"], is_nt=r["is_nt"], length=r["length"]), ensure_ascii=False) + "\n")
print("saved", args.out)
