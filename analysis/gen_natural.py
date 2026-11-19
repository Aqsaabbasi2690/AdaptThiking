"""Natural-policy sampling (the model picks its own mode) with standalone vLLM, graded with the repo's grader."""
import argparse, json, os, re, signal, sys, time
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from src.adapt_think_rm import adapt_think_rm

PATHS = {"gsm8k": "data/test/preprocessed_data/gsm8k.parquet",
         "math": "data/test/preprocessed_data/math.parquet",
         "aime": "data/test/preprocessed_data/aime*16.parquet",   # already 16 copies per problem -> use --n 1
         "train": "data/train/preprocessed_data/deepscaler.parquet",
         "val": "data/train/preprocessed_data/val200.parquet"}
KW = re.compile(r"\b(Wait|Alternatively)\b")
EOS_TXT = "<｜end▁of▁sentence｜>"

class TO(Exception): pass
def _h(s, f): raise TO()
signal.signal(signal.SIGALRM, _h)
def grade(sol, gold):
    signal.alarm(20)
    try: return float(adapt_think_rm("math", sol, gold)["acc"])
    except Exception: return 0.0
    finally: signal.alarm(0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.expanduser("~/models/AdaptThink-1.5B-delta0.05"))
    ap.add_argument("--dataset", required=True, choices=list(PATHS))
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_tokens", type=int, default=16384)
    ap.add_argument("--gpu_util", type=float, default=0.85)
    ap.add_argument("--save_ids", action="store_true")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    df = pd.read_parquet(ROOT / PATHS[args.dataset])
    if args.limit:
        df = df.sample(n=min(args.limit, len(df)), random_state=args.seed) if args.dataset == "train" else df.iloc[: args.limit]
    if args.num_shards > 1:
        df = df.iloc[args.shard::args.num_shards]
    tok = AutoTokenizer.from_pretrained(args.model)
    eos_id = tok.eos_token_id
    keep, problems, golds, ids_list = [], [], [], []
    for i, (_, row) in zip(df.index, df.iterrows()):
        msgs = [dict(m) for m in row["prompt"]]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        assert text.endswith("<think>\n"), repr(text[-30:])
        ids = tok(text, add_special_tokens=False).input_ids
        if len(ids) > 4096:
            continue
        keep.append(int(i)); problems.append(msgs[-1]["content"]); golds.append(row["reward_model"]["ground_truth"]); ids_list.append(ids)
    print(f"{len(ids_list)} problems, n={args.n}", flush=True)

    llm = LLM(model=args.model, dtype="bfloat16", gpu_memory_utilization=args.gpu_util,
              max_model_len=4096 + args.max_tokens, max_num_seqs=128, seed=args.seed)
    sp = SamplingParams(n=args.n, temperature=0.6, top_p=0.95, max_tokens=args.max_tokens)
    t0 = time.time()
    outs = llm.generate([{"prompt_token_ids": i} for i in ids_list], sp)
    print(f"generation done in {time.time()-t0:.0f}s", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    recs = []
    with open(args.out, "w", encoding="utf-8") as f:
        for k, o in enumerate(outs):
            for s, c in enumerate(o.outputs):
                ids = list(c.token_ids)
                if c.finish_reason == "stop" and (not ids or ids[-1] != eos_id):
                    ids.append(eos_id)
                text = tok.decode(ids, skip_special_tokens=False).replace(EOS_TXT, "")
                acc = grade(text, golds[k])
                body = text.lstrip()
                nt = body.startswith("</think>")
                rec = dict(idx=keep[k], sample=s, problem=problems[k], gold=golds[k], length=len(ids), finish=c.finish_reason,
                           acc=acc, is_nt=nt, implicit=bool(nt and KW.search(body.split("</think>", 1)[-1])), text=text)
                if args.save_ids:
                    rec["token_ids"] = ids
                recs.append(rec)
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    m = lambda x: sum(x) / len(x) if x else float("nan")
    nts = [r for r in recs if r["is_nt"]]
    summ = dict(dataset=args.dataset, model=args.model, n_samples=len(recs), acc=100 * m([r["acc"] for r in recs]),
                avg_len=m([r["length"] for r in recs]), nt_share=100 * m([r["is_nt"] for r in recs]),
                implicit_among_nt=100 * m([r["implicit"] for r in nts]), cut_at_max=100 * m([r["finish"] == "length" for r in recs]))
    print(json.dumps(summ), flush=True)
    json.dump(summ, open(args.out.replace(".jsonl", ".summary.json"), "w"))

if __name__ == "__main__":
    main()
    sys.stdout.flush(); os._exit(0)   # vLLM can hang at interpreter exit
