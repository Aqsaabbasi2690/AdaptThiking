"""Modification 4: ONLINE implicit-thinking fallback (early abort + regenerate in Thinking mode).

Same sampling, data and grader as analysis/gen_natural.py, so outputs are directly comparable
with outputs/natural/*.jsonl. For every sample:

  1. Sample the first response token (temperature 0.6, top-p 0.95), i.e. the model picks its mode.
  2. Continue the response. If the mode is NoThinking, generation STOPS as soon as a trigger
     word ("Wait" / "Alternatively") appears, or (optional) after --nt_max_tokens tokens.
  3. Aborted NoThinking samples are regenerated in Thinking mode (prefix "Alright", token 71486,
     the same forced-thinking prefix the paper uses during training).

Reported length is the HONEST total cost: tokens spent on the aborted NoThinking attempt
+ tokens of the Thinking regeneration. With --no_fallback the script is plain natural sampling
(useful for MMLU, which has no natural-sampling run yet).

Sampling token 1 and then the rest is the same distribution as sampling the whole response
at once, so the "natural" part matches gen_natural.py.
"""
import argparse, json, os, re, signal, sys, time
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.adapt_think_rm import adapt_think_rm

PATHS = {"gsm8k": "data/test/preprocessed_data/gsm8k.parquet",
         "math": "data/test/preprocessed_data/math.parquet",
         "aime": "data/test/preprocessed_data/aime*16.parquet",   # 16 copies per problem -> use --n 1
         "mmlu": "data/test/preprocessed_data/mmlu.parquet",
         "val": "data/train/preprocessed_data/val200.parquet"}
EOT, WSTART = 151649, 71486          # "</think>", "Alright"
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

def finish_ids(c, eos_id):
    """Token ids of a finished completion, with EOS appended on a natural stop (as in gen_natural.py)."""
    ids = list(c.token_ids)
    if c.finish_reason == "stop" and c.stop_reason is None and (not ids or ids[-1] != eos_id):
        ids.append(eos_id)
    return ids

def run(llm, tok, ids_list, args, SamplingParams):
    """Returns, per problem, a list of per-sample dicts. Separated from main() so it can be tested with a fake llm."""
    eos_id = tok.eos_token_id
    triggers = [t for t in args.triggers.split(",") if t]
    base = dict(temperature=0.6, top_p=0.95)

    # Stage 1: first token = mode choice.
    first = llm.generate([{"prompt_token_ids": i} for i in ids_list],
                         SamplingParams(n=args.n, max_tokens=1, **base))
    jobs = []  # (problem k, sample s, first token, is_nt_start)
    for k, o in enumerate(first):
        for s, c in enumerate(o.outputs):
            t0 = c.token_ids[0]
            jobs.append((k, s, t0, tok.decode([t0]).lstrip().startswith("</think>")))

    # Stage 2: continue every sample; NoThinking samples get the trigger stop words.
    prompts, sps = [], []
    for k, s, t0, nt in jobs:
        prompts.append({"prompt_token_ids": ids_list[k] + [t0]})
        if nt and not args.no_fallback:
            mt = args.nt_max_tokens - 1 if args.nt_max_tokens else args.max_tokens - 1
            sps.append(SamplingParams(n=1, max_tokens=mt, stop=triggers, include_stop_str_in_output=True, **base))
        else:
            sps.append(SamplingParams(n=1, max_tokens=args.max_tokens - 1, **base))
    cont = llm.generate(prompts, sps)

    recs = [[None] * args.n for _ in ids_list]
    redo = []  # jobs whose NoThinking attempt was aborted
    for j, ((k, s, t0, nt), o) in enumerate(zip(jobs, cont)):
        c = o.outputs[0]
        ids = [t0] + finish_ids(c, eos_id)
        aborted = False
        if nt and not args.no_fallback:
            hit_kw = c.finish_reason == "stop" and c.stop_reason in triggers
            hit_len = bool(args.nt_max_tokens) and c.finish_reason == "length"
            aborted = hit_kw or hit_len
            reason = "keyword" if hit_kw else ("length" if hit_len else "")
        if aborted:
            redo.append((k, s, len(ids), reason))
        else:
            recs[k][s] = dict(ids=ids, finish=c.finish_reason, fallback=False, aborted_len=0, abort_reason="")

    # Stage 3: regenerate aborted samples in Thinking mode.
    if redo:
        out3 = llm.generate([{"prompt_token_ids": ids_list[k] + [WSTART]} for k, *_ in redo],
                            SamplingParams(n=1, max_tokens=args.max_tokens - 1, **base))
        for (k, s, alen, reason), o in zip(redo, out3):
            c = o.outputs[0]
            recs[k][s] = dict(ids=[WSTART] + finish_ids(c, eos_id), finish=c.finish_reason,
                              fallback=True, aborted_len=alen, abort_reason=reason)
    return recs, len(redo), len(jobs)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="merged model dir, e.g. outputs/sft/A3_lr3e5 or ~/models/AdaptThink-1.5B-delta0.05")
    ap.add_argument("--dataset", required=True, choices=list(PATHS))
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="random subset of this many problems (seed --seed); 0 = all")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_tokens", type=int, default=16384)
    ap.add_argument("--nt_max_tokens", type=int, default=0, help="also abort NoThinking attempts longer than this (0 = off)")
    ap.add_argument("--triggers", default="Wait,Alternatively")
    ap.add_argument("--no_fallback", action="store_true", help="plain natural sampling")
    ap.add_argument("--gpu_util", type=float, default=0.85)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    df = pd.read_parquet(ROOT / PATHS[args.dataset])
    if args.limit:
        df = df.sample(n=min(args.limit, len(df)), random_state=args.seed)
    tok = AutoTokenizer.from_pretrained(args.model)
    keep, problems, golds, ids_list = [], [], [], []
    for i, (_, row) in zip(df.index, df.iterrows()):
        msgs = [dict(m) for m in row["prompt"]]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        assert text.endswith("<think>\n"), repr(text[-30:])
        ids = tok(text, add_special_tokens=False).input_ids
        if len(ids) > 4096:
            continue
        keep.append(int(i)); problems.append(msgs[-1]["content"]); golds.append(row["reward_model"]["ground_truth"]); ids_list.append(ids)
    print(f"{len(ids_list)} problems, n={args.n}, fallback={'off' if args.no_fallback else 'on'}", flush=True)

    llm = LLM(model=args.model, dtype="bfloat16", gpu_memory_utilization=args.gpu_util,
              max_model_len=4096 + args.max_tokens, max_num_seqs=128, seed=args.seed)
    t0 = time.time()
    recs, n_redo, n_all = run(llm, tok, ids_list, args, SamplingParams)
    secs = time.time() - t0
    print(f"generation done in {secs:.0f}s; fallback on {n_redo}/{n_all} samples", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with open(args.out, "w", encoding="utf-8") as f:
        for k in range(len(ids_list)):
            for s, r in enumerate(recs[k]):
                text = tok.decode(r["ids"], skip_special_tokens=False).replace(EOS_TXT, "")
                body = text.lstrip(); nt = body.startswith("</think>")
                rec = dict(idx=keep[k], sample=s, problem=problems[k], gold=golds[k],
                           length=len(r["ids"]) + r["aborted_len"],      # total tokens generated (honest cost)
                           final_length=len(r["ids"]), aborted_len=r["aborted_len"], abort_reason=r["abort_reason"],
                           fallback=r["fallback"], finish=r["finish"], acc=grade(text, golds[k]), is_nt=nt,
                           implicit=bool(nt and KW.search(body.split("</think>", 1)[-1])), text=text)
                rows.append(rec); f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    m = lambda x: sum(x) / len(x) if x else float("nan")
    summ = dict(dataset=args.dataset, model=args.model, fallback=not args.no_fallback, nt_max_tokens=args.nt_max_tokens,
                n_samples=len(rows), acc=100 * m([r["acc"] for r in rows]), avg_len=m([r["length"] for r in rows]),
                nt_share=100 * m([r["is_nt"] for r in rows]), fallback_share=100 * m([r["fallback"] for r in rows]),
                cut_at_max=100 * m([r["finish"] == "length" for r in rows]), seconds=secs)
    print(json.dumps(summ), flush=True)
    json.dump(summ, open(args.out.replace(".jsonl", ".summary.json"), "w"))

if __name__ == "__main__":
    main()
    sys.stdout.flush(); os._exit(0)   # vLLM can hang at interpreter exit
