"""Generate, for every test problem: P(</think>) at the first token, one forced-NoThinking answer
and one forced-Thinking answer, using standalone vLLM (single 20 GB GPU). Grading is done separately."""
import argparse, json, math, os, time
import pandas as pd
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

EOT, WSTART = 151649, 71486  # "</think>", "Alright"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.expanduser("~/models/AdaptThink-1.5B-delta0.05"))
    ap.add_argument("--dataset", default="math", choices=["math", "gsm8k"])
    ap.add_argument("--limit", type=int, default=0, help="0 = all problems")
    ap.add_argument("--max_tokens", type=int, default=16384)
    ap.add_argument("--gpu_util", type=float, default=0.85)
    ap.add_argument("--eager", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    df = pd.read_parquet(f"data/test/preprocessed_data/{args.dataset}.parquet")
    if args.limit:
        df = df.iloc[: args.limit]
    tok = AutoTokenizer.from_pretrained(args.model)
    problems, golds, ids_list = [], [], []
    for _, row in df.iterrows():
        msgs = [dict(m) for m in row["prompt"]]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        assert text.endswith("<think>\n"), repr(text[-30:])
        problems.append(msgs[-1]["content"])
        golds.append(row["reward_model"]["ground_truth"])
        ids_list.append(tok(text, add_special_tokens=False).input_ids)
    n = len(ids_list)

    llm = LLM(model=args.model, dtype="bfloat16", gpu_memory_utilization=args.gpu_util,
              max_model_len=4096 + args.max_tokens, max_num_seqs=128,
              enforce_eager=args.eager, seed=0)
    sp_first = SamplingParams(temperature=1.0, max_tokens=1, logprobs=20)
    sp_gen = SamplingParams(temperature=0.6, top_p=0.95, max_tokens=args.max_tokens - 1)

    t0 = time.time()
    first = llm.generate([{"prompt_token_ids": i} for i in ids_list], sp_first)
    p_nt = []
    for o in first:
        lp = o.outputs[0].logprobs[0]
        p_nt.append(math.exp(lp[EOT].logprob) if EOT in lp else 0.0)
    print(f"first-token probs done in {time.time()-t0:.0f}s", flush=True)

    nt = llm.generate([{"prompt_token_ids": i + [EOT]} for i in ids_list], sp_gen)
    print(f"NoThinking generations done at {time.time()-t0:.0f}s", flush=True)
    th = llm.generate([{"prompt_token_ids": i + [WSTART]} for i in ids_list], sp_gen)
    print(f"Thinking generations done at {time.time()-t0:.0f}s", flush=True)

    out = args.out or f"outputs/router/{os.path.basename(args.model)}__{args.dataset}.jsonl"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for k in range(n):
            a, b = nt[k].outputs[0], th[k].outputs[0]
            f.write(json.dumps(dict(
                idx=k, problem=problems[k], gold=golds[k], p_nt=p_nt[k],
                nt_response=tok.decode([EOT] + list(a.token_ids)), nt_len=1 + len(a.token_ids), nt_finish=a.finish_reason,
                think_response=tok.decode([WSTART] + list(b.token_ids)), think_len=1 + len(b.token_ids), think_finish=b.finish_reason,
            ), ensure_ascii=False) + "\n")
    mean = lambda x: sum(x) / len(x)
    print(f"saved {out}: n={n}, mean p_nt={mean(p_nt):.3f}, frac p_nt>0.5={mean([p > 0.5 for p in p_nt]):.3f}, "
          f"mean NT len={mean([1+len(nt[k].outputs[0].token_ids) for k in range(n)]):.0f}, "
          f"mean Thinking len={mean([1+len(th[k].outputs[0].token_ids) for k in range(n)]):.0f}", flush=True)


if __name__ == "__main__":
    main()
