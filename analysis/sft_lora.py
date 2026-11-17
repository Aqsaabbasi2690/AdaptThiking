"""LoRA SFT on self-distilled samples. Loss on response tokens only, chunked CE; merges and saves a full model."""
import argparse, json, math, os, random, time
import torch, torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model

ap = argparse.ArgumentParser()
ap.add_argument("--model", default=os.path.expanduser("~/models/AdaptThink-1.5B-delta0.05"))
ap.add_argument("--data", default="outputs/natural/sft_data.jsonl")
ap.add_argument("--out", required=True)
ap.add_argument("--rank", type=int, default=16)
ap.add_argument("--alpha", type=int, default=0, help="0 means 2*rank")
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--epochs", type=int, default=2)
ap.add_argument("--bs", type=int, default=8, help="examples per optimizer step")
ap.add_argument("--max_examples", type=int, default=0)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
random.seed(args.seed); torch.manual_seed(args.seed)

tok = AutoTokenizer.from_pretrained(args.model)
rows = [json.loads(l) for l in open(args.data, encoding="utf-8")]
if args.max_examples:
    rows = rows[: args.max_examples]
exs = []
for r in rows:
    text = tok.apply_chat_template([{"role": "user", "content": r["problem"]}], tokenize=False, add_generation_prompt=True)
    assert text.endswith("<think>\n")
    exs.append((tok(text, add_special_tokens=False).input_ids, r["token_ids"]))
print(f"{len(exs)} examples, {sum(len(r) for _, r in exs)} response tokens", flush=True)

model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa").cuda()
model.config.use_cache = False
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
cfg = LoraConfig(r=args.rank, lora_alpha=args.alpha or 2 * args.rank, lora_dropout=0.05, task_type="CAUSAL_LM",
                 target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
model = get_peft_model(model, cfg)
model.print_trainable_parameters()
base = model.get_base_model()
inner, head = base.model, base.lm_head
params = [p for p in model.parameters() if p.requires_grad]
opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
spe = math.ceil(len(exs) / args.bs); total = spe * args.epochs
sched = get_cosine_schedule_with_warmup(opt, max(1, int(0.05 * total)), total)

def ce_chunked(h, labels, chunk=1024):
    tot = h.new_zeros((), dtype=torch.float32)
    for i in range(0, h.size(0), chunk):
        def f(hc, lc):
            return F.cross_entropy(head(hc).float(), lc, reduction="sum")
        tot = tot + checkpoint(f, h[i:i + chunk], labels[i:i + chunk], use_reentrant=False)
    return tot

model.train(); t0 = time.time(); step = 0; log = []
for ep in range(args.epochs):
    order = list(range(len(exs))); random.shuffle(order)
    for b in range(0, len(order), args.bs):
        batch = [exs[i] for i in order[b:b + args.bs]]
        ntok = sum(len(r) for _, r in batch); lv = 0.0
        for p, r in batch:
            ids = torch.tensor([p + r], device="cuda")
            h = inner(input_ids=ids, use_cache=False).last_hidden_state[0]
            loss = ce_chunked(h[len(p) - 1:-1], ids[0, len(p):]) / ntok
            loss.backward(); lv += loss.item()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True); step += 1
        log.append(dict(step=step, epoch=ep, loss=lv, lr=sched.get_last_lr()[0]))
        if step % 5 == 0 or step == total:
            print(f"step {step}/{total} epoch {ep} loss {lv:.4f} lr {sched.get_last_lr()[0]:.2e} elapsed {time.time()-t0:.0f}s", flush=True)

model = model.merge_and_unload()
model.config.use_cache = True
os.makedirs(args.out, exist_ok=True)
model.save_pretrained(args.out, safe_serialization=True); tok.save_pretrained(args.out)
json.dump(dict(args=vars(args), log=log, seconds=time.time() - t0), open(os.path.join(args.out, "train_log.json"), "w"))
print("saved", args.out, flush=True)
