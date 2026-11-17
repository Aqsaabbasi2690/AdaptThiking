"""Gradio demo: compare the original AdaptThink-1.5B (delta 0.05) with the LoRA-tuned variants.
Shows which mode the model chose (Thinking vs NoThinking), P(</think>) at the first token, tokens and time."""
import os, threading, time
from pathlib import Path
import torch, gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

ROOT = Path(__file__).resolve().parent.parent
EOT, WSTART = 151649, 71486          # "</think>", "Alright"
EOS_TXT = "<｜end▁of▁sentence｜>"
MODELS = {
    "AdaptThink-1.5B delta0.05 (original)": os.path.expanduser("~/models/AdaptThink-1.5B-delta0.05"),
    "LoRA A3 (lr 3e-5)": str(ROOT / "outputs/sft/A3_lr3e5"),
    "LoRA A7 (mode-preserving, lr 3e-5)": str(ROOT / "outputs/sft/A7_modeshort"),
}
MODELS = {k: v for k, v in MODELS.items() if os.path.isdir(v)}
_cache = {}

def get(name):
    if name not in _cache:
        tok = AutoTokenizer.from_pretrained(MODELS[name])
        model = AutoModelForCausalLM.from_pretrained(MODELS[name], torch_dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
        _cache[name] = (tok, model)
    return _cache[name]

def run(question, model_name, mode, max_new, temperature):
    if not question or not question.strip():
        yield "", "Type a question first."; return
    tok, model = get(model_name)
    text = tok.apply_chat_template([{"role": "user", "content": question}], tokenize=False, add_generation_prompt=True)
    ids = tok(text, add_special_tokens=False, return_tensors="pt").input_ids.to("cuda")
    with torch.no_grad():
        probs = torch.softmax(model(ids).logits[0, -1].float(), -1)
    p_nt, p_th = probs[EOT].item(), probs[WSTART].item()
    prefix = ""
    if mode.startswith("NoThinking"):
        ids = torch.cat([ids, torch.tensor([[EOT]], device="cuda")], 1); prefix = tok.decode([EOT])
    elif mode.startswith("Thinking"):
        ids = torch.cat([ids, torch.tensor([[WSTART]], device="cuda")], 1); prefix = tok.decode([WSTART])
    streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=False)
    kw = dict(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=int(max_new), do_sample=True,
              temperature=float(temperature), top_p=0.95, streamer=streamer)
    threading.Thread(target=model.generate, kwargs=kw).start()
    out, t0 = prefix, time.time()
    def info(done=False):
        body = out.lstrip()
        chosen = "NoThinking" if body.startswith("</think>") else ("Thinking" if body else "...")
        n = len(tok(out, add_special_tokens=False).input_ids) if done else None
        return (f"Mode used: {chosen} ({'forced' if not mode.startswith('Auto') else 'chosen by the model'})\n"
                f"P(</think>) at the first token: {p_nt:.3f}   P('Alright'): {p_th:.3f}\n"
                f"Time: {time.time()-t0:.1f}s" + (f"   Tokens: {n}" if n else ""))
    for piece in streamer:
        out += piece
        yield out.replace(EOS_TXT, ""), info()
    out = out.replace(EOS_TXT, "")
    yield out, info(done=True)

with gr.Blocks(title="AdaptThink demo") as demo:
    gr.Markdown("## AdaptThink-1.5B: when does the model think?\nCompare the original checkpoint with LoRA-tuned variants on the same question. "
                "The answer starts with `</think>` when the model skips thinking.")
    q = gr.Textbox(label="Question", lines=3, value="Janet's ducks lay 16 eggs per day. She eats 3 and bakes muffins with 4. She sells the rest at $2 per egg. How much does she make daily?")
    with gr.Row():
        m = gr.Dropdown(list(MODELS), value=list(MODELS)[0], label="Model")
        mode = gr.Radio(["Auto (model chooses)", "NoThinking (force)", "Thinking (force)"], value="Auto (model chooses)", label="Mode")
    with gr.Row():
        mx = gr.Slider(256, 8192, value=2048, step=256, label="Max new tokens")
        tp = gr.Slider(0.1, 1.2, value=0.6, step=0.1, label="Temperature")
    go = gr.Button("Run")
    info_box = gr.Textbox(label="What happened", lines=3)
    ans = gr.Textbox(label="Model output", lines=18)
    go.click(run, [q, m, mode, mx, tp], [ans, info_box])
demo.queue().launch(server_name="0.0.0.0", server_port=7860)
