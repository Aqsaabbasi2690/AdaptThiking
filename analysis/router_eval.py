"""Grade forced-NoThinking / forced-Thinking answers with the repo's own grader, then simulate threshold routers
on P(</think>) offline. Controls: random mix at the same NoThinking share, oracle, the model's own stochastic policy."""
import json, signal, sys
import numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.adapt_think_rm import adapt_think_rm

DS = sys.argv[1] if len(sys.argv) > 1 else "math"
SRC = ROOT / "outputs/router" / f"AdaptThink-1.5B-delta0.05__{DS}.jsonl"
CACHE = ROOT / "outputs/router" / f"graded_{DS}.json"
EOS = "<｜end▁of▁sentence｜>"
# Our own delta-sweep results on MATH500 (results/delta_sweep.md): (label, avg length, acc)
DELTA_PTS = [("orig", 4593, 81.0), ("d0", 2489, 83.8), ("d0.01", 2631, 83.4), ("d0.02", 2215, 82.8),
             ("d0.05", 1796, 83.2), ("d0.075", 1670, 81.2), ("d0.1", 1220, 80.6)] if DS == "math" else []

class TO(Exception): pass
def _h(s, f): raise TO()
signal.signal(signal.SIGALRM, _h)

def grade(sol, gold):
    signal.alarm(20)
    try:
        return float(adapt_think_rm("math", sol.replace(EOS, ""), gold)["acc"])
    except Exception:
        return 0.0
    finally:
        signal.alarm(0)

rows = [json.loads(l) for l in open(SRC, encoding="utf-8")]
if CACHE.exists():
    g = json.load(open(CACHE))
else:
    g = {"nt": [grade(r["nt_response"], r["gold"]) for r in rows],
         "th": [grade(r["think_response"], r["gold"]) for r in rows]}
    json.dump(g, open(CACHE, "w"))
a_nt, a_th = np.array(g["nt"]), np.array(g["th"])
l_nt = np.array([r["nt_len"] for r in rows], float); l_th = np.array([r["think_len"] for r in rows], float)
p = np.array([r["p_nt"] for r in rows]); n = len(rows)
out = []
out.append(f"## Offline threshold router on {DS} (AdaptThink-1.5B-delta0.05), n={n}, 1 sample per mode\n")
out.append(f"Forced NoThinking: acc {100*a_nt.mean():.1f}%, avg len {l_nt.mean():.0f}, cut at 16K: {sum(r['nt_finish']=='length' for r in rows)}")
out.append(f"Forced Thinking:   acc {100*a_th.mean():.1f}%, avg len {l_th.mean():.0f}, cut at 16K: {sum(r['think_finish']=='length' for r in rows)}\n")

rng = np.random.default_rng(0)
perm = rng.permutation(n); A, B = perm[: n // 2], perm[n // 2:]
B_BOOT = 2000
boot = rng.integers(0, n, size=(B_BOOT, n))

def pt(route, idx=None):
    s = slice(None) if idx is None else idx
    r = route[s]
    return (np.where(r, a_nt[s], a_th[s]).mean(), np.where(r, l_nt[s], l_th[s]).mean(), r.mean())

out.append("| tau | NoThinking % | acc % | avg len | random-mix acc % (same NT share) | acc minus random-mix [95% CI] | acc % half A | acc % half B |")
out.append("|---|---|---|---|---|---|---|---|")
taus = [0.0] + [round(x, 2) for x in np.arange(0.05, 1.0, 0.05)] + [1.01]
curve = []
for t in taus:
    route = p >= t
    acc, ln, f = pt(route)
    rnd_acc = f * a_nt.mean() + (1 - f) * a_th.mean()
    rnd_len = f * l_nt.mean() + (1 - f) * l_th.mean()
    diffs = []
    for idx in boot:
        r = route[idx]; fb = r.mean()
        diffs.append(np.where(r, a_nt[idx], a_th[idx]).mean() - (fb * a_nt[idx].mean() + (1 - fb) * a_th[idx].mean()))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    curve.append((t, ln, acc, f, rnd_len, rnd_acc))
    out.append(f"| {t:.2f} | {100*f:.1f} | {100*acc:.1f} | {ln:.0f} | {100*rnd_acc:.1f} | {100*(acc-rnd_acc):+.1f} [{100*lo:+.1f}, {100*hi:+.1f}] | {100*pt(route,A)[0]:.1f} | {100*pt(route,B)[0]:.1f} |")

stoch = (np.mean(p * a_nt + (1 - p) * a_th), np.mean(p * l_nt + (1 - p) * l_th), p.mean())
both = np.where(a_nt >= a_th, a_nt, a_th)
orc_len = np.where(a_nt >= a_th, l_nt, l_th)
out.append(f"\nModel's own stochastic policy (route to NoThinking with probability p, T=1): acc {100*stoch[0]:.1f}%, len {stoch[1]:.0f}, NT share {100*stoch[2]:.1f}%")
out.append(f"Oracle (always the better mode, shorter on ties; not achievable): acc {100*both.mean():.1f}%, len {orc_len.mean():.0f}")
if DELTA_PTS:
    out.append("\nOur delta-sweep points for reference (label: avg len, acc %): " + ", ".join(f"{a}: {b}, {c}" for a, b, c in DELTA_PTS))
text = "\n".join(out)
print(text)
(ROOT / "results").mkdir(exist_ok=True)
(ROOT / "results" / f"router_curve_{DS}.md").write_text(text + "\n", encoding="utf-8")
try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot([c[1] for c in curve], [100*c[2] for c in curve], "o-", label="threshold router (tau sweep)")
    ax.plot([c[4] for c in curve], [100*c[5] for c in curve], "--", color="gray", label="random mix (same NT share)")
    if DELTA_PTS:
        ax.scatter([b for _, b, _ in DELTA_PTS], [c for _, _, c in DELTA_PTS], marker="s", color="C3", label="released delta checkpoints (ours)")
    ax.scatter([stoch[1]], [100*stoch[0]], marker="*", s=120, color="C2", label="model's own policy (p, T=1)")
    ax.scatter([orc_len.mean()], [100*both.mean()], marker="^", color="k", label="oracle")
    ax.set_xlabel("avg response length (tokens)"); ax.set_ylabel("accuracy %"); ax.set_title(f"{DS}: accuracy vs length"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(ROOT / "results" / f"router_curve_{DS}.png", dpi=140); print("saved plot")
except Exception as e:
    print("(plot skipped:", e, ")")
