import numpy as np, json, glob
from config import *
def load(pat):
    d = {}
    for f in sorted(glob.glob(pat)): d.update(dict(np.load(f)))
    return d
R = {"1": load("out/snr_p1_bs2.npz"), "1b": load("out/snr_p1b_bs2.npz"), "2": load("out/snr_p2_bs2_R*.npz")}
B = {"1": dict(np.load("out/snr_p1.npz")), "1b": dict(np.load("out/snr_p1b.npz")), "2": dict(np.load("out/snr_p2.npz"))}
v = R["2"]["valid"]; db = lambda x: np.maximum(10*np.log10(np.maximum(x, 1e-30)), -60)
cfg = [(n, y) for n in RIS_CAND for y in YAW_OFFSETS]
def S(r, m, n, y, s, b="c"): return r[m][f"p{m}_{n}_{y:+.0f}_{s}_{b}"][v]
def sc(r, m, n, y, s, b="c"): x = S(r, m, n, y, s, b); return (float(np.mean(db(x) >= 10)), float(db(x).mean()))
def best(r, m, s, b="c"): return max(cfg, key=lambda c: sc(r, m, *c, s, b))
out = {}
for tag, r in (("BS1", B), ("BS2", R)):
    o = {"noRIS_cov": float(np.mean(db(r["2"]["p2_direct"][v]) >= 10)), "noRIS_snr": float(db(r["2"]["p2_direct"][v]).mean())}
    for s in N_SIDES:
        p1 = best(r, "1", s); p1b = best(r, "1b", s); p2 = best(r, "2", s)
        q = {f"{b}bit": best(r, "2", s, b) for b in (1, 2, 3)}
        o[s*s] = dict(P1=f"{p1[0]} {p1[1]:+.0f}", P1_pred=sc(r, "1", *p1, s), P1_inRT=sc(r, "2", *p1, s),
                     P1b=f"{p1b[0]} {p1b[1]:+.0f}", P1b_pred=sc(r, "1b", *p1b, s), P1b_inRT=sc(r, "2", *p1b, s),
                     P2=f"{p2[0]} {p2[1]:+.0f}", P2_val=sc(r, "2", *p2, s),
                     quant_best={k: f"{c[0]} {c[1]:+.0f}" for k, c in q.items()},
                     quant_loss_db={k: sc(r, "2", *p2, s)[1] - sc(r, "2", *p2, s, int(k[0]))[1] for k in q})
    out[tag] = o
json.dump(out, open("out/bs_compare.json", "w"), indent=1)
for tag in out:
    print(tag, "noRIS", round(100*out[tag]["noRIS_cov"], 1), round(out[tag]["noRIS_snr"], 1))
    for s in N_SIDES:
        d = out[tag][s*s]
        print(f"  N={s*s}: P1 {d['P1']} pred {100*d['P1_pred'][0]:.1f}%/{d['P1_pred'][1]:.1f} inRT {100*d['P1_inRT'][0]:.1f}%/{d['P1_inRT'][1]:.1f} | "
              f"P1b {d['P1b']} pred {100*d['P1b_pred'][0]:.1f}% inRT {100*d['P1b_inRT'][0]:.1f}% | RT {d['P2']} {100*d['P2_val'][0]:.1f}%/{d['P2_val'][1]:.1f} | "
              f"quant best {d['quant_best']} loss {{{', '.join(f'{k}:{x:.2f}' for k, x in d['quant_loss_db'].items())}}}")
