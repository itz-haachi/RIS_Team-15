import numpy as np, json
from scipy.stats import wilcoxon
from config import *
r1=dict(np.load("out/snr_p1.npz")); r1b=dict(np.load("out/snr_p1b.npz")); r2=dict(np.load("out/snr_p2.npz"))
v=r2["valid"]; TH=10.0
db=lambda x: np.maximum(10*np.log10(np.maximum(x,1e-30)),-60)
rng=np.random.default_rng(2026); drops=[rng.choice(v.sum(),200,replace=False) for _ in range(20)]
def st(x):
    c=[np.mean(db(x[d])>=TH) for d in drops]; m=[db(x[d]).mean() for d in drops]; se=[np.mean(np.log2(1+x[d])) for d in drops]
    return dict(cov=np.mean(c),cov_sd=np.std(c,ddof=1),snr=np.mean(m),snr_sd=np.std(m,ddof=1),se=np.mean(se),se_sd=np.std(se,ddof=1))
S=lambda r,p,n,y,s,b="c": r[f"p{p}_{n}_{y:+.0f}_{s}_{b}"][v]
cfg=[(n,y) for n in RIS_CAND for y in YAW_OFFSETS]
def best(r,p,s): return max(cfg,key=lambda c:(np.mean(db(S(r,p,*c,s))>=TH),db(S(r,p,*c,s)).mean()))
out={}
for s in N_SIDES:
    b1=best(r1,1,s); b1b=best(r1b,"1b",s); b2=best(r2,2,s); ref=S(r2,2,*b2,s)
    rows={"no_ris":(r2["p2_direct"][v],"-"),"P1":(S(r1,1,*b1,s),b1),"P1b":(S(r1b,"1b",*b1b,s),b1b),
          "P2_at_P1":(S(r2,2,*b1,s),b1),"P2_at_P1b":(S(r2,2,*b1b,s),b1b),"P2":(ref,b2)}
    for bb in (3,2,1): rows[f"P3_{bb}bit"]=(S(r2,2,*b2,s,bb),b2)
    o={}
    for k,(x,c) in rows.items():
        d=st(x); d["cfg"]=c if c=="-" else f"{c[0]}, {c[1]:+.0f}°"
        d["delta"]=float(db(x).mean()-db(ref).mean())
        d["p"]=None if k in("P2",) else float(wilcoxon(db(x),db(ref)).pvalue)
        o[k]=d
    out[s*s]=o
json.dump(out,open("out/final_stats_th10.json","w"),indent=1,default=float)
for k,d in out[16384].items(): print(f"{k:10s} {d['cfg']:10s} cov {d['cov']:.3f}±{d['cov_sd']:.3f} snr {d['snr']:.1f}±{d['snr_sd']:.1f} se {d['se']:.2f}±{d['se_sd']:.2f} Δ {d['delta']:.2f} p {d['p']}")
for s in out: print(s, {k:(round(d['cov'],3),round(d['snr'],1),d['cfg']) for k,d in out[s].items()})
