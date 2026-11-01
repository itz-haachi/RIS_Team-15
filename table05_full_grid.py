import numpy as np, json, csv
from config import *
r1=dict(np.load("out/snr_p1.npz")); r1b=dict(np.load("out/snr_p1b.npz")); r2=dict(np.load("out/snr_p2.npz"))
v=r2["valid"]; TH=10.0
db=lambda x: np.maximum(10*np.log10(np.maximum(x,1e-30)),-60)
S=lambda r,p,n,y,s,b="c": r[f"p{p}_{n}_{y:+.0f}_{s}_{b}"][v]
cfg=[(n,y) for n in RIS_CAND for y in YAW_OFFSETS]
def best(r,p,s): return max(cfg,key=lambda c:(np.mean(db(S(r,p,*c,s))>=TH),db(S(r,p,*c,s)).mean()))
sd=json.load(open("out/final_stats_th10.json"))
rows=[]
for s in N_SIDES:
    b1=best(r1,1,s); b1b=best(r1b,"1b",s); b2=best(r2,2,s)
    cases={"no_ris":(r2["p2_direct"][v],"-"),"P1":(S(r1,1,*b1,s),b1),"P1b":(S(r1b,"1b",*b1b,s),b1b),
           "P2_at_P1":(S(r2,2,*b1,s),b1),"P2_at_P1b":(S(r2,2,*b1b,s),b1b),"P2":(S(r2,2,*b2,s),b2)}
    for bb in (3,2,1): cases[f"P3_{bb}bit"]=(S(r2,2,*b2,s,bb),b2)
    for k,(x,c) in cases.items():
        e=sd[str(s*s)][k]
        rows.append([s*s,k,c if c=="-" else f"{c[0]} {c[1]:+.0f}", f"{100*np.mean(db(x)>=TH):.1f}", f"{db(x).mean():.1f}",
                     f"{np.mean(np.log2(1+x)):.2f}", f"{100*e['cov_sd']:.1f}", f"{e['snr_sd']:.1f}", f"{e['se_sd']:.2f}"])
with open("csv/table05_full_grid.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["N","case","placement","coverage_pct","mean_snr_db","spectral_eff_bps_hz","coverage_sd_pct","mean_snr_sd_db","spectral_eff_sd"]); w.writerows(rows)
for r in rows:
    if r[0]==16384: print(*r)
