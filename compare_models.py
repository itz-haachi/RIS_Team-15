import numpy as np, json, pandas as pd
from config import *
r = {"1": dict(np.load("out/snr_p1.npz")), "1b": dict(np.load("out/snr_p1b.npz")), "2": dict(np.load("out/snr_p2.npz"))}
v = r["2"]["valid"]; TH = 10**(10/10)
db = lambda x: np.maximum(10*np.log10(np.maximum(x, 1e-30)), -60)
S = lambda m, n, y, s: r[m][f"p{m}_{n}_{y:+.0f}_{s}_c"][v]
cfgs = [(n, y) for n in RIS_CAND for y in YAW_OFFSETS]
score = lambda m, n, y, s: (np.mean(S(m, n, y, s) >= TH), db(S(m, n, y, s)).mean())
out = {"noRIS": {m: dict(cov=float(np.mean(r[m][f"p{m}_direct"][v] >= TH)), snr=float(db(r[m][f"p{m}_direct"][v]).mean())) for m in r}}
rows = []
for s in N_SIDES:
    for m in ("1", "1b"):
        n, y = max(cfgs, key=lambda c: score(m, *c, s))
        pc, ps = score(m, n, y, s); rc, rs = score("2", n, y, s)
        rows.append(dict(N=s*s, model={"1": "Phase 1 empty-world", "1b": "Phase 1b blockage-aware"}[m], choice=f"{n} {y:+.0f}",
                         predicted_cov=round(100*pc, 1), predicted_snr=round(ps, 1), achieved_cov_RT=round(100*rc, 1), achieved_snr_RT=round(rs, 1)))
    n, y = max(cfgs, key=lambda c: score("2", *c, s)); rc, rs = score("2", n, y, s)
    rows.append(dict(N=s*s, model="Phase 2 ray-traced optimum", choice=f"{n} {y:+.0f}", predicted_cov=None, predicted_snr=None,
                     achieved_cov_RT=round(100*rc, 1), achieved_snr_RT=round(rs, 1)))
df = pd.DataFrame(rows); df.to_csv("out/three_model_comparison_th10.csv", index=False)
print(json.dumps(out, indent=1)); print(df.to_string(index=False))

rk = []
for n in RIS_CAND:
    d = {"RIS": n}
    for m in ("1", "1b", "2"):
        yb = max(YAW_OFFSETS, key=lambda y: score(m, n, y, 128))
        c, sn = score(m, n, yb, 128); d[f"P{m}_cov"] = round(100*c, 1); d[f"P{m}_snr"] = round(sn, 1)
    rk.append(d)
rk = pd.DataFrame(rk); rk.to_csv("out/three_model_per_candidate_N16384.csv", index=False); print(rk.to_string(index=False))

from scipy.stats import spearmanr
sp = [dict(model={"1": "Phase 1 ideal", "1b": "Phase 1b blockage-aware"}[m], reference="Phase 2 ray traced",
           metric="mean SNR at best orientation, N=16384", n_sites=len(rk),
           spearman_rho=round(spearmanr(rk[f"P{m}_snr"], rk["P2_snr"]).correlation, 2)) for m in ("1", "1b")]
import os; os.makedirs("csv", exist_ok=True)
pd.DataFrame(sp).to_csv("csv/table_spearman.csv", index=False); print(pd.DataFrame(sp).to_string(index=False))
