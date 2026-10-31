\
\

import numpy as np, json
from config import *
from analysis import cascade, direct, SUBS, PT, SIG2
rt = dict(np.load("out/rt_paths.npz")); ue, valid = rt["ue"], rt["valid"]
hd_all = direct(2, rt, ue)
db = lambda x: np.maximum(10*np.log10(np.maximum(x, 1e-30)), -60)
out = {}
for yaw, Ns in ((30.0, [32, 96, 128]), (45.0, [64])):
    cn_all = cascade(2, rt, ue, "R1", yaw)[valid]; hd = hd_all[valid]; U = ue[valid]
    for s in Ns:
        cn = cn_all[:, SUBS[s]]
        res = {}
        th_opt = np.angle(hd)[:, None] - np.angle(cn)
        for b in (None, 3, 2, 1):
            q = (lambda t: t) if b is None else (lambda t, st=2*np.pi/2**b: np.round(t/st)*st)
            per_ue = PT*np.abs(hd + np.sum(cn*np.exp(1j*q(th_opt)), 1))**2/SIG2
            row = {"per_UE": (float(np.mean(db(per_ue) >= 10)), float(db(per_ue).mean()))}
            for K in (1, 4, 8, 16):
                edges = np.linspace(UE_Y[0]-0.5, UE_Y[-1]+0.5, K+1)
                best = np.full(len(U), -np.inf)
                for k in range(K):
                    yc = (edges[k]+edges[k+1])/2
                    i = np.argmin((U[:, 0]-40)**2 + (U[:, 1]-yc)**2)
                    th = q(th_opt[i])
                    snr = PT*np.abs(hd + cn @ np.exp(1j*th))**2/SIG2
                    best = np.maximum(best, snr)
                row[f"K{K}"] = (float(np.mean(db(best) >= 10)), float(db(best).mean()))
            res["cont" if b is None else f"{b}bit"] = row
        out[s*s] = dict(yaw=yaw, **res)
        print(s*s, yaw, {k: {kk: (round(100*v[0], 1), round(v[1], 1)) for kk, v in r.items()} for k, r in res.items()}, flush=True)
json.dump(out, open("out/codebook_R1.json", "w"), indent=1)
