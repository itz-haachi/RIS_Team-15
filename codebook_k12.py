import numpy as np, json
from config import *
from analysis import cascade, direct, SUBS, PT, SIG2
rt = dict(np.load("out/rt_paths.npz")); ue, valid = rt["ue"], rt["valid"]
hd = direct(2, rt, ue)[valid]; U = ue[valid]
db = lambda x: np.maximum(10*np.log10(np.maximum(x, 1e-30)), -60)
out = {}
for yaw, s in ((30.0, 128), (45.0, 64)):
    cn = cascade(2, rt, ue, "R1", yaw)[valid][:, SUBS[s]]
    th_opt = np.angle(hd)[:, None] - np.angle(cn)
    r = {}
    for nx, ny in ((1, 1), (1, 2), (1, 4)):
        xs = [40.0]; ys = np.linspace(11, 95, 2*ny+1)[1::2]
        for b in (None, 2, 1):
            q = (lambda t: t) if b is None else (lambda t, st=2*np.pi/2**b: np.round(t/st)*st)
            best = np.full(len(U), -np.inf)
            for xc in xs:
                for yc in ys:
                    i = np.argmin((U[:, 0]-xc)**2 + (U[:, 1]-yc)**2)
                    best = np.maximum(best, PT*np.abs(hd + cn @ np.exp(1j*q(th_opt[i])))**2/SIG2)
            r[f"K{nx*ny}_{'cont' if b is None else str(b)+'bit'}"] = (round(100*float(np.mean(db(best) >= 10)), 1), round(float(db(best).mean()), 1))
    out[s*s] = r; print(s*s, r, flush=True)
json.dump(out, open("out/codebook_k12_R1.json", "w"), indent=1)
