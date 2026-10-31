\
\
\

import json, numpy as np, time
from config import *
from analysis import per_element, snr_from, SUBS, NMAX, PT, SIG2

BLK_DB = 40.0; BLK = 10**(-BLK_DB/20)
G = json.load(open("scene/geometry.json"))
BOXES = [(np.array([b[0], b[2], 0.0]), np.array([b[1], b[3], b[4]])) for b in G["buildings"].values()]

def blocked(p0, P1):

    P1 = np.atleast_2d(P1); d = P1 - p0; res = np.zeros(len(P1), bool)
    for lo, hi in BOXES:
        with np.errstate(divide="ignore", invalid="ignore"):
            t1 = (lo - p0)/d; t2 = (hi - p0)/d
        par = np.abs(d) < 1e-12; inside = (p0 >= lo) & (p0 <= hi)
        tmin = np.where(par, np.where(inside, -np.inf, np.inf), np.minimum(t1, t2))
        tmax = np.where(par, np.where(inside, np.inf, -np.inf), np.maximum(t1, t2))
        te, tl = tmin.max(1), tmax.min(1)
        res |= (np.maximum(te, 0) <= np.minimum(tl, 1)) & (te <= tl)
    return res

def fs(d): return LAM/(4*np.pi*d)*np.exp(-1j*K0*d)

if __name__ == "__main__":
    ue, valid, _ = ue_grid()
    dd = np.linalg.norm(ue - BS, axis=1)
    bd = blocked(BS, ue)
    hd = (fs(dd)*np.where(bd, BLK, 1.0)).astype(np.complex64)
    res = {"p1b_direct": PT*np.abs(hd)**2/SIG2, "p1b_direct_blocked": bd}
    info = {}
    for name in RIS_CAND:
        c = np.array(RIS_CAND[name][0])
        bg = bool(blocked(BS, c[None])[0]); bh = blocked(c, ue)
        info[name] = dict(BS_RIS_blocked=bg, frac_UE_links_blocked=float(bh[valid].mean()))
        dg = np.linalg.norm(BS - c)
        ga = np.array([fs(dg)*(BLK if bg else 1.0)]); gp = BS[None]
        dh = np.linalg.norm(ue - c, axis=1)
        ha = (fs(dh)*np.where(bh, BLK, 1.0))[:, None]; hp = ue[:, None, :]
        for yaw in YAW_OFFSETS:
            t = time.time()
            P, nvec = ris_elements(name, NMAX, yaw)
            g = per_element(ga, gp, c, P, nvec)
            cn = np.empty((len(ue), len(P)), np.complex64)
            for s in range(0, len(ue), 24):
                cn[s:s+24] = per_element(ha[s:s+24], hp[s:s+24], c, P, nvec)*g[None]
            for sd in N_SIDES:
                res[f"p1b_{name}_{yaw:+.0f}_{sd}_c"] = snr_from(hd, cn[:, SUBS[sd]], None)
            print(f"1b {name} yaw{yaw:+.0f} {time.time()-t:.1f}s", flush=True)
    np.savez_compressed(f"out/snr_p1b{TAG}.npz", ue=ue, valid=valid, **res)
    json.dump(info, open(f"out/p1b_blockage_info{TAG}.json", "w"), indent=1)
    print(json.dumps(info, indent=1))
