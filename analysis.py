import numpy as np, json, time
from config import *

PT = 10**(PT_DBM/10)/1e3; SIG2 = 10**(NOISE_DBM/10)/1e3
BITS = [None, 1, 2, 3]
NMAX = max(N_SIDES)

def subset_idx(side):
    off = (NMAX - side)//2
    I, J = np.meshgrid(np.arange(off, off+side), np.arange(off, off+side))
    return (J*NMAX + I).ravel()
SUBS = {s: subset_idx(s) for s in N_SIDES}

def per_element(a, Lref_pt, c, P, nvec):
\

    dvc = Lref_pt - c
    rc = np.linalg.norm(dvc, axis=-1)
    cos = np.einsum('...k,k->...', dvc, nvec) / np.maximum(rc, 1e-9)
    F = np.clip(cos, 0, None)
    w = a * np.sqrt(G_EL*F)
    rn = np.linalg.norm(Lref_pt[..., None, :, :] - P[:, None, :], axis=-1)
    ph = np.exp(-1j*K0*(rn - rc[..., None, :])).astype(np.complex64)
    return np.einsum('...nk,...k->...n', ph, w.astype(np.complex64))

def snr_from(hd, cn, bits, pn_std=0.0, rng=None):

    th = np.angle(hd)[:, None] - np.angle(cn)
    if bits is not None:
        st = 2*np.pi/2**bits
        th = np.round(th/st)*st
    if pn_std > 0:
        th = th + rng.normal(0, pn_std, th.shape)
    y = hd + np.sum(cn*np.exp(1j*th), axis=1)
    return PT*np.abs(y)**2/SIG2

def threegpp_umi_gain(ue):
    d2 = np.linalg.norm(ue[:, :2]-BS[:2], axis=1); d3 = np.linalg.norm(ue-BS, axis=1); f = FC/1e9
    pl_los = 32.4 + 21*np.log10(d3) + 20*np.log10(f)
    pl_nl = np.maximum(pl_los, 22.4 + 35.3*np.log10(d3) + 21.3*np.log10(f) - 0.3*(UE_H-1.5))
    plos = np.where(d2 <= 18, 1.0, 18/d2 + np.exp(-d2/36)*(1-18/d2))
    return plos*10**(-pl_los/10) + (1-plos)*10**(-pl_nl/10)

def links(phase, rt, ue, name):

    c = np.array(RIS_CAND[name][0])
    if phase == 1:
        dg = np.linalg.norm(BS-c); ga = np.array([LAM/(4*np.pi*dg)*np.exp(-1j*K0*dg)]); gp = BS[None]
        dh = np.linalg.norm(ue-c, axis=1); ha = (LAM/(4*np.pi*dh)*np.exp(-1j*K0*dh))[:, None]; hp = ue[:, None, :]
        return ga, gp, ha, hp
    i = list(RIS_CAND).index(name)
    return rt["g_a"][i, 0], rt["g_vl"][i, 0], rt[f"h{name}_a"][:, 0], rt[f"h{name}_vf"][:, 0]

def direct(phase, rt, ue):
    if phase == 1: return np.zeros(len(ue), np.complex64)
    return rt["d_a"][:, 0].sum(-1)

def cascade(phase, rt, ue, name, yaw, chunk=24):
    P, nvec = ris_elements(name, NMAX, yaw)
    ga, gp, ha, hp = links(phase, rt, ue, name)
    c = np.array(RIS_CAND[name][0])
    g = per_element(ga, gp, c, P, nvec)
    out = np.empty((len(ue), len(P)), np.complex64)
    for s in range(0, len(ue), chunk):
        out[s:s+chunk] = per_element(ha[s:s+chunk], hp[s:s+chunk], c, P, nvec) * g[None]
    return out

if __name__ == "__main__":
    import sys
    phase = int(sys.argv[1])
    rt = dict(np.load(f"out/rt_paths{TAG}.npz"))
    ue, valid = rt["ue"], rt["valid"]
    U = len(ue); res = {}
    sites = sys.argv[2].split(",") if len(sys.argv) > 2 else list(RIS_CAND)
    if True:
        hd = direct(phase, rt, ue)
        res[f"p{phase}_direct"] = PT*np.abs(hd)**2/SIG2
        if phase == 1:
            res["p1_3gpp_expected"] = PT*threegpp_umi_gain(ue)/SIG2
            d3 = np.linalg.norm(ue-BS, axis=1); f = FC/1e9
            res["p1_3gpp_nlos"] = PT*10**(-np.maximum(32.4+21*np.log10(d3)+20*np.log10(f), 22.4+35.3*np.log10(d3)+21.3*np.log10(f))/10)/SIG2
        for name in sites:
            for yaw in YAW_OFFSETS:
                t = time.time()
                cn = cascade(phase, rt, ue, name, yaw)
                for s in N_SIDES:
                    sub = cn[:, SUBS[s]]
                    for b in (BITS if phase == 2 else [None]):
                        res[f"p{phase}_{name}_{yaw:+.0f}_{s}_{b or 'c'}"] = snr_from(hd, sub, b)
                print(f"phase {phase} {name} yaw{yaw:+.0f} {time.time()-t:.1f}s", flush=True)
        np.savez_compressed(f"out/snr_p{phase}{TAG}" + ("_" + "_".join(sites) if len(sys.argv) > 2 else "") + ".npz", ue=ue, valid=valid, **res)
    print("done")
