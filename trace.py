\

import time, numpy as np, sionna.rt as rt, mitsuba as mi
from config import *

def setup(scene):
    scene.frequency = FC
    scene.tx_array = rt.PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
    scene.rx_array = rt.PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")

def extract(p, tx_pos, rx_pos):
    a = np.array(p.a[0]) + 1j*np.array(p.a[1])
    a = a[:, 0, :, 0, :]
    tau = np.array(p.tau)
    vert = np.array(p.vertices)
    inter = np.array(p.interactions)
    depth = (inter != 0).sum(0)
    R, T, P = a.shape
    k = min(KPATHS, P)
    order = np.argsort(-np.abs(a), axis=-1)[..., :k]
    take = lambda x: np.take_along_axis(x, order, -1)
    A, TAU, DEP = take(a), take(tau), take(depth)
    vf = np.take_along_axis(vert[0], order[..., None], 2)
    Dm = vert.shape[0]
    vl = np.zeros_like(vf)
    for d in range(1, Dm+1):
        vd = np.take_along_axis(vert[d-1], order[..., None], 2)
        vl = np.where((DEP == d)[..., None], vd, vl)
    rxp = np.broadcast_to(rx_pos[:, None, None, :], vf.shape)
    txp = np.broadcast_to(tx_pos[None, :, None, :], vf.shape)
    vf = np.where((DEP == 0)[..., None], rxp, vf)
    vl = np.where((DEP == 0)[..., None], txp, vl)
    L = TAU * C0
    A = np.where(np.abs(A) > 0, A, 0)
    return dict(a=A.astype(np.complex64), L=L.astype(np.float64), vf=vf.astype(np.float64), vl=vl.astype(np.float64), depth=DEP.astype(np.int8))

def run(scene, txs, rxs, tag):
    for n in list(scene.transmitters): scene.remove(n)
    for n in list(scene.receivers): scene.remove(n)
    for i, p in enumerate(txs): scene.add(rt.Transmitter(f"tx{i}", position=mi.Point3f(*map(float, p))))
    for i, p in enumerate(rxs): scene.add(rt.Receiver(f"rx{i}", position=mi.Point3f(*map(float, p))))
    t = time.time()
    p = rt.PathSolver()(scene, **TRACE)
    d = extract(p, np.array(txs, float), np.array(rxs, float))
    print(f"[{tag}] {len(txs)} tx x {len(rxs)} rx traced in {time.time()-t:.0f}s, max paths {np.array(p.tau).shape[-1]}", flush=True)
    return d

if __name__ == "__main__":
    sc = rt.load_scene("scene/scene.xml"); setup(sc)
    ue, valid, shp = ue_grid()
    ris_c = np.array([RIS_CAND[k][0] for k in RIS_CAND])
    out = {}
    d = run(sc, [BS], ue, "BS->UE");    out.update({f"d_{k}": v for k, v in d.items()})
    d = run(sc, [BS], ris_c, "BS->RIS"); out.update({f"g_{k}": v for k, v in d.items()})
    for i, name in enumerate(RIS_CAND):
        d = run(sc, [ris_c[i]], ue, f"{name}->UE")
        out.update({f"h{name}_{k}": v for k, v in d.items()})
        np.savez_compressed("out/rt_paths.npz", ue=ue, valid=valid, **out)
    print("done")
