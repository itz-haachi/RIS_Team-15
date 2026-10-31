import numpy as np, sionna.rt as rt
from config import *
from trace import setup, run
assert TAG, "set RIS_TAG"
sc = rt.load_scene("scene/scene.xml"); setup(sc)
ue, valid, _ = ue_grid()
ris_c = np.array([RIS_CAND[k][0] for k in RIS_CAND])
base = dict(np.load("out/rt_paths.npz"))
out = {k: v for k, v in base.items() if k.startswith("h") or k in ("ue", "valid")}
d = run(sc, [BS], ue, "BS2->UE");  out.update({f"d_{k}": v for k, v in d.items()})
d = run(sc, [BS], ris_c, "BS2->RIS"); out.update({f"g_{k}": v for k, v in d.items()})
np.savez_compressed(f"out/rt_paths{TAG}.npz", **out); print("done", BS)
