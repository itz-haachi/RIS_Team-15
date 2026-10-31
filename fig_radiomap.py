import json, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from config import *
pg = np.load("out/radiomap_noRIS_pathgain_db.npy"); snr = pg + PT_DBM - NOISE_DBM
g = json.load(open("scene/geometry.json"))
fig, ax = plt.subplots(figsize=(8, 5.6))
snr = np.where(pg <= -290, np.nan, snr)
im = ax.imshow(snr, origin="lower", extent=[-100,120,-60,100], cmap="Blues", vmin=-40, vmax=60)
for n,(x0,x1,y0,y1,h,m) in g["buildings"].items():
    ax.add_patch(plt.Rectangle((x0,y0),x1-x0,y1-y0,fc="#d9d7d0",ec="#888",lw=.5))
ax.contour(np.linspace(-99.5,119.5,snr.shape[1]), np.linspace(-59.5,99.5,snr.shape[0]), np.nan_to_num(snr,nan=-99), levels=[GAMMA_TH_DB], colors="#e34948", linewidths=.8)
ax.plot(*BS[:2],"^",c="#e34948",ms=9); ax.text(BS[0],BS[1]+3,"BS",ha="center",fontsize=8)
for k,(c,y) in RIS_CAND.items(): ax.plot(c[0],c[1],"s",c="#eb6834",ms=5); ax.text(c[0]+1.5,c[1]+1.5,k,fontsize=7)
fig.colorbar(im, label="SNR without RIS (dB)"); ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
ax.set_title(f"Ray-traced SNR map without RIS (Sionna RT, 28 GHz, 1.5 m); red = {GAMMA_TH_DB:.0f} dB contour", fontsize=9)
plt.tight_layout(); plt.savefig("out/fig_radiomap_noRIS.png", dpi=200)
