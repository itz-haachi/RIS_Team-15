import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from config import *
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
df = pd.read_csv("partB/noRIS_baseline_per_UE.csv"); _, _, SHP = ue_grid()
g = np.where(df.valid, np.maximum(df.SNR_dB, -60), np.nan).reshape(SHP)
gl = np.where(df.valid, np.maximum(df.SNR_localmean_dB, -60), np.nan).reshape(SHP)
fig, axs = plt.subplots(1, 3, figsize=(11, 6.2), gridspec_kw=dict(width_ratios=[1, 1, 1.6], wspace=0.45))
ext = [30.5, 49.5, 10.5, 95.5]
for ax, G, t in ((axs[0], g, "Narrowband SNR\n(coherent sum, Eq. 1)"), (axs[1], gl, "Local-mean SNR\n(power sum of paths)")):
    im = ax.imshow(G, origin="lower", extent=ext, cmap="Blues", vmin=-40, vmax=40)
    ax.contour(UE_X, UE_Y, np.nan_to_num(G, nan=-99), levels=[10], colors="#e34948", linewidths=1)
    ax.set_title(t, fontsize=9); ax.set_xlabel("x (m)")
axs[0].set_ylabel("y (m), distance up the side street")
fig.colorbar(im, ax=axs[:2], orientation="horizontal", shrink=.9, pad=.14, label="SNR (dB); red contour = 10 dB")
v = df[df.valid]; rows = v.groupby("y_m")
axs[2].plot(rows.SNR_localmean_dB.apply(lambda s: np.maximum(s, -60).mean()).index, rows.SNR_localmean_dB.apply(lambda s: np.maximum(s, -60).mean()).values, color="#1baf7a", lw=2, label="local mean")
axs[2].plot(rows.SNR_dB.apply(lambda s: np.maximum(s, -60).mean()).index, rows.SNR_dB.apply(lambda s: np.maximum(s, -60).mean()).values, color="#2a78d6", lw=2, label="narrowband (coherent)")
axs[2].axhline(10, color="#e34948", lw=1, ls="--"); axs[2].text(92, 11, "10 dB", color="#e34948", fontsize=8, ha="right")
axs[2].axvspan(54.5, 58.5, color="#ccc", alpha=.5); axs[2].text(56.5, 30, "kiosk", ha="center", fontsize=8)
axs[2].set_xlabel("y (m), distance up the side street"); axs[2].set_ylabel("mean SNR across street width (dB)")
axs[2].set_title("SNR falls ~55 dB from the junction to the dead zone", fontsize=9); axs[2].legend(frameon=False, fontsize=8); axs[2].grid(color="#eee")
fig.suptitle("No-RIS baseline over the side street (Sionna RT, 28 GHz, 40 dBm EIRP, noise -87 dBm)", fontsize=10)
plt.savefig("partB/fig_noRIS_sidestreet.png", dpi=200, bbox_inches="tight")
