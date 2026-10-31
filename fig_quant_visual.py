import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
d = np.load("out/snr_p2.npz"); valid = d["valid"]
db = lambda x: np.maximum(10*np.log10(np.maximum(x, 1e-30)), -60)
cases = [("c", "Continuous\n(any angle)", None), ("3", "3-bit\n(8 angles)", 3), ("2", "2-bit\n(4 angles)", 2), ("1", "1-bit\n(2 angles)", 1)]
want = np.deg2rad(125)
fig = plt.figure(figsize=(14, 10.5))
for j, (r, lab, b) in enumerate(cases):
    ax = fig.add_subplot(2, 4, j+1, projection="polar")
    ax.set_yticks([]); ax.set_xticks(np.deg2rad([0, 90, 180, 270])); ax.set_xticklabels(["0°", "90°", "180°", "270°"], fontsize=9)
    ax.set_ylim(0, 1.15); ax.grid(alpha=.3)
    if b is None:
        ax.plot(np.linspace(0, 2*np.pi, 300), np.ones(300), color="#2e9e5b", lw=5)
        ax.annotate("", xy=(want, 1), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color="#1f6fd1", lw=2.5))
        sub = "uses exactly 125°"
    else:
        st = 2*np.pi/2**b; states = np.arange(2**b)*st; got = np.round(want/st)*st % (2*np.pi)
        ax.plot(states, np.ones_like(states), "o", color="#2e9e5b", ms=13, mec="k")
        ax.annotate("", xy=(want, 1), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color="#1f6fd1", lw=2, ls="--"))
        ax.annotate("", xy=(got, 1), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color="#d62728", lw=2.5))
        sub = f"wants 125° → gets {np.rad2deg(got):.0f}°\n(error {abs(np.rad2deg(got)-125):.0f}°)"
    ax.set_title(f"{lab}\n{sub}", fontsize=11, fontweight="bold", pad=14)
for j, (r, lab, b) in enumerate(cases):
    s = d[f"p2_R1_+30_128_{r}"].astype(float); cov = 100*np.mean(s[valid] >= 10); m = db(s[valid]).mean()
    g = db(s); g[~valid] = np.nan
    ax = fig.add_subplot(2, 4, j+5)
    im = ax.imshow(g.reshape(85, 19), origin="lower", extent=[30.5, 49.5, 10.5, 95.5], cmap="viridis", vmin=0, vmax=45)
    ax.add_patch(plt.Rectangle((37, 55), 6, 3, color="k")); ax.set_xticks([]); ax.set_yticks([])
    ref = db(d["p2_R1_+30_128_c"][valid]).mean()
    ax.set_title(f"{lab.splitlines()[0]}\n{cov:.1f}% covered\nmean SNR {m:.1f} dB", fontsize=11, fontweight="bold")
    ax.set_xlabel("baseline" if r == "c" else f"loss: {ref-m:.2f} dB", fontsize=13, fontweight="bold", color="#555" if r == "c" else "#d62728")
cax = fig.add_axes([0.91, 0.08, 0.015, 0.33]); fig.colorbar(im, cax=cax, label="SNR (dB)")
fig.text(0.02, 0.73, "What each RIS\nelement can do", fontsize=12, fontweight="bold", rotation=90, va="center")
fig.text(0.02, 0.28, "Result in the street\n(R1, 16,384 elements)", fontsize=12, fontweight="bold", rotation=90, va="center")
fig.text(0.5, 0.965, "Phase quantization: blue dashed = angle the element should use,  red = nearest angle the hardware allows",
         ha="center", fontsize=12)
plt.subplots_adjust(left=0.07, right=0.88, top=0.88, bottom=0.03, hspace=0.45, wspace=0.35)
plt.savefig("out/fig_quant_visual.png", dpi=150)
