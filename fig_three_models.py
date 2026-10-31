import pandas as pd, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
rk = pd.read_csv("out/three_model_per_candidate_N16384.csv")
M = [("P1", "Phase 1: empty world (free space)", "#9a9a9a"), ("P1b", "Phase 1b: blockage-aware (flat 40 dB)", "#eb6834"), ("P2", "Phase 2: ray-traced (Sionna RT)", "#2a78d6")]
x = np.arange(len(rk)); w = 0.26
fig, axs = plt.subplots(1, 2, figsize=(12, 4.3))
for ax, key, lab, lo in ((axs[0], "cov", "coverage P(SNR ≥ 10 dB) (%)", 0), (axs[1], "snr", "mean SNR over side street (dB)", -60)):
    for i, (m, l, c) in enumerate(M):
        vals = rk[f"{m}_{key}"].values
        ax.bar(x + (i-1)*w, vals - lo, w*0.92, bottom=lo, color=c, label=l)
    ax.set_xticks(x, rk.RIS); ax.set_ylabel(lab); ax.grid(axis="y", color="#eee"); ax.set_axisbelow(True)
    ax.axvspan(-0.5, 0.5, color="#2a78d6", alpha=.06)
axs[1].set_ylim(-60, 35); axs[0].set_ylim(0, 105)
h, l = axs[0].get_legend_handles_labels(); fig.legend(h, l, frameon=False, fontsize=9, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.95))
axs[0].text(0, 101, "RT best", ha="center", fontsize=8, color="#2a78d6")
fig.suptitle("Each RIS candidate (best orientation, N = 16,384) under three propagation models", y=1.0)
plt.tight_layout(rect=(0, 0, 1, 0.88)); plt.savefig("out/fig_three_models_per_candidate.png", dpi=200, bbox_inches="tight")
