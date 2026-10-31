import json, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

d = json.load(open("out/final_stats_th10.json")); Ns = [1024, 4096, 9216, 16384]
series = [("P1", "Phase 1 ideal, empty world (prediction)", "#52514e", "--", "s"),
          ("P1b", "Phase 1b blockage-aware (prediction)", "#6250d6", "--", "D"),
          ("P2", "Phase 2 ray-traced, continuous", "#2a78d6", "-", "o"),
          ("P3_3bit", "Phase 3, 3-bit", "#eb6834", "-", "o"),
          ("P3_2bit", "Phase 3, 2-bit", "#1baf7a", "-", "o"),
          ("P3_1bit", "Phase 3, 1-bit", "#eda100", "-", "o")]
fig, axs = plt.subplots(1, 2, figsize=(11, 4.3))
for ax, k, lab in ((axs[0], "cov", "coverage probability P(γ ≥ 10 dB)"), (axs[1], "snr", "mean SNR over the area (dB)")):
    for key, l, c, ls, mk in series:
        ax.errorbar(Ns, [d[str(n)][key][k] for n in Ns], [d[str(n)][key][k+"_sd"] for n in Ns], color=c, ls=ls, marker=mk, ms=6, lw=2, capsize=3, label=l)
    ax.axhline(d["1024"]["no_ris"][k], color="#999", ls=":", lw=1.2, label="no RIS (ray-traced)")
    ax.set_xscale("log", base=2); ax.set_xticks(Ns, [f"{n:,}\n({int(np.sqrt(n))}×{int(np.sqrt(n))})" for n in Ns])
    ax.set_xlabel("number of RIS elements N"); ax.set_ylabel(lab); ax.grid(color="#eee")
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=4, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, 1.02))
plt.tight_layout(rect=(0, 0, 1, 0.88)); plt.savefig("out/final_fig_vs_N.png", dpi=220)

fig, ax = plt.subplots(figsize=(12.5, 4.8)); ax.axis("off"); ax.set_xlim(0, 12.4); ax.set_ylim(0, 4.6)
def box(x, y, w, h, title, lines, fc):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.12", fc=fc, ec="#333", lw=1.2))
    ax.text(x+w/2, y+h-0.28, title, ha="center", va="top", fontsize=10.5, weight="bold")
    ax.text(x+0.15, y+h-0.75, "\n".join(lines), ha="left", va="top", fontsize=8.6, linespacing=1.45)
box(0.1, 2.35, 2.7, 2.1, "Inputs", ["3D scene: buildings,", "ITU materials", "BS, UE grid, 6 RIS sites", "7 orientations, N = 32²…128²", "28 GHz, 100 MHz, γth = 10 dB"], "#f2f2f2")
box(3.2, 2.35, 2.8, 2.1, "Phase 1 / 1b: Ideal", ["Analytical channels:", "1: free space, no blockage", "1b: −40 dB if a link", "     crosses a building", "Continuous optimal θn"], "#e8eef9")
box(6.4, 2.35, 2.8, 2.1, "Phase 2: Site-specific", ["Sionna RT ray tracing:", "LOS, reflections (≤4),", "diffraction, blockage", "Per-element g_n, h_r,n", "Continuous optimal θn"], "#dce9fb")
box(9.6, 2.35, 2.7, 2.1, "Phase 3: Hardware", ["Quantise θn to", "1-, 2-, 3-bit states", "Optional phase noise", "σ = 0–30°"], "#fbe9dc")
for x0, x1 in ((2.8, 3.2), (6.0, 6.4), (9.2, 9.6)):
    ax.annotate("", xy=(x1, 3.4), xytext=(x0, 3.4), arrowprops=dict(arrowstyle="-|>", lw=1.6, color="#333"))
box(3.2, 0.1, 9.1, 1.8, "Outputs for every phase (Eqs. 1–3)", ["SNR and coverage maps · coverage probability and mean SNR per placement · optimal site/orientation per N",
    "Ideal choice re-evaluated in the ray-traced scene · coverage/SNR vs N · 1/2/3-bit and phase-noise loss"], "#f7f7f7")
for x in (4.6, 7.8, 10.95):
    ax.annotate("", xy=(x, 1.85), xytext=(x, 2.35), arrowprops=dict(arrowstyle="-|>", lw=1.2, color="#666"))
plt.savefig("out/final_fig1_workflow.png", dpi=220, bbox_inches="tight")
print("ok")
