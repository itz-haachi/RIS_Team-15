\
\
\

import os, json, numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import *
from analysis import cascade, direct, snr_from

os.makedirs("csv", exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e6e6"})
RES = ["c", 3, 2, 1]; BL = {"c": "continuous", 3: "3-bit", 2: "2-bit", 1: "1-bit"}
COL = {"c": "#2a78d6", 3: "#eb6834", 2: "#1baf7a", 1: "#eda100"}
TH = 10**(GAMMA_TH_DB/10); FLOOR = -60.0
db = lambda x: np.maximum(10*np.log10(np.maximum(x, 1e-30)), FLOOR)
cov = lambda x: float(np.mean(x >= TH)); msnr = lambda x: float(np.mean(db(x))); se = lambda x: float(np.mean(np.log2(1 + x)))

r1 = dict(np.load("out/snr_p1.npz")); r1b = dict(np.load("out/snr_p1b.npz")); r2 = dict(np.load("out/snr_p2.npz"))
valid = r2["valid"]
def S(r, ph, n, y, s, b="c"): return r[f"p{ph}_{n}_{y:+.0f}_{s}_{b}"][valid]
cfgs = [(n, y) for n in RIS_CAND for y in YAW_OFFSETS]
def best(r, ph, s, b="c"): return max(cfgs, key=lambda c: (cov(S(r, ph, *c, s, b)), msnr(S(r, ph, *c, s, b))))
Ns = [s*s for s in N_SIDES]
opt = {(s, b): best(r2, 2, s, b) for s in N_SIDES for b in RES}
p1c = {s: best(r1, 1, s) for s in N_SIDES}

SN = max(N_SIDES); rows = []
for n in RIS_CAND:
    for ph, r, lab in ((1, r1, "Phase 1 ideal"), ("1b", r1b, "Phase 1b blockage-aware"), (2, r2, "Phase 2 ray traced")):
        y = max(YAW_OFFSETS, key=lambda yy: (cov(S(r, ph, n, yy, SN)), msnr(S(r, ph, n, yy, SN))))
        rows.append(dict(site=n, model=lab, best_yaw_deg=y, coverage=cov(S(r, ph, n, y, SN)), mean_snr_db=msnr(S(r, ph, n, y, SN))))
pd.DataFrame(rows).to_csv("csv/fig04_sites_under_three_models.csv", index=False)

rows = []
for s in N_SIDES:
    n, y = opt[(s, "c")]
    for b in RES:
        x = S(r2, 2, n, y, s, b)
        rows.append(dict(N=s*s, placement=f"{n} {y:+.0f}", resolution=BL[b], coverage=cov(x), mean_snr_db=msnr(x), spectral_eff_bpshz=se(x)))
f8 = pd.DataFrame(rows); f8.to_csv("csv/fig08_phase3_metrics_vs_N.csv", index=False)
fig, axs = plt.subplots(1, 3, figsize=(13, 4))
for b in RES:
    d = f8[f8.resolution == BL[b]]
    for ax, k in zip(axs, ["coverage", "mean_snr_db", "spectral_eff_bpshz"]): ax.plot(range(4), d[k], "o-", color=COL[b], lw=2, label=BL[b])
for ax, lab in zip(axs, [f"coverage P(γ ≥ {GAMMA_TH_DB:.0f} dB)", "mean SNR (dB)", "spectral efficiency (bit/s/Hz)"]):
    ax.set_xticks(range(4)); ax.set_xticklabels(Ns); ax.set_xlabel("RIS elements N"); ax.set_ylabel(lab)
axs[0].legend(frameon=False, fontsize=8); fig.suptitle("Phase 3: effect of phase quantisation (RT optimum placement per N)")
plt.tight_layout(); plt.savefig("out/paper_fig08_phase3_metrics_vs_N.png", dpi=200)

theory = {b: -20*np.log10(np.sinc(1/2**b)) for b in (1, 2, 3)}
rows = []
for s in N_SIDES:
    for b in (3, 2, 1):
        rows.append(dict(N=s*s, resolution=BL[b], loss_db=float(f8[(f8.N == s*s) & (f8.resolution == "continuous")].mean_snr_db.iloc[0]
                                                               - f8[(f8.N == s*s) & (f8.resolution == BL[b])].mean_snr_db.iloc[0]), theory_db=theory[b]))
f9 = pd.DataFrame(rows); f9.to_csv("csv/fig09_quantization_loss_vs_theory.csv", index=False)
fig, ax = plt.subplots(figsize=(7, 4.2)); w = 0.25
for i, b in enumerate((3, 2, 1)):
    d = f9[f9.resolution == BL[b]]
    ax.bar(np.arange(4) + (i-1)*w, d.loss_db, w, color=COL[b], label=f"{BL[b]} (measured)")
    ax.plot(np.arange(4) + (i-1)*w, d.theory_db, "_", color="k", ms=14, mew=2, label="theory sinc² bound" if i == 0 else None)
ax.set_xticks(range(4)); ax.set_xticklabels(Ns); ax.set_xlabel("RIS elements N"); ax.set_ylabel("SNR loss vs continuous (dB)")
ax.legend(frameon=False, fontsize=8); ax.set_title("Quantisation degradation vs theory")
plt.tight_layout(); plt.savefig("out/paper_fig09_quantization_loss_vs_theory.png", dpi=200)

rows = []
for s in N_SIDES:
    for b in RES:
        rows.append(dict(N=s*s, resolution=BL[b], phase1_placement=f"{p1c[s][0]} {p1c[s][1]:+.0f}", rt_optimum=f"{opt[(s, b)][0]} {opt[(s, b)][1]:+.0f}",
                         coverage_lost=cov(S(r2, 2, *opt[(s, b)], s, b)) - cov(S(r2, 2, *p1c[s], s, b))))
f10 = pd.DataFrame(rows); f10.to_csv("csv/fig10_cost_of_ideal_site.csv", index=False)
fig, ax = plt.subplots(figsize=(7, 4.2)); w = 0.2
for i, b in enumerate(RES):
    ax.bar(np.arange(4) + (i-1.5)*w, f10[f10.resolution == BL[b]].coverage_lost, w, color=COL[b], label=BL[b])
ax.set_xticks(range(4)); ax.set_xticklabels(Ns); ax.set_xlabel("RIS elements N"); ax.set_ylabel("coverage lost by keeping Phase-1 placement")
ax.legend(frameon=False, fontsize=8); ax.set_title("Is the ideal-model placement still optimal?")
plt.tight_layout(); plt.savefig("out/paper_fig10_cost_of_ideal_site.png", dpi=200)

rt = dict(np.load("out/rt_paths.npz")); n2, y2 = opt[(SN, "c")]
cn = cascade(2, rt, rt["ue"], n2, y2)[valid]; hd = direct(2, rt, rt["ue"])[valid]
prng = np.random.default_rng(7); rows = []
for sd in [0, 5, 10, 20, 30]:
    for b in RES:
        v = [snr_from(hd, cn, None if b == "c" else b, np.deg2rad(sd), prng) for _ in range(1 if sd == 0 else 20)]
        c = [cov(x) for x in v]; m = [msnr(x) for x in v]
        rows.append(dict(sigma_deg=sd, resolution=BL[b], mean_snr_db=np.mean(m), mean_snr_sd=np.std(m), coverage=np.mean(c), coverage_sd=np.std(c)))
f7 = pd.DataFrame(rows); f7.to_csv("csv/fig07_phase_noise.csv", index=False)
fig, axs = plt.subplots(1, 2, figsize=(11, 4))
for b in RES:
    d = f7[f7.resolution == BL[b]]
    axs[0].errorbar(d.sigma_deg, d.mean_snr_db, d.mean_snr_sd, color=COL[b], marker="o", lw=2, capsize=3, label=BL[b])
    axs[1].errorbar(d.sigma_deg, d.coverage, d.coverage_sd, color=COL[b], marker="o", lw=2, capsize=3)
axs[0].set_ylabel("mean SNR (dB)"); axs[1].set_ylabel("coverage")
for ax in axs: ax.set_xlabel("phase noise σ (deg)")
axs[0].legend(frameon=False, fontsize=8); fig.suptitle(f"Phase noise ({n2} {y2:+.0f}°, N = {SN*SN}, 20 MC runs)")
plt.tight_layout(); plt.savefig("out/paper_fig07_phase_noise.png", dpi=200)

fs = json.load(open("out/final_stats_th10.json"))
pd.DataFrame([dict(N=int(N), row=k, **{kk: vv for kk, vv in v.items() if kk not in ("p",)}) for N, d in fs.items() for k, v in d.items()]) \
  .to_csv("csv/table05_and_fig06_stats_all_N.csv", index=False)
cb = {**json.load(open("out/codebook2_R1.json"))}
k12 = json.load(open("out/codebook_k12_R1.json"))
pd.DataFrame([dict(N=int(N), profile=k, coverage_pct=v[0], mean_snr_db=v[1]) for src in (k12, cb) for N, d in src.items() for k, v in d.items()]) \
  .drop_duplicates(["N", "profile"]).to_csv("csv/table06_codebook.csv", index=False)
bs = json.load(open("out/bs_compare.json"))
pd.json_normalize(bs, sep="_").T.to_csv("csv/table07_second_bs.csv", header=["value"])
print(f8.round(3).to_string()); print(f9.round(2).to_string()); print(f10.round(3).to_string())
print(f7[f7.resolution == "continuous"].round(3).to_string())
