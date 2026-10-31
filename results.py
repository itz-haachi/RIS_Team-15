import numpy as np, json, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
from config import *
from analysis import cascade, direct, snr_from, SUBS, PT, SIG2
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e6e6"})
COL = {"c": "#2a78d6", 3: "#eb6834", 2: "#1baf7a", 1: "#eda100", "p1": "#52514e"}
BL = {"c": "continuous", 3: "3-bit", 2: "2-bit", 1: "1-bit"}
TH = 10**(GAMMA_TH_DB/10); FLOOR = -60.0

r1 = dict(np.load("out/snr_p1.npz")); r2 = dict(np.load("out/snr_p2.npz"))
valid = r2["valid"]; ue = r2["ue"]; _, _, SHP = ue_grid()
db = lambda x: np.maximum(10*np.log10(np.maximum(x, 1e-30)), FLOOR)
def S(phase, name, yaw, s, b="c"): return (r1 if phase == 1 else r2)[f"p{phase}_{name}_{yaw:+.0f}_{s}_{b}"][valid]
cov = lambda x: float(np.mean(x >= TH))
msnr = lambda x: float(np.mean(db(x)))

cfgs = [(n, y) for n in RIS_CAND for y in YAW_OFFSETS]
sweep = {}
for ph in (1, 2):
    for s in N_SIDES:
        for b in (["c"] if ph == 1 else ["c", 1, 2, 3]):
            sweep[(ph, s, b)] = {(n, y): (cov(S(ph, n, y, s, b)), msnr(S(ph, n, y, s, b))) for n, y in cfgs}
best = {k: max(v, key=lambda c: v[c]) for k, v in sweep.items()}

out = {"params": dict(fc_GHz=FC/1e9, Pt_dBm_EIRP=PT_DBM, noise_dBm=round(NOISE_DBM, 2), gamma_th_dB=GAMMA_TH_DB,
                      n_ue=int(valid.sum()), N=[s*s for s in N_SIDES])}
out["no_ris"] = {f"phase{ph}": dict(coverage=cov(r[f"p{ph}_direct"][valid]), mean_snr_db=msnr(r[f"p{ph}_direct"][valid]))
                 for ph, r in ((1, r1), (2, r2))}
out["optimum"] = {f"P{k[0]}_N{k[1]**2}_{BL[k[2]]}": dict(ris=v[0], yaw_offset=v[1], coverage=sweep[k][v][0], mean_snr_db=sweep[k][v][1])
                  for k, v in best.items()}

out["ideal_choice_in_RT"] = {}
for s in N_SIDES:
    n1, y1 = best[(1, s, "c")]; n2, y2 = best[(2, s, "c")]
    out["ideal_choice_in_RT"][f"N{s*s}"] = dict(phase1_choice=f"{n1} {y1:+.0f}", predicted_cov=sweep[(1, s, "c")][(n1, y1)][0],
        realized_cov_RT=sweep[(2, s, "c")][(n1, y1)][0], RT_optimum=f"{n2} {y2:+.0f}", RT_opt_cov=sweep[(2, s, "c")][(n2, y2)][0],
        realized_snr_RT=sweep[(2, s, "c")][(n1, y1)][1], RT_opt_snr=sweep[(2, s, "c")][(n2, y2)][1])

SN = max(N_SIDES)
out["per_candidate_N16384"] = {n: {f"P{ph}": dict(zip(["best_yaw", "coverage", "mean_snr_db"],
        (lambda yy: (yy, *sweep[(ph, SN, "c")][(n, yy)]))(max(YAW_OFFSETS, key=lambda y: sweep[(ph, SN, "c")][(n, y)]))))
        for ph in (1, 2)} for n in RIS_CAND}

rng = np.random.default_rng(2026); R, K = 20, 200
drops = [rng.choice(valid.sum(), K, replace=False) for _ in range(R)]
def stats(x):
    c = [cov(x[d]) for d in drops]; m = [msnr(x[d]) for d in drops]; se = [float(np.mean(np.log2(1+x[d]))) for d in drops]
    return dict(cov=float(np.mean(c)), cov_sd=float(np.std(c, ddof=1)), snr=float(np.mean(m)), snr_sd=float(np.std(m, ddof=1)),
                se=float(np.mean(se)), se_sd=float(np.std(se, ddof=1)), cov_all=cov(x), snr_all=msnr(x), median_snr=float(np.median(db(x))))
def pval(x, ref):
    try: return float(wilcoxon(db(x), db(ref)).pvalue)
    except ValueError: return float("nan")

n1, y1 = best[(1, SN, "c")]; n2, y2 = best[(2, SN, "c")]
ref = S(2, n2, y2, SN)
rows = [("No RIS (ray-traced direct link only)", r2["p2_direct"][valid], "-"),
        (f"Phase 1 - ideal, continuous ({n1}, {y1:+.0f} deg)", S(1, n1, y1, SN), "-"),
        (f"Phase 2 - ray-traced, continuous, Phase-1 placement ({n1}, {y1:+.0f} deg)", S(2, n1, y1, SN), None),
        (f"Phase 2 - ray-traced, continuous, re-optimised ({n2}, {y2:+.0f} deg)", ref, "ref"),
        (f"Phase 3 - 3-bit ({n2}, {y2:+.0f} deg)", S(2, n2, y2, SN, 3), None),
        (f"Phase 3 - 2-bit ({n2}, {y2:+.0f} deg)", S(2, n2, y2, SN, 2), None),
        (f"Phase 3 - 1-bit ({n2}, {y2:+.0f} deg)", S(2, n2, y2, SN, 1), None)]
out["table4"] = []
for lab, x, p in rows:
    d = stats(x); d["label"] = lab; d["N"] = SN*SN
    d["p_vs_P2cont"] = p if p is not None else pval(x, ref)
    d["delta_snr_vs_P2cont_db"] = float(np.mean(db(x)) - np.mean(db(ref)))
    out["table4"].append(d)
out["table4_phase1_vs_phase2_p"] = pval(S(1, n1, y1, SN), ref)

nsweep = {}
for s in N_SIDES:
    nb, yb = best[(2, s, "c")]
    nsweep[s*s] = {"P1": stats(S(1, *best[(1, s, "c")], s)), "P1_placement_in_RT": stats(S(2, *best[(1, s, "c")], s))}
    for b in ["c", 3, 2, 1]:
        nsweep[s*s][f"P2_{BL[b]}"] = stats(S(2, nb, yb, s, b))
        nsweep[s*s][f"P2_{BL[b]}_reopt"] = dict(cfg=f"{best[(2,s,b)][0]} {best[(2,s,b)][1]:+.0f}", **stats(S(2, *best[(2, s, b)], s, b)))
out["N_sweep"] = nsweep

rt = dict(np.load("out/rt_paths.npz"))
cn = cascade(2, rt, ue, n2, y2)[valid]; hd = direct(2, rt, ue)[valid]
pn = {}
prng = np.random.default_rng(7)
for sd in [0, 5, 10, 20, 30]:
    for b in ["c", 3, 2, 1]:
        vals = [snr_from(hd, cn, None if b == "c" else b, np.deg2rad(sd), prng) for _ in range(1 if sd == 0 else 5)]
        pn[f"{BL[b]}_{sd}"] = dict(sigma_deg=sd, cov=float(np.mean([cov(v) for v in vals])), cov_sd=float(np.std([cov(v) for v in vals])),
                                   snr=float(np.mean([msnr(v) for v in vals])), snr_sd=float(np.std([msnr(v) for v in vals])))
out["phase_noise"] = pn

out["theory_quant_loss_db"] = {b: float(-20*np.log10(np.sinc(1/2**b))) for b in (1, 2, 3)}
json.dump(out, open(f"out/results_th{GAMMA_TH_DB:.0f}.json", "w"), indent=1, default=str)

def grid(x):
    g = np.full(valid.shape, np.nan); g[valid] = db(x); return g.reshape(SHP)
ext = [UE_X[0]-.5, UE_X[-1]+.5, UE_Y[0]-.5, UE_Y[-1]+.5]
maps = [("No RIS\n(ray-traced)", r2["p2_direct"]),
        (f"Phase 1 ideal\n{n1} {y1:+.0f}°", r1[f"p1_{n1}_{y1:+.0f}_{SN}_c"]),
        (f"Phase 2 RT, Phase-1\nplacement {n1} {y1:+.0f}°", r2[f"p2_{n1}_{y1:+.0f}_{SN}_c"]),
        (f"Phase 2 RT, re-opt.\n{n2} {y2:+.0f}°", r2[f"p2_{n2}_{y2:+.0f}_{SN}_c"]),
        (f"Phase 3 2-bit\n{n2} {y2:+.0f}°", r2[f"p2_{n2}_{y2:+.0f}_{SN}_2"]),
        (f"Phase 3 1-bit\n{n2} {y2:+.0f}°", r2[f"p2_{n2}_{y2:+.0f}_{SN}_1"])]
fig, axs = plt.subplots(1, 6, figsize=(13, 6.3), sharey=True)
for ax, (t, x) in zip(axs, maps):
    g = np.full(valid.shape, np.nan); g[:] = db(x); g[~valid] = np.nan; g = g.reshape(SHP)
    im = ax.imshow(g, origin="lower", extent=ext, cmap="Blues", vmin=-40, vmax=40, aspect="equal")
    ax.contour(np.arange(31, 50), np.arange(11, 96), g, levels=[GAMMA_TH_DB], colors="#e34948", linewidths=1)
    ax.set_title(t, fontsize=9); ax.set_xlabel("x (m)"); ax.grid(False)
    ax.text(40, 92, f"cov {cov(x[valid]):.2f}", ha="center", fontsize=8, bbox=dict(fc="white", ec="none", alpha=.8))
axs[0].set_ylabel("y (m)")
cb = fig.colorbar(im, ax=axs, shrink=.8, pad=.01); cb.set_label(f"SNR (dB); red contour = {GAMMA_TH_DB:.0f} dB threshold")
fig.suptitle(f"SNR maps over the side street (N = {SN*SN} elements, 28 GHz)", y=.98)
plt.savefig(f"out/th{GAMMA_TH_DB:.0f}_fig_snr_maps.png", dpi=200, bbox_inches="tight")

fig, axs = plt.subplots(1, 2, figsize=(11, 3.8))
for ax, ph in zip(axs, (1, 2)):
    M = np.array([[sweep[(ph, SN, "c")][(n, y)][1] for y in YAW_OFFSETS] for n in RIS_CAND])
    im = ax.imshow(M, cmap="Blues", vmin=-40, vmax=40, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i,j]:.0f}", ha="center", va="center", fontsize=8, color="white" if M[i, j] > 10 else "#0b0b0b")
    bi = list(RIS_CAND).index(best[(ph, SN, "c")][0]); bj = YAW_OFFSETS.index(best[(ph, SN, "c")][1])
    ax.add_patch(plt.Rectangle((bj-.5, bi-.5), 1, 1, fill=False, ec="#e34948", lw=2))
    ax.set_xticks(range(len(YAW_OFFSETS)), [f"{y:+.0f}°" for y in YAW_OFFSETS]); ax.set_yticks(range(len(RIS_CAND)), list(RIS_CAND))
    ax.set_xlabel("orientation offset from facade normal"); ax.grid(False)
    ax.set_title(["Phase 1: ideal (free-space RIS links)", "Phase 2: site-specific ray tracing"][ph-1], fontsize=10)
fig.colorbar(im, ax=axs, label="mean SNR over area (dB)")
plt.savefig(f"out/th{GAMMA_TH_DB:.0f}_fig_placement_heatmap.png", dpi=200, bbox_inches="tight")

Ns = [s*s for s in N_SIDES]
fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
for ax, key, lab in ((axs[0], "cov", f"coverage probability  P(γ ≥ {GAMMA_TH_DB:.0f} dB)"), (axs[1], "snr", "mean SNR over area (dB)")):
    ax.errorbar(Ns, [nsweep[n]["P1"][key] for n in Ns], [nsweep[n]["P1"][key+"_sd"] for n in Ns], color=COL["p1"], ls="--", marker="o", ms=6, lw=2, capsize=3, label="Phase 1 ideal (continuous)")
    for b in ["c", 3, 2, 1]:
        ax.errorbar(Ns, [nsweep[n][f"P2_{BL[b]}"][key] for n in Ns], [nsweep[n][f"P2_{BL[b]}"][key+"_sd"] for n in Ns], color=COL[b], marker="o", ms=6, lw=2, capsize=3,
                    label=f"{'Phase 2 RT' if b=='c' else 'Phase 3'} {BL[b]}")
    ax.axhline(stats(r2["p2_direct"][valid])[key], color="#999", lw=1, ls=":", label="no RIS (RT)")
    ax.set_xscale("log", base=2); ax.set_xticks(Ns, [f"{n}\n({int(np.sqrt(n))}²)" for n in Ns]); ax.set_xlabel("RIS elements N"); ax.set_ylabel(lab)
axs[0].legend(fontsize=8, frameon=False)
fig.suptitle("Coverage and SNR vs element count (error bars: ±1 SD over 20 random drops of 200 UEs)", fontsize=10)
plt.tight_layout(); plt.savefig(f"out/th{GAMMA_TH_DB:.0f}_fig2_vs_N.png", dpi=200)

fig, ax = plt.subplots(figsize=(7, 4.3))
cdfs = [("No RIS (RT)", r2["p2_direct"][valid], "#999", ":"), ("Phase 1 ideal", S(1, n1, y1, SN), COL["p1"], "--"),
        (f"Phase 2 RT, Phase-1 placement ({n1})", S(2, n1, y1, SN), "#6250d6", "-."), (f"Phase 2 RT, re-optimised ({n2})", ref, COL["c"], "-"),
        ("Phase 3 3-bit", S(2, n2, y2, SN, 3), COL[3], "-"), ("Phase 3 2-bit", S(2, n2, y2, SN, 2), COL[2], "-"), ("Phase 3 1-bit", S(2, n2, y2, SN, 1), COL[1], "-")]
for lab, x, c, ls in cdfs:
    v = np.sort(db(x)); ax.plot(v, np.arange(1, len(v)+1)/len(v), color=c, ls=ls, lw=2, label=lab)
ax.axvline(GAMMA_TH_DB, color="#e34948", lw=1); ax.set_xlabel("SNR (dB)"); ax.set_ylabel("CDF over UE locations"); ax.set_xlim(-60, 60)
ax.legend(fontsize=8, frameon=False, loc="upper left"); ax.set_title(f"SNR distribution, N = {SN*SN}", fontsize=10)
plt.tight_layout(); plt.savefig(f"out/th{GAMMA_TH_DB:.0f}_fig_snr_cdf.png", dpi=200)

fig, ax = plt.subplots(figsize=(6.5, 4))
sds = [0, 5, 10, 20, 30]
for b in ["c", 3, 2, 1]:
    ax.errorbar(sds, [pn[f"{BL[b]}_{s}"]["snr"] for s in sds], [pn[f"{BL[b]}_{s}"]["snr_sd"] for s in sds], color=COL[b], marker="o", lw=2, capsize=3, label=BL[b])
ax.set_xlabel("phase-noise std σ (deg)"); ax.set_ylabel("mean SNR over area (dB)"); ax.legend(frameon=False, fontsize=8)
ax.set_title(f"Phase 3 with phase noise ({n2} {y2:+.0f}°, N = {SN*SN}, 5 MC runs)", fontsize=10)
plt.tight_layout(); plt.savefig(f"out/th{GAMMA_TH_DB:.0f}_fig_phase_noise.png", dpi=200)

fig, ax = plt.subplots(figsize=(6.5, 4))
ths = np.arange(-20, 41, 1)
for lab, x, c, ls in cdfs:
    ax.plot(ths, [np.mean(db(x) >= t) for t in ths], color=c, ls=ls, lw=2, label=lab)
ax.set_xlabel("SNR threshold γth (dB)"); ax.set_ylabel("coverage probability"); ax.legend(fontsize=7, frameon=False)
plt.tight_layout(); plt.savefig(f"out/th{GAMMA_TH_DB:.0f}_fig_cov_vs_threshold.png", dpi=200)
print(json.dumps({k: out[k] for k in ["no_ris", "ideal_choice_in_RT", "table4", "theory_quant_loss_db"]}, indent=1))
