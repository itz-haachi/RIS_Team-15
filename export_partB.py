import numpy as np, pandas as pd, json, os
from config import *
O = "partB"; os.makedirs(O, exist_ok=True)
d = dict(np.load("out/rt_paths.npz")); ue, valid = d["ue"], d["valid"]
TYPE = {0: "LOS"}
def paths_df(prefix, tx_names, rx_names):
    a, L, dep, vf, vl = (d[f"{prefix}_{k}"] for k in ("a", "L", "depth", "vf", "vl"))
    rows = []
    R, T, K = a.shape
    for r in range(R):
        for t in range(T):
            for k in range(K):
                if abs(a[r, t, k]) == 0: continue
                rows.append(dict(tx=tx_names[t], rx=rx_names[r], path_rank=k+1,
                    gain_dB=20*np.log10(abs(a[r, t, k])), phase_rad=float(np.angle(a[r, t, k])),
                    amp_real=float(a[r, t, k].real), amp_imag=float(a[r, t, k].imag),
                    length_m=float(L[r, t, k]), delay_ns=float(L[r, t, k]/C0*1e9),
                    n_interactions=int(dep[r, t, k]), type="LOS" if dep[r, t, k] == 0 else f"{dep[r,t,k]}-bounce/diffracted",
                    first_vertex_x=vf[r,t,k,0], first_vertex_y=vf[r,t,k,1], first_vertex_z=vf[r,t,k,2],
                    last_vertex_x=vl[r,t,k,0], last_vertex_y=vl[r,t,k,1], last_vertex_z=vl[r,t,k,2]))
    return pd.DataFrame(rows)
ue_names = [f"UE{i:04d}" for i in range(len(ue))]
ris = list(RIS_CAND)
paths_df("d", ["BS"], ue_names).round(6).to_csv(f"{O}/channel_paths_BS_to_UE.csv", index=False)
paths_df("g", ["BS"], ris).round(6).to_csv(f"{O}/channel_paths_BS_to_RIS.csv", index=False)
pd.concat([paths_df(f"h{n}", [n], ue_names) for n in ris]).round(6).to_csv(f"{O}/channel_paths_RIS_to_UE.csv", index=False)

a = d["d_a"][:, 0]; dep = d["d_depth"][:, 0]
pg = np.sum(np.abs(a)**2, -1); los = np.any((dep == 0) & (np.abs(a) > 0), -1)
hd = a.sum(-1)
pg_coh = np.abs(hd)**2
snr = PT_DBM + 10*np.log10(np.maximum(pg_coh, 1e-30)) - NOISE_DBM
snr_avg = PT_DBM + 10*np.log10(np.maximum(pg, 1e-30)) - NOISE_DBM
ue_df = pd.DataFrame(dict(ue=ue_names, x_m=ue[:,0], y_m=ue[:,1], z_m=ue[:,2], valid=valid, LOS=los,
    n_paths=(np.abs(a) > 0).sum(-1), path_gain_dB=10*np.log10(np.maximum(pg_coh,1e-30)),
    rx_power_dBm=PT_DBM+10*np.log10(np.maximum(pg_coh,1e-30)), SNR_dB=snr, SNR_localmean_dB=snr_avg,
    hd_real=hd.real, hd_imag=hd.imag, covered_0dB=snr >= 0, covered_10dB=snr >= 10)).round(4)
ue_df.to_csv(f"{O}/noRIS_baseline_per_UE.csv", index=False)

ga = d["g_a"][:, 0]; gd = d["g_depth"][:, 0]
rows = []
for i, n in enumerate(ris):
    c = np.array(RIS_CAND[n][0]); fs = 20*np.log10(LAM/(4*np.pi*np.linalg.norm(BS-c)))
    rows.append(dict(ris=n, x_m=c[0], y_m=c[1], z_m=c[2], facade_normal_deg=RIS_CAND[n][1],
        LOS_to_BS=bool(np.any((gd[i] == 0) & (np.abs(ga[i]) > 0))), raytraced_gain_dB=10*np.log10(np.sum(np.abs(ga[i])**2)),
        free_space_gain_dB=fs))
pd.DataFrame(rows).round(2).to_csv(f"{O}/RIS_candidates_BS_link.csv", index=False)

pgm = np.load("out/radiomap_noRIS_pathgain_db.npy")
ys, xs = np.arange(pgm.shape[0]) - 59.5, np.arange(pgm.shape[1]) - 99.5
X, Y = np.meshgrid(xs, ys); m = pgm > -290
pd.DataFrame(dict(x_m=X[m], y_m=Y[m], path_gain_localmean_dB=pgm[m], SNR_localmean_dB=pgm[m]+PT_DBM-NOISE_DBM)).round(2).to_csv(f"{O}/noRIS_radiomap_full_area_1m.csv", index=False)
v = ue_df[ue_df.valid]
summ = dict(n_UE=int(len(v)), LOS_fraction=float(v.LOS.mean()), coverage_0dB=float(v.covered_0dB.mean()), coverage_10dB=float(v.covered_10dB.mean()),
            mean_SNR_dB=float(np.maximum(v.SNR_dB, -60).mean()), median_SNR_dB=float(v.SNR_dB.median()),
            localmean_coverage_0dB=float((v.SNR_localmean_dB>=0).mean()), localmean_coverage_10dB=float((v.SNR_localmean_dB>=10).mean()),
            localmean_mean_SNR_dB=float(np.maximum(v.SNR_localmean_dB,-60).mean()),
            mean_paths_per_UE=float(v.n_paths.mean()))

bands = []
for y0 in range(10, 100, 10):
    s = v[(v.y_m >= y0) & (v.y_m < y0+10)]
    bands.append(dict(y_band_m=f"{y0}-{y0+10}", n=len(s), mean_SNR_dB=round(float(np.maximum(s.SNR_dB,-60).mean()),1),
                      coverage_0dB=round(float(s.covered_0dB.mean()),3), coverage_10dB=round(float(s.covered_10dB.mean()),3)))
pd.DataFrame(bands).to_csv(f"{O}/noRIS_SNR_by_distance.csv", index=False)
json.dump(dict(summary=summ, bands=bands), open(f"{O}/noRIS_summary.json","w"), indent=1)
print(json.dumps(summ, indent=1)); print(pd.DataFrame(bands).to_string(index=False))
for f in sorted(os.listdir(O)): print(f, os.path.getsize(f"{O}/{f}"))
