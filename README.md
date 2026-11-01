# RIS-Assisted Coverage Enhancement in Blocked/NLOS Environments

Code, scene and results for the paper *RIS-Assisted Coverage Enhancement in Blocked/NLOS Environments: A Site-Specific, Phase-Quantization-Aware Framework* (VIT, Team 15).

Everything needed to reproduce every table and figure is here:
- the synthetic 28 GHz intersection (Sionna RT scene, 1,587 UE locations, RIS sites R1–R6)
- the full pipeline: Phase 1 (ideal), Phase 1b (blockage-aware), Phase 2 (Sionna RT ray tracing), Phase 3 (1/2/3-bit quantization, phase noise)
- the codebook (K-profile) and second-BS experiments
- the ray-traced paths, per-UE SNR for all 168 configurations, and CSVs behind each table and figure

The scene is synthetic; nothing here is measured data.

## Setup
Python 3.10–3.12, CPU only.
```
pip install -r requirements.txt
```
**Windows:** Sionna RT needs LLVM. Install it (`winget install LLVM.LLVM`), then run
`setx DRJIT_LIBLLVM_PATH "C:\Program Files\LLVM\bin\LLVM-C.dll"` and open a new terminal.
**macOS:** `brew install llvm`.

Run every command from this folder. All parameters are in `config.py`:
28 GHz, 100 MHz, 40 dBm EIRP, noise −87 dBm, γth = 10 dB, BS (−20, 6, 10) m, RIS candidates R1–R6 at 6 m height,
orientations −45…+45° in 15° steps, N = 32², 64², 96², 128².

## Quick demo (seconds)
```
python demo_live.py     # traces 4 users, prints SNR table, saves out/demo_rays.png and out/scene_3d.html
jupyter lab demo.ipynb  # scene + rays in an interactive 3D view
```

## Reproducing the paper
`out/` already holds every intermediate result, so steps 1–3 can be skipped.

| Step | Command | Output | Paper |
|---|---|---|---|
| 1 | `python build_scene.py` | `scene/` | §3.2, Table 2 |
| 2 | `python trace.py` (~25 min) | `out/rt_paths.npz` | Phase 2 channels |
| 3 | `python analysis.py 1`, `python phase1b.py`, `python analysis.py 2` | `out/snr_p1.npz`, `snr_p1b.npz`, `snr_p2.npz` | Phases 1, 1b, 2, 3 |
| 4 | `RIS_TH=10 python results.py`, `python final_stats.py`, `python table05_full_grid.py` | `out/results_th10.json`, `final_stats_th10.json`, `csv/table05_full_grid.csv`, Fig. 5 | Table 5, §4.1–4.2 |
| 5 | `python compare_models.py`, `python fig_three_models.py` | Fig. 4 | §4.1 |
| 6 | `python codebook2.py`, `python codebook_k12.py` | `out/codebook*_R1.json` | Table 6, §4.3 |
| 7 | `RIS_BS=100,-6,10 RIS_TAG=_bs2 python trace_bs2.py`, then step 3 with the same variables, `python compare_bs2.py` | `out/*_bs2*`, `out/bs_compare.json` | Table 7, §4.4 |
| 8 | `python paper_figs.py` | Figs. 7–10 and `csv/` | Figs. 7–10 |
| 9 | `python fig_scene.py`, `python fig_final.py`, `python export_partB.py`, `python fig_partB.py`, `python render.py` | Figs. 1, 2, 3, 6; `partB/` | §3, §4 |

`RIS_TH=0` gives the 0 dB threshold sensitivity check (`out/results_th0.json`).

## Figure and table map
| Paper | Script | Data |
|---|---|---|
| Fig. 1 scene | `fig_scene.py` | `scene/` |
| Fig. 2 framework | `fig_final.py` | — |
| Fig. 3 no-RIS baseline | `fig_partB.py` | `partB/noRIS_*.csv` |
| Fig. 4 sites under three models | `fig_three_models.py` | `csv/fig04_sites_under_three_models.csv` |
| Fig. 5 SNR maps | `results.py` | `out/snr_p1.npz`, `out/snr_p2.npz` |
| Fig. 6 coverage/SNR vs N | `fig_final.py` | `csv/table05_and_fig06_stats_all_N.csv` |
| Fig. 7 phase noise | `paper_figs.py` | `csv/fig07_phase_noise.csv` |
| Fig. 8 Phase 3 metrics vs N | `paper_figs.py` | `csv/fig08_phase3_metrics_vs_N.csv` |
| Fig. 9 quantization loss vs theory | `paper_figs.py` | `csv/fig09_quantization_loss_vs_theory.csv` |
| Fig. 10 cost of the ideal site | `paper_figs.py` | `csv/fig10_cost_of_ideal_site.csv` |
| Table 5 | `table05_full_grid.py` (± from `final_stats.py`) | `csv/table05_full_grid.csv` |
| Table 6 codebook | `codebook2.py`, `codebook_k12.py` | `csv/table06_codebook.csv` |
| Table 7 second BS | `compare_bs2.py` | `csv/table07_second_bs.csv` |
| Spearman ranking (§4.1) | `compare_models.py` | `csv/table_spearman.csv` |

Extra explainer figures (not in the paper): `fig_codebook_visual.py` (codebook K), `fig_quant_visual.py` (phase quantization).

## Notes
- **UE count.** The grid has 19 × 85 = 1,615 points; the 28 points inside the kiosk footprint are removed, leaving 1,587. Files covering the whole grid (for example `partB/noRIS_baseline_per_UE.csv`) have 1,615 rows, and the 1,587 used in the paper are marked `valid = True`.
- **Zero-path UEs.** Three valid UEs (UE0920–UE0922, at x = 39–41 m, y = 59 m, directly behind the kiosk) receive no ray-traced path without an RIS. They are stored at −173 dB SNR in `partB/noRIS_baseline_per_UE.csv` and, like every UE, are floored at −60 dB for the mean SNR (Eq. (5) in the paper). They are included in the −13.9 dB no-RIS mean; excluding them would give −13.8 dB.
- **Spearman ranking.** The rank correlations between each simplified model and ray tracing (0.43 for Phase 1, 0.66 for Phase 1b; six candidate sites) are saved by `compare_models.py` in `csv/table_spearman.csv`.

## Statistics
Point estimates are full-grid values over all 1,587 UEs (deterministic ray tracing), as in `csv/table05_full_grid.csv`. ± values are the SD over 20 random subsamples of 200 UEs (seed 2026). `csv/table05_and_fig06_stats_all_N.csv` holds the means of those subsamples (used for Fig. 6), so its values differ slightly from the full-grid values in Table 5. Phase noise uses 20 Monte Carlo draws.

## Licence
Code: MIT (`LICENSE`). Scene, data and results: CC BY 4.0.

## Citation
See `CITATION.cff`.
