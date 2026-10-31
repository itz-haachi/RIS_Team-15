# Part B - 3D scene, ray tracing, no-RIS baseline (Team 15)

Tool: NVIDIA Sionna RT 2.2.0 (Mitsuba 3, CPU). Carrier 28 GHz, BW 100 MHz, BS 40 dBm EIRP (isotropic), noise -87 dBm.
Ray tracing: max depth 4, LOS + specular reflection + edge diffraction, no transmission through buildings, 1e6 rays/source, seed 42.

## Scene (folder `scene/` in the full zip, files fig_scene_3d.png / fig_scene_top.png)
Synthetic intersection, 220 m x 160 m: main street (x) + side street (y), 4 buildings 18-25 m (concrete, brick, glass),
metal kiosk in the side street, metal truck in the main street, concrete ground. BS at (-20, 6, 10) m.
UE grid: side street, 1 m spacing, 1.5 m height, 1,587 points (kiosk footprint removed).
RIS candidates R1-R6 on facades at 6 m (positions only; RIS optimisation is Parts A/C).

## Files
| File | Content |
|---|---|
| channel_paths_BS_to_UE.csv | direct channel hd: strongest 16 ray-traced paths per UE (complex amplitude, gain, delay, type, interaction points) |
| channel_paths_BS_to_RIS.csv | channel g: BS -> each RIS candidate centre |
| channel_paths_RIS_to_UE.csv | channel h_r: each RIS candidate centre -> each UE |
| RIS_candidates_BS_link.csv | each RIS candidate: LOS to BS?, ray-traced vs free-space BS->RIS gain |
| noRIS_baseline_per_UE.csv | per UE: LOS flag, #paths, path gain, received power, SNR (coherent) and local-mean SNR, coverage flags |
| noRIS_SNR_by_distance.csv | mean SNR and coverage in 10 m bands up the side street |
| noRIS_radiomap_full_area_1m.csv | full-area 1 m radio map (local mean) from Sionna RadioMapSolver |
| noRIS_summary.json | headline baseline numbers |
| fig_noRIS_sidestreet.png | coverage maps over the side street + SNR vs distance |
| fig_noRIS_radiomap_full_area.png | coverage map of the whole scene |

## Two SNR definitions
- SNR_dB (primary): |sum of path amplitudes|^2 - the narrowband channel used in Eq. (1). Includes multipath fading.
- SNR_localmean_dB: sum of |path amplitude|^2 - average power over a small area (no fading). Matches Sionna radio maps.

## How Parts A and C use these files
Per-element RIS channels are built from the RIS-centre paths: for path k with amplitude a_k and interaction point v_k next to
the RIS, element n gets a_k * exp(-j k0 (|v_k - p_n| - |v_k - c|)) * sqrt(G cos(theta_k)), G = pi (lambda/2 elements).
first_vertex_* (RIS->UE) and last_vertex_* (BS->RIS) columns give v_k. Complex amplitude = amp_real + j amp_imag.
