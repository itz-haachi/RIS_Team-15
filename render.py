import numpy as np, sionna.rt as rt, mitsuba as mi
from config import *
sc = rt.load_scene("scene/scene.xml"); sc.frequency = FC
sc.tx_array = rt.PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
sc.rx_array = rt.PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
sc.add(rt.Transmitter("BS", position=mi.Point3f(*map(float,BS)), display_radius=2.5))
for k,(c,y) in RIS_CAND.items():
    sc.add(rt.Receiver(k, position=mi.Point3f(*map(float,c)), display_radius=1.5, color=(0.92,0.41,0.2)))
cam = rt.Camera(position=mi.Point3f(-60, -110, 120), look_at=mi.Point3f(35, 30, 0))

for k in RIS_CAND: sc.remove(k)
rm = rt.RadioMapSolver()(sc, center=mi.Point3f(10, 20, 1.5), orientation=mi.Point3f(0,0,0), size=mi.Point2f(220, 160), cell_size=mi.Point2f(1,1),
                         max_depth=4, refraction=False, diffraction=True, samples_per_tx=int(2e7))
sc.render_to_file(camera=cam, filename="out/fig_radiomap_noRIS_sionna.png", resolution=(1400, 900), num_samples=64,
                  radio_map=rm, rm_metric="path_gain", rm_vmin=-160, rm_vmax=-70)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import numpy as np
pg=10*np.log10(np.maximum(np.array(rm.path_gain)[0],1e-30))
np.save("out/radiomap_noRIS_pathgain_db.npy", pg)
fig,ax=plt.subplots(figsize=(8,6))
im=ax.imshow(pg+PT_DBM-NOISE_DBM, origin="lower", extent=[-100,120,-60,100], cmap="Blues", vmin=-40, vmax=60)
ax.plot(*BS[:2],"^",c="#e34948",ms=9)
for k,(c,y) in RIS_CAND.items(): ax.plot(c[0],c[1],"s",c="#eb6834",ms=5); ax.text(c[0]+1.5,c[1]+1.5,k,fontsize=7)
fig.colorbar(im,label="SNR without RIS (dB)"); ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_title("Ray-traced SNR map, no RIS (Sionna RT, 28 GHz, UE height 1.5 m)", fontsize=10)
plt.tight_layout(); plt.savefig("out/fig_radiomap_noRIS.png", dpi=200)
print("ok")
