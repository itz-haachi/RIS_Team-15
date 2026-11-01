import time, numpy as np, sionna.rt as rt, mitsuba as mi
from config import FC, BS, PT_DBM, NOISE_DBM, TRACE

USERS = {
    "U1 junction (y=15 m)":       (40, 15, 1.5),
    "U2 mid street (y=40 m)":     (40, 40, 1.5),
    "U3 behind kiosk (y=62 m)":   (40, 62, 1.5),
    "U4 deep street (y=90 m)":    (40, 90, 1.5),
}

print("1) Loading the scene (buildings + ITU materials) ...")
sc = rt.load_scene("scene/scene.xml", merge_shapes=False)
sc.frequency = FC
sc.tx_array = rt.PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
sc.rx_array = rt.PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
for name, obj in sc.objects.items():
    print(f"   {name:<12} material: {obj.radio_material.name}")

print(f"\n2) Placing the BS at ({BS[0]:.0f}, {BS[1]:.0f}, {BS[2]:.0f}) m and {len(USERS)} users")
sc.add(rt.Transmitter("BS", position=mi.Point3f(*map(float, BS)), display_radius=2.5))
for i, p in enumerate(USERS.values()):
    sc.add(rt.Receiver(f"U{i+1}", position=mi.Point3f(*map(float, p)), display_radius=1.5))

print(f"\n3) Ray tracing at {FC/1e9:.0f} GHz (up to {TRACE['max_depth']} bounces, diffraction on) ...")
t = time.time()
paths = rt.PathSolver()(sc, **TRACE)
print(f"   done in {time.time()-t:.1f} s")

a = np.array(paths.a[0]) + 1j * np.array(paths.a[1])
a = a[:, 0, 0, 0, :]
inter = np.array(paths.interactions)
nb = (inter != 0).sum(0)[:, 0, :]
print(f"\n4) Results (SNR = {PT_DBM:.0f} dBm - path loss - noise {NOISE_DBM:.0f} dBm; coverage needs >= 10 dB)")
print(f"   {'user':<26}{'paths':>6}{'LOS?':>6}{'strongest path':>17}{'SNR (dB)':>10}")
for i, name in enumerate(USERS):
    ok = np.abs(a[i]) > 0
    g = np.sum(np.abs(a[i][ok]) ** 2)
    snr = PT_DBM + 10 * np.log10(max(g, 1e-30)) - NOISE_DBM
    los = bool(np.any(ok & (nb[i] == 0)))
    if ok.any():
        k = np.argmax(np.abs(a[i])); kind = "direct (LOS)" if nb[i, k] == 0 else f"{nb[i, k]} bounce(s)"
    else:
        kind = "none"
    print(f"   {name:<26}{ok.sum():>6}{'yes' if los else 'no':>6}{kind:>17}{snr:>10.1f}")

print("\n5) Rendering the rays -> out/demo_rays.png")
cam = rt.Camera(position=mi.Point3f(15, -75, 150), look_at=mi.Point3f(30, 45, 0))
sc.render_to_file(camera=cam, filename="out/demo_rays.png", paths=paths, resolution=(1400, 900), num_samples=16)
print("   saved. Open it to show the rays bouncing off the buildings.")

try:
    import view3d
    print("6) Interactive 3D model ->", view3d.make(paths, BS, USERS), " (open in a browser, drag to rotate)")
except ImportError:
    print("6) (pip install plotly to also get the interactive 3D model)")
