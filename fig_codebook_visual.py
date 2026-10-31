import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from config import *
from analysis import cascade, direct, SUBS, PT, SIG2
rt = dict(np.load("out/rt_paths.npz")); ue, valid = rt["ue"], rt["valid"]
hd = direct(2, rt, ue)[valid]; U = ue[valid]
db = lambda x: np.maximum(10*np.log10(np.maximum(x, 1e-30)), -60)
cn = cascade(2, rt, ue, "R1", 30.0)[valid][:, SUBS[128]]
th = np.angle(hd)[:, None] - np.angle(cn)

def codebook(nx, ny):
    xs = np.linspace(31, 49, nx+2)[1:-1] if nx > 1 else [40.0]
    ys = np.linspace(11, 95, 2*ny+1)[1::2]
    best = np.full(len(U), -np.inf); foci = []
    for xc in xs:
        for yc in ys:
            i = np.argmin((U[:, 0]-xc)**2 + (U[:, 1]-yc)**2); foci.append(U[i, :2])
            best = np.maximum(best, PT*np.abs(hd + cn @ np.exp(1j*th[i]))**2/SIG2)
    return db(best), np.array(foci)

cases = [("No RIS", db(PT*np.abs(hd)**2/SIG2), None)]
for K, (nx, ny) in ((1, (1, 1)), (2, (1, 2)), (4, (1, 4)), (16, (2, 8)), (64, (4, 16))):
    s, f = codebook(nx, ny); cases.append((f"K = {K}", s, f))
ideal = PT*(np.abs(hd) + np.abs(cn).sum(1))**2/SIG2
cases.append(("Ideal: re-tuned\nfor every user", db(ideal), None))

def grid(v):
    g = np.full(len(ue), np.nan); g[valid] = v; return g.reshape(85, 19)

cmap = ListedColormap(["#d9d9d9", "#2e9e5b"])
fig, axs = plt.subplots(1, 7, figsize=(17, 7.2))
for ax, (lab, s, f) in zip(axs, cases):
    ax.imshow(grid((s >= 10).astype(float)), origin="lower", extent=[30.5, 49.5, 10.5, 95.5], cmap=cmap, vmin=0, vmax=1)
    ax.add_patch(plt.Rectangle((37, 55), 6, 3, color="k"))
    if f is not None: ax.plot(f[:, 0], f[:, 1], "*", color="#ffcc00", mec="k", mew=0.6, ms=12 if len(f) < 10 else 7)
    ax.plot(40, 9, "s", color="#1f6fd1", ms=9, clip_on=False)
    cov = 100*np.mean(s >= 10)
    ax.set_title(f"{lab}\n{cov:.1f}% covered", fontsize=13, fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([]); ax.set_xlim(30.5, 49.5); ax.set_ylim(8, 95.5)
axs[0].set_ylabel("side street  (blue square = RIS, below the street)", fontsize=11)
fig.suptitle("Fixed codebook with K profiles (RIS R1, 16,384 elements)   green = covered (SNR ≥ 10 dB),  ★ = where each profile is aimed",
             fontsize=13)
plt.tight_layout(rect=(0, 0, 1, 0.95)); plt.savefig("out/fig_codebook_visual.png", dpi=150)
print([(c[0], round(100*np.mean(c[1] >= 10), 1)) for c in cases])
