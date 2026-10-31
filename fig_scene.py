import json, numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from config import *
g = json.load(open("scene/geometry.json"))
MATC = {"concrete": "#b9b6ad", "brick": "#c98a6b", "glass": "#9cc3d9", "metal": "#7d7d85"}
def faces(x0,x1,y0,y1,h):
    p=lambda x,y,z:(x,y,z)
    return [[p(x0,y0,h),p(x1,y0,h),p(x1,y1,h),p(x0,y1,h)],[p(x0,y0,0),p(x1,y0,0),p(x1,y0,h),p(x0,y0,h)],
            [p(x1,y0,0),p(x1,y1,0),p(x1,y1,h),p(x1,y0,h)],[p(x1,y1,0),p(x0,y1,0),p(x0,y1,h),p(x1,y1,h)],
            [p(x0,y1,0),p(x0,y0,0),p(x0,y0,h),p(x0,y1,h)]]
fig = plt.figure(figsize=(11, 7.5)); ax = fig.add_subplot(111, projection="3d")
for n,(x0,x1,y0,y1,h,m) in g["buildings"].items():
    ax.add_collection3d(Poly3DCollection(faces(x0,x1,y0,y1,h), facecolor=MATC[m], edgecolor="#555", linewidths=0.4, alpha=0.55))
X0,X1,Y0,Y1 = g["ground"]
ax.add_collection3d(Poly3DCollection([[(X0,Y0,0),(X1,Y0,0),(X1,Y1,0),(X0,Y1,0)]], facecolor="#e9e7e1", alpha=0.3))
ue,valid,_ = ue_grid()
ax.scatter(ue[valid,0],ue[valid,1],ue[valid,2],s=0.6,c="#2a78d6",alpha=0.5,label="UE grid (1,615 pts, side street)")
ax.scatter(*BS, s=90, c="#e34948", marker="^", depthshade=False, label="BS (28 GHz, 10 m)")
for k,(c,yaw) in RIS_CAND.items():
    ax.scatter(*c, s=60, c="#eb6834", marker="s", depthshade=False)
    ax.text(c[0], c[1], c[2]+2.5, k, fontsize=9, color="#111")
ax.scatter([],[],c="#eb6834",marker="s",label="RIS candidates R1–R6 (6 m)")
ax.set_xlim(-100,120); ax.set_ylim(-60,100); ax.set_zlim(0,60)
ax.set_box_aspect((220,160,60)); ax.view_init(elev=38, azim=-62)
ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_zlabel("z (m)")
from matplotlib.patches import Patch
h,l = ax.get_legend_handles_labels()
h += [Patch(facecolor=MATC[m],alpha=.6,label=f"ITU {m}") for m in MATC]
ax.legend(handles=h, loc="upper left", fontsize=8, frameon=False)
ax.set_title("3D site-specific scene: blocked urban intersection (side street = NLOS dead zone)")
plt.tight_layout(); plt.savefig("out/fig_scene_3d.png", dpi=200)

fig, ax = plt.subplots(figsize=(8,6.2))
for n,(x0,x1,y0,y1,h,m) in g["buildings"].items():
    ax.add_patch(plt.Rectangle((x0,y0),x1-x0,y1-y0,fc=MATC[m],ec="#555",lw=.6,alpha=.8))
    ax.text((x0+x1)/2,(y0+y1)/2,f"{n.split('_')[0]}\n{m}, {h} m",ha="center",va="center",fontsize=7)
ax.add_patch(plt.Rectangle((30.5,10.5),19,85,fill=False,ec="#2a78d6",ls="--",lw=1.2))
ax.text(40,97,"evaluation area",ha="center",fontsize=8,color="#2a78d6")
ax.plot(*BS[:2],"^",c="#e34948",ms=10); ax.text(BS[0],BS[1]+3,"BS",ha="center",fontsize=9)
for k,(c,yaw) in RIS_CAND.items():
    ax.plot(c[0],c[1],"s",c="#eb6834",ms=6)
    ax.annotate("",xy=(c[0]+6*np.cos(np.deg2rad(yaw)),c[1]+6*np.sin(np.deg2rad(yaw))),xytext=(c[0],c[1]),arrowprops=dict(arrowstyle="->",color="#eb6834"))
    ax.text(c[0]+1.5,c[1]-4 if c[1]<0 else c[1]+1.5,k,fontsize=8)
ax.set_xlim(-100,120); ax.set_ylim(-60,100); ax.set_aspect("equal"); ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
ax.set_title("Top view (arrows = RIS facade normal)")
plt.tight_layout(); plt.savefig("out/fig_scene_top.png", dpi=200)
