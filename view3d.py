import json, numpy as np
import plotly.graph_objects as go
from config import RIS_CAND

MAT_COL = {"concrete": "#b8b8b8", "brick": "#b5654d", "glass": "#7f86d6", "metal": "#4a4a4a"}
USER_COL = ["#2ca02c", "#ff7f0e", "#d62728", "#9467bd"]

def box(x0, x1, y0, y1, h, color, name):
    x = [x0, x1, x1, x0, x0, x1, x1, x0]; y = [y0, y0, y1, y1, y0, y0, y1, y1]; z = [0, 0, 0, 0, h, h, h, h]
    i = [0, 0, 4, 4, 0, 0, 1, 1, 2, 2, 3, 3]; j = [1, 2, 5, 6, 1, 5, 2, 6, 3, 7, 0, 4]; k = [2, 3, 6, 7, 5, 4, 6, 5, 7, 6, 4, 7]
    return go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k, color=color, opacity=0.55, flatshading=True,
                     name=name, hovertext=name, hoverinfo="text", showlegend=False)

def make(paths, bs, users, out="out/scene_3d.html", per_user=12):
    g = json.load(open("scene/geometry.json"))
    tr = [go.Mesh3d(x=[-100, 120, 120, -100], y=[-60, -60, 100, 100], z=[0, 0, 0, 0], i=[0, 0], j=[1, 2], k=[2, 3],
                    color="#e8e4da", opacity=1, hoverinfo="skip", showlegend=False)]
    for n, (x0, x1, y0, y1, h, m) in g["buildings"].items():
        tr.append(box(x0, x1, y0, y1, h, MAT_COL[m], f"{n} ({m}, {h} m)"))
    a = np.array(paths.a[0]) + 1j * np.array(paths.a[1]); a = np.abs(a[:, 0, 0, 0, :])
    v = np.array(paths.vertices)
    nb = (np.array(paths.interactions) != 0).sum(0)[:, 0, :]
    for u, (name, pos) in enumerate(users.items()):
        col = USER_COL[u % len(USER_COL)]
        order = [p for p in np.argsort(-a[u]) if a[u, p] > 0][:per_user]
        xs, ys, zs = [], [], []
        for p in order:
            pts = [bs] + [v[d, u, 0, p] for d in range(nb[u, p])] + [pos]
            for q in pts: xs.append(q[0]); ys.append(q[1]); zs.append(q[2])
            xs.append(None); ys.append(None); zs.append(None)
        tr.append(go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", line=dict(color=col, width=3), name=f"rays to {name}"))
        tr.append(go.Scatter3d(x=[pos[0]], y=[pos[1]], z=[pos[2]], mode="markers+text", text=[name.split()[0]],
                               marker=dict(size=6, color=col), textposition="top center", showlegend=False, hovertext=name))
    tr.append(go.Scatter3d(x=[bs[0]], y=[bs[1]], z=[bs[2]], mode="markers+text", text=["BS"], name="Base station",
                           marker=dict(size=9, color="red", symbol="diamond"), textposition="top center"))
    rc = np.array([c for c, _ in RIS_CAND.values()])
    tr.append(go.Scatter3d(x=rc[:, 0], y=rc[:, 1], z=rc[:, 2], mode="markers+text", text=list(RIS_CAND), name="RIS candidates",
                           marker=dict(size=5, color="#00a6a6", symbol="square"), textposition="top center"))
    fig = go.Figure(tr)
    fig.update_layout(title="28 GHz ray tracing (Sionna RT) — drag to rotate, scroll to zoom, click legend to hide",
                      scene=dict(aspectmode="data", xaxis_title="x (m)", yaxis_title="y (m)", zaxis_title="z (m)",
                                 camera=dict(eye=dict(x=-0.6, y=-1.3, z=0.9), center=dict(x=0.1, y=0.05, z=-0.2))),
                      margin=dict(l=0, r=0, t=40, b=0), legend=dict(x=0, y=0.95))
    fig.write_html(out, include_plotlyjs=True)
    return out
