import numpy as np, os, json
D = os.path.join(os.path.dirname(__file__), "scene")

BUILDINGS = {
    "B1_NW": (-100, 30, 10, 100, 25, "concrete"),
    "B2_NE": (50, 120, 10, 100, 20, "brick"),
    "B3_SW": (-100, 20, -60, -10, 22, "glass"),
    "B4_SE": (20, 120, -60, -10, 18, "concrete"),
    "K1_kiosk": (37, 43, 55, 58, 3.5, "metal"),
    "T1_truck": (2, 12, 4, 6.5, 3.5, "metal"),
}
GROUND = (-100, 120, -60, 100)

def box(x0, x1, y0, y1, z0, z1):
    v = np.array([[x0,y0,z0],[x1,y0,z0],[x1,y1,z0],[x0,y1,z0],[x0,y0,z1],[x1,y0,z1],[x1,y1,z1],[x0,y1,z1]], float)
    f = [[0,2,1],[0,3,2],[4,5,6],[4,6,7],[0,1,5],[0,5,4],[1,2,6],[1,6,5],[2,3,7],[2,7,6],[3,0,4],[3,4,7]]
    return v, np.array(f)

def write_ply(path, v, f):
    with open(path, "w") as fh:
        fh.write(f"ply\nformat ascii 1.0\nelement vertex {len(v)}\nproperty float x\nproperty float y\nproperty float z\n"
                 f"element face {len(f)}\nproperty list uchar int vertex_indices\nend_header\n")
        for p in v: fh.write(f"{p[0]} {p[1]} {p[2]}\n")
        for t in f: fh.write(f"3 {t[0]} {t[1]} {t[2]}\n")

def main():
    os.makedirs(os.path.join(D, "meshes"), exist_ok=True)
    mats = {"concrete": 0.3, "brick": 0.3, "glass": 0.1, "metal": 0.01, "ground_concrete": 1.0}
    xml = ['<scene version="2.1.0">']
    for m, t in mats.items():
        itu = "concrete" if m == "ground_concrete" else m
        xml.append(f'<bsdf type="itu-radio-material" id="mat-{m}"><string name="type" value="{itu}"/><float name="thickness" value="{t}"/></bsdf>')
    x0, x1, y0, y1 = GROUND
    v = np.array([[x0,y0,0],[x1,y0,0],[x1,y1,0],[x0,y1,0]], float); f = np.array([[0,1,2],[0,2,3]])
    write_ply(os.path.join(D, "meshes", "ground.ply"), v, f)
    xml.append('<shape type="ply" id="mesh-ground"><string name="filename" value="meshes/ground.ply"/>'
               '<boolean name="face_normals" value="true"/><ref id="mat-ground_concrete" name="bsdf"/></shape>')
    for n, (a, b, c, d, h, m) in BUILDINGS.items():
        v, f = box(a, b, c, d, 0, h)
        write_ply(os.path.join(D, "meshes", f"{n}.ply"), v, f)
        xml.append(f'<shape type="ply" id="mesh-{n}"><string name="filename" value="meshes/{n}.ply"/>'
                   f'<boolean name="face_normals" value="true"/><ref id="mat-{m}" name="bsdf"/></shape>')
    xml.append("</scene>")
    open(os.path.join(D, "scene.xml"), "w").write("\n".join(xml))
    json.dump({"buildings": BUILDINGS, "ground": GROUND}, open(os.path.join(D, "geometry.json"), "w"), indent=1)
    print("scene written")

if __name__ == "__main__":
    main()
