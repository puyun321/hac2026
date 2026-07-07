"""
Convert .msh asteroid shape files to voxelised .npy training labels.

Grid: [-2,2] x [-2,2] x [-1,1], pitch=0.01 -> 400x400x200 uint8

Usage:
    python msh_to_npy.py
"""

import struct
from pathlib import Path
import numpy as np
import trimesh

_BASE    = Path(__file__).resolve().parent.parent
DATA_DIR = _BASE / "data"
OUT_DIR  = _BASE / "npy_data"

PITCH = 0.01
NX, NY, NZ = 400, 400, 200
XMIN, YMIN, ZMIN = -2.0, -2.0, -1.0

_ELEM_NNODES = {1: 2, 2: 3, 3: 4, 4: 4, 5: 8, 6: 6, 7: 5,
                8: 3, 9: 6, 10: 9, 11: 10, 15: 1, 16: 8, 17: 20}


def read_msh(path):
    """Read Gmsh 2.2 binary .msh → (vertices float64 (N,3), faces int32 (M,3))."""
    raw = path.read_bytes()

    def section(name):
        s = f"${name}\n".encode()
        e = f"$End{name}\n".encode()
        si = raw.find(s)
        ei = raw.find(e, si)
        p  = si + len(s)
        eol = raw.index(b"\n", p)
        n = int(raw[p:eol])
        return raw[eol + 1: ei], n

    # Nodes: [int32 id, double x, double y, double z] x n
    node_raw, n_nodes = section("Nodes")
    STRIDE = 4 + 24
    verts = np.empty((n_nodes, 3), dtype=np.float64)
    for i in range(n_nodes):
        verts[i] = struct.unpack_from("<ddd", node_raw, i * STRIDE + 4)

    # Elements
    elem_raw, n_total = section("Elements")
    tris = []
    p, done = 0, 0
    while done < n_total:
        etype, nb, ntags = struct.unpack_from("<iii", elem_raw, p)
        p += 12
        nn = _ELEM_NNODES.get(etype, 0)
        stride = 4 + ntags * 4 + nn * 4
        if etype == 2:
            for _ in range(nb):
                off = p + 4 + ntags * 4
                n1, n2, n3 = struct.unpack_from("<iii", elem_raw, off)
                tris.append((n1 - 1, n2 - 1, n3 - 1))
                p += stride
        else:
            p += nb * stride
        done += nb

    faces = np.array(tris, dtype=np.int32) if tris else np.zeros((0, 3), np.int32)
    return verts, faces


def msh_to_voxel(msh_path):
    verts, faces = read_msh(msh_path)

    # centre and scale to fit inside the contest bounding box
    verts = verts - verts.mean(axis=0)
    scale = 0.95 / (np.abs(verts).max(axis=0) / np.array([2.0, 2.0, 1.0])).max()
    verts = verts * scale

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
    if not mesh.is_watertight:
        trimesh.repair.fill_holes(mesh)

    vox = mesh.voxelized(pitch=PITCH).fill()
    pts = vox.points

    V  = np.zeros((NX, NY, NZ), dtype=np.uint8)
    ix = np.floor((pts[:, 0] - XMIN) / PITCH).astype(int)
    iy = np.floor((pts[:, 1] - YMIN) / PITCH).astype(int)
    iz = np.floor((pts[:, 2] - ZMIN) / PITCH).astype(int)
    ok = (ix >= 0) & (ix < NX) & (iy >= 0) & (iy < NY) & (iz >= 0) & (iz < NZ)
    V[ix[ok], iy[ok], iz[ok]] = 1
    return V


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    msh_files = sorted(DATA_DIR.glob("AsteroidModel*/*.msh"))

    if not msh_files:
        print(f"No .msh files found under {DATA_DIR}")
        return

    for msh in msh_files:
        out = OUT_DIR / f"{msh.parent.name}.npy"
        print(f"  {msh.parent.name} ...", end=" ", flush=True)
        V = msh_to_voxel(msh)
        np.save(out, V)
        print(f"ok  ({int(V.sum())} voxels)  -> {out.name}")

    print(f"\n{len(msh_files)} files saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
