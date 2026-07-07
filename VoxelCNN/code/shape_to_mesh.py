
import struct
import numpy as np
from scipy.spatial import ConvexHull
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — required for thread safety
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from pathlib import Path

SHAPE_FILE = Path(r"C:\Users\TKU\Desktop\lab\research\hac2026\simulation\shape_result.out")
OUTPUT_DIR  = Path(r"C:\Users\TKU\Desktop\lab\research\hac2026\simulation")


def read_gaussian_image(filepath):
    """Return (normals, areas) arrays from convexinv .out file."""
    normals, areas = [], []
    with open(filepath) as f:
        n_facets = int(f.readline())
        for _ in range(n_facets):
            areas.append(float(f.readline()))
            normals.append(list(map(float, f.readline().split())))
    return np.array(normals), np.array(areas)


def write_obj(path, vertices, faces):
    with open(path, 'w') as f:
        f.write("# Asteroid shape model (convexinv -> dual reconstruction)\n")
        for v in vertices:
            f.write(f"v {v[0]:.8f} {v[1]:.8f} {v[2]:.8f}\n")
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
    print(f"Written OBJ  -> {path}")


def write_ply(path, vertices, faces):
    with open(path, 'w') as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(vertices)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_indices\nend_header\n")
        for v in vertices:
            f.write(f"{v[0]:.8f} {v[1]:.8f} {v[2]:.8f}\n")
        for face in faces:
            f.write(f"3 {face[0]} {face[1]} {face[2]}\n")
    print(f"Written PLY  -> {path}")


def normalize_mesh(verts, cylinder_R=None):
    """
    Normalize vertices to competition coordinates: z ∈ [-1, 1], xy centroid at origin.

    Without cylinder_R: uniform scale so z spans [-1, 1].
    With cylinder_R: scale z independently to [-1, 1], then scale only xy so
    max_xy_radius = cylinder_R.  Both constraints are satisfied simultaneously
    because the competition guarantees the true shape touches both z=±1 planes
    AND the cylinder wall.
    """
    verts = np.asarray(verts, dtype=np.float64).copy()
    verts[:, 0] -= verts[:, 0].mean()
    verts[:, 1] -= verts[:, 1].mean()
    z_lo, z_hi = verts[:, 2].min(), verts[:, 2].max()
    verts[:, 2] -= (z_lo + z_hi) / 2
    verts *= 2.0 / (z_hi - z_lo)          # uniform scale: z → [-1, 1]
    if cylinder_R is not None:
        xy_max = float(np.sqrt(verts[:, 0] ** 2 + verts[:, 1] ** 2).max())
        if xy_max > 0:
            xy_scale = cylinder_R / xy_max
            verts[:, 0] *= xy_scale        # scale only xy, z stays in [-1, 1]
            verts[:, 1] *= xy_scale
    return verts


def write_stl(path, vertices, faces):
    """Write binary STL file."""
    vertices = np.asarray(vertices, dtype=np.float32)
    with open(path, 'wb') as f:
        f.write(b'\0' * 80)
        f.write(struct.pack('<I', len(faces)))
        for face in faces:
            v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
            n = np.cross(v1 - v0, v2 - v0).astype(np.float32)
            nl = np.linalg.norm(n)
            if nl > 0:
                n /= nl
            f.write(struct.pack('<3f', *n))
            f.write(struct.pack('<3f', *v0))
            f.write(struct.pack('<3f', *v1))
            f.write(struct.pack('<3f', *v2))
            f.write(struct.pack('<H', 0))
    print(f"Written STL  -> {path}")


def write_msh(path, vertices, faces):
    """Write Gmsh 2.2 binary mesh (.msh) compatible with the competition reference format."""
    n_nodes = len(vertices)
    n_elems = len(faces)
    with open(path, 'wb') as f:
        # MeshFormat section (text)
        f.write(b"$MeshFormat\n")
        f.write(b"2.2 1 8\n")
        f.write(struct.pack('<i', 1))   # endian check integer
        f.write(b"\n$EndMeshFormat\n")

        # Nodes section
        f.write(f"$Nodes\n{n_nodes}\n".encode())
        for i, v in enumerate(vertices, 1):
            f.write(struct.pack('<i3d', i, v[0], v[1], v[2]))
        f.write(b"\n$EndNodes\n")

        # Elements section — all triangles (type 2), 2 tags
        f.write(f"$Elements\n{n_elems}\n".encode())
        f.write(struct.pack('<iii', 2, n_elems, 2))  # type=2, count, num_tags=2
        for i, face in enumerate(faces, 1):
            # elem-number, tag1 (physical=0), tag2 (elementary=1), n1, n2, n3 (1-based)
            f.write(struct.pack('<6i', i, 0, 1, face[0]+1, face[1]+1, face[2]+1))
        f.write(b"\n$EndElements\n")
    print(f"Written MSH  -> {path}")


def reconstruct_polyhedron(normals, areas):
    """
    Dual-polytope reconstruction: each Gaussian-image triangle (n_a, n_b, n_c)
    maps to one polyhedron vertex via [n_a; n_b; n_c] @ p = [h_a, h_b, h_c].
    h ∝ (mean_area / area)^0.25 preserves elongation; h=1 everywhere would force a sphere.
    """
    # Step 1: triangulate the unit sphere using the normals
    gauss_hull = ConvexHull(normals)

    # Support-function estimate.
    # The Gauss-curvature relation gives: Gaussian-image area σ ∝ h^{-4}.
    # Therefore h ∝ σ^{-1/4} = (mean/area)^{1/4}.
    # Using h ∝ area (the forward relation) builds the *dual* polytope instead
    # of the shape itself — for a cube that dual is an octahedron (diamond).
    areas_safe = np.maximum(areas, areas.max() * 1e-3)  # avoid division by zero
    h = (areas_safe.mean() / areas_safe) ** 0.25

    # Step 2: for each Gaussian triangle, compute the polyhedron vertex
    verts = []
    for tri in gauss_hull.simplices:
        A   = normals[tri]   # 3×3 matrix of face normals
        rhs = h[tri]         # area-derived support function values
        try:
            p = np.linalg.solve(A, rhs)
            verts.append(p)
        except np.linalg.LinAlgError:
            pass
    verts = np.array(verts)

    # Remove statistical outliers before taking the convex hull — triangles
    # where all three normals have very small area can still produce far-out
    # vertices that would otherwise collapse the hull to a spiky shape.
    r = np.linalg.norm(verts, axis=1)
    verts = verts[r <= np.percentile(r, 95)]

    # Step 3: final convex hull of reconstructed vertices
    poly = ConvexHull(verts)
    return verts, poly.simplices


def fix_winding(vertices, simplices):
    """Ensure all triangle normals point outward from the centroid."""
    centroid = vertices.mean(axis=0)
    faces = []
    for f in simplices:
        v0, v1, v2 = vertices[f]
        n   = np.cross(v1 - v0, v2 - v0)
        mid = (v0 + v1 + v2) / 3.0
        if np.dot(n, mid - centroid) < 0:
            faces.append([f[0], f[2], f[1]])
        else:
            faces.append(list(f))
    return faces


def plot_3d(vertices, faces, areas_per_face, save_path, title="Asteroid Shape Model (convexinv)"):
    """
    3D plot with faces coloured by local area (proxy for surface roughness).
    areas_per_face: one value per face (mean of constituent vertex areas).
    """
    from matplotlib.colors import Normalize

    fig = plt.figure(figsize=(10, 8))
    ax  = fig.add_subplot(111, projection='3d')

    norm = Normalize(vmin=np.percentile(areas_per_face, 5),
                     vmax=np.percentile(areas_per_face, 95))
    # matplotlib.colormaps replaces the removed cm.get_cmap() in matplotlib 3.9+
    cmap   = matplotlib.colormaps.get_cmap('YlOrBr')
    colors = [cmap(norm(a)) for a in areas_per_face]

    polys = [[vertices[i] for i in face] for face in faces]
    mesh  = Poly3DCollection(polys, shade=False, edgecolor='k',
                             linewidth=0.05, alpha=0.90)
    mesh.set_facecolor(colors)
    ax.add_collection3d(mesh)

    lo, hi = vertices.min(axis=0), vertices.max(axis=0)
    c = (lo + hi) / 2
    r = (hi - lo).max() / 2 * 1.15
    ax.set_xlim(c[0]-r, c[0]+r)
    ax.set_ylim(c[1]-r, c[1]+r)
    ax.set_zlim(c[2]-r, c[2]+r)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    ax.set_box_aspect([1, 1, 1])

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.1, label='Face area (relative)')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved plot   -> {save_path}")
    plt.close()


def run(shape_file=None, output_dir=None):
    sf  = Path(shape_file)  if shape_file  else SHAPE_FILE
    out = Path(output_dir)  if output_dir  else OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)

    # Read optional params file for the plot title
    params_path = sf.parent / "params_result.out"
    title = "Asteroid Shape Model (convexinv)"
    if params_path.exists():
        with open(params_path) as pf:
            line = pf.readline().split()
            if len(line) >= 3:
                lam, bet, per = line[0], line[1], line[2]
                title = f"Asteroid Shape Model  (λ={lam}°, β={bet}°, P={per} h)"

    print(f"Reading: {sf}")
    normals, areas = read_gaussian_image(sf)
    print(f"  {len(normals)} facets in Gaussian image")
    print(f"  total surface area = {areas.sum():.2f}  (convexinv units)")
    print(f"  equivalent radius  = {np.sqrt(areas.sum() / (4*np.pi)):.3f}")

    print("Reconstructing 3D mesh via dual-polytope method...")
    verts, simplices = reconstruct_polyhedron(normals, areas)
    faces = fix_winding(verts, simplices)
    print(f"  {len(verts)} vertices, {len(faces)} faces")

    R_equiv = np.sqrt(areas.sum() / (4 * np.pi))
    verts_scaled = verts * R_equiv

    write_obj(out / "asteroid.obj", verts_scaled, faces)
    write_ply(out / "asteroid.ply", verts_scaled, faces)
    write_msh(out / "asteroid.msh", verts_scaled, faces)

    face_areas = np.array([
        areas[np.argmax(normals @ (verts[f].mean(axis=0) /
                                   np.linalg.norm(verts[f].mean(axis=0))))]
        for f in faces
    ])
    plot_3d(verts_scaled, faces, face_areas, out / "asteroid_shape.png", title=title)


def main():
    run()

if __name__ == "__main__":
    main()
