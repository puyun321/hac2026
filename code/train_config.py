import struct
import sys
from pathlib import Path
from scipy.spatial import ConvexHull

import numpy as np
import torch
from torch.utils.data import TensorDataset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import plotly.graph_objects as go
from plotly.subplots import make_subplots

_CODE_DIR = Path(__file__).resolve().parent
_BASE     = _CODE_DIR.parent
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

data_dir = _BASE / "data"
sim_dir  = _BASE / "simulation"
pred_dir = _BASE / "prediction"

CYLINDER_RADII = {
    "01": 1.12,
    "02": 1.42,
    "03": 0.88,
    "04": 1.475,
    "05": 1.22,
    "06": 0.925,
    "07": 1.205,
    "08": 1.24,
    "09": 0.67,
}

n_sphere_dirs = 256
n_aug         = 72
n_epochs      = 600
batch_size    = 32
lr            = 3e-4
weight_decay  = 1e-3
noise_std     = 0.005
device        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_NPY_DIR = _BASE / "npy_data"

training_catalog = {
    "Asteroid01": {
        "true_msh":       data_dir / "AsteroidModel01_shape_public" / "asteroid1_vesta_nasa_real_units.msh",
        "true_npy":       _NPY_DIR / "AsteroidModel01_shape_public.npy",
        "sim_dir":        sim_dir  / "Asteroid01",
        "lc_blender":     data_dir / "AsteroidModel01_shape_public" / "Asteroid1_lightcurve_data"
                                   / "Asteroid01_lightcurve_intensity_blender.txt",
        "lc_bin_blender": data_dir / "AsteroidModel01_shape_public" / "Asteroid1_lightcurve_data"
                                   / "Asteroid01_lightcurve_binary_blender.txt",
        "lc_real":        data_dir / "AsteroidModel01_shape_public" / "Asteroid1_lightcurve_data"
                                   / "Asteroid01_lightcurve_intensity.txt",
        "lc_bin_real":    data_dir / "AsteroidModel01_shape_public" / "Asteroid1_lightcurve_data"
                                   / "Asteroid01_lightcurve_binary.txt",
        "cylinder_R":     CYLINDER_RADII["01"],
    },
    "Asteroid02": {
        "true_msh":       data_dir / "AsteroidModel02_shape_public" / "asteroid2_sawed_off_cube_real_units.msh",
        "true_npy":       _NPY_DIR / "AsteroidModel02_shape_public.npy",
        "sim_dir":        sim_dir  / "Asteroid02",
        "lc_blender":     data_dir / "AsteroidModel02_shape_public" / "Asteroid2_lightcurve_data"
                                   / "Asteroid02_lightcurve_intensity_blender.txt",
        "lc_bin_blender": data_dir / "AsteroidModel02_shape_public" / "Asteroid2_lightcurve_data"
                                   / "Asteroid02_lightcurve_binary_blender.txt",
        "lc_real":        data_dir / "AsteroidModel02_shape_public" / "Asteroid2_lightcurve_data"
                                   / "Asteroid02_lightcurve_intensity.txt",
        "lc_bin_real":    data_dir / "AsteroidModel02_shape_public" / "Asteroid2_lightcurve_data"
                                   / "Asteroid02_lightcurve_binary.txt",
        "cylinder_R":     CYLINDER_RADII["02"],
    },
    "Asteroid03": {
        "true_msh":       data_dir / "AsteroidModel03_shape_public" / "asteroid3_mithra_nasa_real_units.msh",
        "true_npy":       _NPY_DIR / "AsteroidModel03_shape_public.npy",
        "sim_dir":        sim_dir  / "Asteroid03",
        "lc_blender":     data_dir / "AsteroidModel03_shape_public" / "Asteroid3_lightcurve_data"
                                   / "Asteroid03_lightcurve_intensity_blender.txt",
        "lc_bin_blender": data_dir / "AsteroidModel03_shape_public" / "Asteroid3_lightcurve_data"
                                   / "Asteroid03_lightcurve_binary_blender.txt",
        "lc_real":        data_dir / "AsteroidModel03_shape_public" / "Asteroid3_lightcurve_data"
                                   / "Asteroid03_lightcurve_intensity.txt",
        "lc_bin_real":    data_dir / "AsteroidModel03_shape_public" / "Asteroid3_lightcurve_data"
                                   / "Asteroid03_lightcurve_binary.txt",
        "cylinder_R":     CYLINDER_RADII["03"],
    },
}


def read_msh_vertices(path):
    """Gmsh 2.2 binary: return (N, 3) vertex array."""
    with open(path, "rb") as f:
        content = f.read()
    idx     = content.find(b"$Nodes\n")
    nl      = content.find(b"\n", idx + 7)
    n_nodes = int(content[idx + 7 : nl])
    start   = nl + 1
    verts   = np.empty((n_nodes, 3), dtype=np.float64)
    for i in range(n_nodes):
        off        = start + i * 28
        _, x, y, z = struct.unpack("<i3d", content[off : off + 28])
        verts[i]   = (x, y, z)
    return verts

def read_npy_verts(npy_path):
    """
    Load a voxelised .npy grid (400×400×200 uint8) and return the convex-hull
    vertices of the occupied voxel centres as a (N, 3) float64 array.

    Using convex-hull vertices guarantees exact support-function values while
    keeping N small (hundreds, not millions).
    """
    from scipy.spatial import ConvexHull as _CH
    V = np.load(npy_path)                    # (400, 400, 200)
    ix, iy, iz = np.where(V > 0)
    pts = np.stack([
        -2.0 + (ix + 0.5) * 0.01,
        -2.0 + (iy + 0.5) * 0.01,
        -1.0 + (iz + 0.5) * 0.01,
    ], axis=1)                               # physical coordinates
    # Subsample before ConvexHull if necessary (QHull limit)
    if len(pts) > 60_000:
        rng = np.random.default_rng(0)
        pts = pts[rng.choice(len(pts), 60_000, replace=False)]
    hull = _CH(pts)
    return pts[hull.vertices].astype(np.float64)


def read_obj_vertices(path):
    verts = []
    with open(path) as f:
        for line in f:
            tok = line.split()
            if tok and tok[0] == "v":
                verts.append([float(tok[1]), float(tok[2]), float(tok[3])])
    return np.array(verts)


def fibonacci_sphere(n=n_sphere_dirs):
    golden = (1 + np.sqrt(5)) / 2
    i      = np.arange(n, dtype=float)
    theta  = np.arccos(np.clip(1 - 2 * (i + 0.5) / n, -1, 1))
    phi    = 2 * np.pi * i / golden
    return np.stack([np.sin(theta) * np.cos(phi),
                     np.sin(theta) * np.sin(phi),
                     np.cos(theta)], axis=1)


dirs = fibonacci_sphere(n_sphere_dirs)


def support_function(verts, dirs=dirs):
    """h(n_i) = max_v  dot(n_i, v).  Returns (N,)."""
    return (dirs @ verts.T).max(axis=1)


# Voxel-based training
VOX_SHAPE = (32, 32, 16)   # downsampled grid for CNN (pitch = 0.125 in each axis)


def load_voxel_target(npy_path, target=VOX_SHAPE):
    """Max-pool the 400×400×200 .npy down to VOX_SHAPE and return float32 array."""
    import torch
    import torch.nn.functional as F
    V = np.load(npy_path).astype(np.float32)
    t = torch.from_numpy(V).unsqueeze(0).unsqueeze(0)   # (1,1,H,W,D)
    t = F.adaptive_max_pool3d(t, target)
    return (t.squeeze().numpy() > 0.5).astype(np.float32)


def _rotate_voxel_z(V, angle_deg):
    """Rotate float32 voxel grid (X,Y,Z) around Z axis by angle_deg."""
    from scipy.ndimage import rotate as _rot
    V_r = _rot(V, angle=angle_deg, axes=(0, 1), reshape=False,
               order=1, mode='constant', cval=0.0)
    return (V_r > 0.5).astype(np.float32)


def build_voxel_samples(lc_obs, lc_pred, voxel_target, n_aug=n_aug):
    """Return list of (x_lc float32 (T,C), y_vox float32 (VX,VY,VZ)) pairs."""
    n_steps = lc_obs.shape[0]
    shifts  = np.linspace(0, n_steps, n_aug, endpoint=False, dtype=int)
    samples = []
    for k in shifts:
        obs_k  = np.roll(lc_obs,  k, axis=0)
        pred_k = np.roll(lc_pred, k, axis=0)
        x_lc   = np.concatenate([obs_k, pred_k], axis=1).astype(np.float32)
        angle  = k * 360.0 / n_steps
        V_k    = _rotate_voxel_z(voxel_target, angle)
        samples.append((x_lc, V_k))
    return samples


def make_voxel_dataset(samples):
    X_lc = torch.from_numpy(np.stack([s[0] for s in samples]))   # (N, T, C)
    Y    = torch.from_numpy(np.stack([s[1] for s in samples]))    # (N, VX, VY, VZ)
    return TensorDataset(X_lc, Y)


# Legacy support-function helpers (kept for reference / physics pipeline)
def rotate_verts_z(verts, angle_rad):
    c, s      = np.cos(angle_rad), np.sin(angle_rad)
    rot       = verts.copy()
    rot[:, 0] = verts[:, 0] * c - verts[:, 1] * s
    rot[:, 1] = verts[:, 0] * s + verts[:, 1] * c
    return rot


def make_dataset(samples):
    X_lc   = torch.from_numpy(np.stack([s[0] for s in samples]))
    X_hsim = torch.from_numpy(np.stack([s[1] for s in samples]))
    Y      = torch.from_numpy(np.stack([s[2] for s in samples]))
    return TensorDataset(X_lc, X_hsim, Y)


def build_samples(lc_obs, lc_pred, true_verts, sim_verts, n_aug=n_aug):
    n_steps = lc_obs.shape[0]
    shifts  = np.linspace(0, n_steps, n_aug, endpoint=False, dtype=int)
    samples = []
    for k in shifts:
        obs_k  = np.roll(lc_obs,  k, axis=0)
        pred_k = np.roll(lc_pred, k, axis=0)
        x_lc   = np.concatenate([obs_k, pred_k], axis=1).astype(np.float32)

        angle   = -k * 2 * np.pi / n_steps
        true_k  = rotate_verts_z(true_verts, angle)
        sim_k   = rotate_verts_z(sim_verts,  angle)

        h_true      = support_function(true_k)
        h_sim_k     = support_function(sim_k)
        h_true_norm = h_true  / np.median(h_true)
        h_sim_norm  = h_sim_k / np.median(h_sim_k)

        samples.append((x_lc, h_sim_norm.astype(np.float32), h_true_norm.astype(np.float32)))
    return samples


def reconstruct_from_support(h, dirs):
    hull  = ConvexHull(dirs)
    verts = []
    for tri in hull.simplices:
        A = dirs[tri]
        try:
            p = np.linalg.solve(A, h[tri])
            verts.append(p)
        except np.linalg.LinAlgError:
            pass
    verts    = np.array(verts)
    r        = np.linalg.norm(verts, axis=1)
    verts    = verts[r <= np.percentile(r, 95)]
    poly     = ConvexHull(verts)
    centroid = verts.mean(axis=0)
    faces = []
    for f in poly.simplices:
        v0, v1, v2 = verts[f]
        if np.dot(np.cross(v1-v0, v2-v0), (v0+v1+v2)/3 - centroid) < 0:
            faces.append([f[0], f[2], f[1]])
        else:
            faces.append(list(f))
    return verts, faces


def _plot_loss(history, save_path):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history["train"], label="train")
    ax.plot(history["val"],   label="val")
    ax.set_xlabel("Epoch"); ax.set_ylabel("MSE Loss")
    ax.set_title("Training Loss")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"loss curve: {save_path}")


def _plot_result(sim_verts, corr_verts, corr_faces, params_path, save_path, title_extra=""):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), subplot_kw={"projection": "3d"})
    label = ""
    if Path(params_path).exists():
        p = Path(params_path).read_text().split()
        if len(p) >= 3:
            label = f"λ={p[0]}°, β={p[1]}°, P={p[2]} h"

    for ax, verts, subtitle in zip(axes,
                                   [sim_verts, corr_verts],
                                   ["Physics model (simulated)", "Corrected"]):
        hull  = ConvexHull(verts)
        polys = [[verts[i] for i in f] for f in hull.simplices]
        mesh  = Poly3DCollection(polys, alpha=0.85, facecolor="lightyellow",
                                 edgecolor="k", linewidth=0.1)
        ax.add_collection3d(mesh)
        lo, hi = verts.min(0), verts.max(0)
        c, r   = (lo + hi) / 2, (hi - lo).max() / 2 * 1.2
        ax.set_xlim(c[0]-r, c[0]+r)
        ax.set_ylim(c[1]-r, c[1]+r)
        ax.set_zlim(c[2]-r, c[2]+r)
        ax.set_box_aspect([1, 1, 1])
        ax.set_title(subtitle, fontsize=9)
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")

    fig.suptitle(f"{title_extra}  {label}", fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def _plot_result_interactive(sim_verts, corr_verts, corr_faces,
                             h_sim, h_corr, dirs,
                             save_path, title_extra=""):
    h_sim_norm  = h_sim  / np.median(h_sim)
    h_corr_norm = h_corr / np.median(h_corr)

    def vert_colors(verts, values):
        norms = np.linalg.norm(verts, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        idx   = ((verts / norms) @ dirs.T).argmax(axis=1)
        return values[idx]

    sim_col  = vert_colors(sim_verts,  h_sim_norm)
    corr_col = vert_colors(corr_verts, h_corr_norm)

    from scipy.spatial import ConvexHull as _CH
    sim_hull  = _CH(sim_verts)
    sim_faces = [list(f) for f in sim_hull.simplices]

    def mesh_trace(verts, faces, vert_colors, colorscale, colorbar_title, showscale):
        fi = [f[0] for f in faces]
        fj = [f[1] for f in faces]
        fk = [f[2] for f in faces]
        return go.Mesh3d(
            x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
            i=fi, j=fj, k=fk,
            intensity=vert_colors.tolist(),
            intensitymode="vertex",
            colorscale=colorscale,
            showscale=showscale,
            colorbar=dict(len=0.45, thickness=14,
                          title=dict(text=colorbar_title, side="right")),
            lighting=dict(ambient=0.45, diffuse=0.8, specular=0.3, roughness=0.5),
            lightposition=dict(x=1, y=1, z=2),
            hovertemplate="x: %{x:.4f}<br>y: %{y:.4f}<br>z: %{z:.4f}<extra></extra>",
        )

    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "scene"}, {"type": "scene"}]],
        subplot_titles=[
            "Physics model  (convexinv + shape_to_mesh)",
            "Corrected shape",
        ],
        horizontal_spacing=0.05,
    )

    fig.add_trace(mesh_trace(sim_verts,  sim_faces,  sim_col,
                             "Viridis", "h(n) norm.",      showscale=True), row=1, col=1)
    fig.add_trace(mesh_trace(corr_verts, corr_faces, corr_col,
                             "Plasma",  "h_corr(n) norm.", showscale=True), row=1, col=2)

    axis_cfg   = dict(showgrid=True, gridcolor="#cccccc",
                      showbackground=True, backgroundcolor="#f0f0f0",
                      tickfont=dict(size=9))
    scene_base = dict(xaxis=axis_cfg, yaxis=axis_cfg, zaxis=axis_cfg, aspectmode="data")

    params_label = ""
    params_path  = Path(save_path).parent.parent / "params_result.out"
    if params_path.exists():
        p = params_path.read_text().split()
        if len(p) >= 3:
            params_label = f"  |  λ={p[0]}°  β={p[1]}°  P={p[2]} h"

    fig.update_layout(
        title=dict(
            text=(f"{title_extra}{params_label}<br>"
                  f"<sup>Left: {len(sim_verts)} verts / {len(sim_faces)} faces  ·  "
                  f"Right: {len(corr_verts)} verts / {len(corr_faces)} faces</sup>"),
            x=0.5, xanchor="center", font=dict(size=14),
        ),
        scene=scene_base,
        scene2=scene_base,
        paper_bgcolor="white",
        margin=dict(l=10, r=10, t=90, b=10),
        height=700,
    )

    html_path = Path(save_path).with_suffix(".html")
    fig.write_html(str(html_path), include_plotlyjs=True, full_html=True)
    print(f"  html: {html_path}")
