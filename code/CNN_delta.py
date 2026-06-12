"""
CNN_delta.py

Delta CNN: learns the voxel-space correction (delta) between the convexinv
simulation shape and the true asteroid shape.

  Input:  x_lc  (B, 360, 56)    real + physics-predicted lightcurves
          x_sim  (B, 32, 32, 16) voxelised simulation/Asteroid0X/asteroid.obj
  Output: delta  (B, 32, 32, 16) additive correction for the sim voxel grid

  Prediction:
      corrected_vox = clip(sim_vox + delta_pred, 0, 1) > 0.5
      → largest connected component → marching cubes → STL

Usage:
    python CNN_delta.py train
    python CNN_delta.py predict <sim_dir>
    python CNN_delta.py predict all
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

_CODE_DIR = Path(__file__).resolve().parent
_BASE     = _CODE_DIR.parent
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from train_config import (
    training_catalog, VOX_SHAPE, n_aug, n_epochs, batch_size,
    lr, weight_decay, noise_std, device,
    data_dir, sim_dir, pred_dir, CYLINDER_RADII,
    load_voxel_target, _rotate_voxel_z,
    read_obj_vertices,
    _plot_loss, _plot_result,
)
from shape_to_mesh import write_obj, write_ply, write_msh, write_stl, normalize_mesh
from msh_to_npy import read_msh as _read_msh_file

model_path = _CODE_DIR / "models" / "delta_shape_model.pth"


# ---------------------------------------------------------------------------
# Sim-voxel helper
# ---------------------------------------------------------------------------

def voxelize_obj(obj_path, vox_shape=VOX_SHAPE):
    """
    Read simulation asteroid.obj, normalize to contest bounding box,
    voxelise, and return float32 occupancy array of shape vox_shape.
    """
    import trimesh

    verts, faces = [], []
    with open(obj_path) as f:
        for line in f:
            tok = line.split()
            if not tok:
                continue
            if tok[0] == "v":
                verts.append([float(tok[1]), float(tok[2]), float(tok[3])])
            elif tok[0] == "f":
                faces.append([int(t.split("/")[0]) - 1 for t in tok[1:]])

    verts = np.array(verts, dtype=np.float64)
    faces = np.array(faces, dtype=np.int32)

    # Same normalisation as msh_to_npy.py so both grids share the same space
    verts -= verts.mean(axis=0)
    scale  = 0.95 / (np.abs(verts).max(axis=0) / np.array([2.0, 2.0, 1.0])).max()
    verts *= scale

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
    if not mesh.is_watertight:
        trimesh.repair.fill_holes(mesh)

    NX, NY, NZ = vox_shape
    vox = mesh.voxelized(pitch=4.0 / NX).fill()
    pts = vox.points

    V  = np.zeros((NX, NY, NZ), dtype=np.float32)
    ix = np.floor((pts[:, 0] + 2.0) * NX / 4.0).astype(int)
    iy = np.floor((pts[:, 1] + 2.0) * NY / 4.0).astype(int)
    iz = np.floor((pts[:, 2] + 1.0) * NZ / 2.0).astype(int)
    ok = (ix >= 0) & (ix < NX) & (iy >= 0) & (iy < NY) & (iz >= 0) & (iz < NZ)
    V[ix[ok], iy[ok], iz[ok]] = 1.0
    return V


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def build_delta_samples(lc_obs, lc_pred, true_vox, sim_vox, n_aug=n_aug):
    """
    Build augmented (x_lc, x_sim, delta) triples.

    delta = rotate(true_vox) - rotate(sim_vox)
    Both shapes are rotated by the same angle that corresponds to the LC phase shift,
    so (LC, sim, delta) remain consistent across augmentations.
    """
    n_steps = lc_obs.shape[0]
    shifts  = np.linspace(0, n_steps, n_aug, endpoint=False, dtype=int)
    samples = []
    for k in shifts:
        obs_k  = np.roll(lc_obs,  k, axis=0)
        pred_k = np.roll(lc_pred, k, axis=0)
        x_lc   = np.concatenate([obs_k, pred_k], axis=1).astype(np.float32)

        angle  = k * 360.0 / n_steps
        true_k = _rotate_voxel_z(true_vox, angle)
        sim_k  = _rotate_voxel_z(sim_vox,  angle)
        delta  = (true_k - sim_k).astype(np.float32)   # values in {-1, 0, +1}

        samples.append((x_lc, sim_k, delta))
    return samples


def make_delta_dataset(samples):
    X_lc  = torch.from_numpy(np.stack([s[0] for s in samples]))   # (N, T, C)
    X_sim = torch.from_numpy(np.stack([s[1] for s in samples]))   # (N, VX, VY, VZ)
    Y     = torch.from_numpy(np.stack([s[2] for s in samples]))   # (N, VX, VY, VZ)
    return TensorDataset(X_lc, X_sim, Y)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class DeltaCNN(nn.Module):
    """
    1D LC encoder + 3D sim encoder → fused latent → 3D transposed-conv decoder → delta.
    """

    def __init__(self, n_lc_channels, vox_shape=VOX_SHAPE):
        super().__init__()
        self.vox_shape = vox_shape

        def conv1d_blk(cin, cout, k):
            return nn.Sequential(
                nn.Conv1d(cin, cout, k, padding=k // 2, bias=False),
                nn.BatchNorm1d(cout), nn.GELU(), nn.MaxPool1d(2),
            )

        self.lc_enc = nn.Sequential(
            conv1d_blk(n_lc_channels, 32, 9),
            conv1d_blk(32, 64, 7),
            conv1d_blk(64, 128, 5),
            nn.AdaptiveAvgPool1d(4),
        )
        # → (B, 128, 4) → flatten → (B, 512)

        def conv3d_blk(cin, cout):
            return nn.Sequential(
                nn.Conv3d(cin, cout, 3, padding=1, bias=False),
                nn.BatchNorm3d(cout), nn.GELU(),
                nn.MaxPool3d(2),
            )

        self.sim_enc = nn.Sequential(
            conv3d_blk(1,   16),    # (B, 16, 16, 16,  8)
            conv3d_blk(16,  32),    # (B, 32,  8,  8,  4)
            conv3d_blk(32,  64),    # (B, 64,  4,  4,  2)
            conv3d_blk(64, 128),    # (B,128,  2,  2,  1)
        )
        # → flatten → (B, 512)

        _seed_ch = 128
        self.fc = nn.Sequential(
            nn.Linear(512 + 512, 512), nn.GELU(), nn.Dropout(0.4),
            nn.Linear(512, _seed_ch * 2 * 2 * 1), nn.GELU(),
        )

        def deconv_blk(cin, cout):
            return nn.Sequential(
                nn.ConvTranspose3d(cin, cout, 4, stride=2, padding=1, bias=False),
                nn.BatchNorm3d(cout), nn.GELU(),
            )

        self.decoder = nn.Sequential(
            deconv_blk(_seed_ch, 64),               # → (B, 64,  4,  4,  2)
            deconv_blk(64, 32),                     # → (B, 32,  8,  8,  4)
            deconv_blk(32, 16),                     # → (B, 16, 16, 16,  8)
            nn.ConvTranspose3d(16, 1, 4, stride=2, padding=1),  # → (B,  1, 32, 32, 16)
        )

    def forward(self, x_lc, x_sim):
        lc_feat  = self.lc_enc(x_lc.permute(0, 2, 1)).flatten(1)    # (B, 512)
        sim_feat = self.sim_enc(x_sim.unsqueeze(1)).flatten(1)        # (B, 512)
        x = self.fc(torch.cat([lc_feat, sim_feat], dim=1))
        x = x.view(x.size(0), 128, 2, 2, 1)
        return self.decoder(x).squeeze(1)                             # (B, VX, VY, VZ)


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def _read_obj_mesh(path):
    """Read .obj → (verts float64, faces int32) without any normalisation."""
    verts, faces = [], []
    with open(path) as f:
        for line in f:
            tok = line.split()
            if not tok:
                continue
            if tok[0] == "v":
                verts.append([float(tok[1]), float(tok[2]), float(tok[3])])
            elif tok[0] == "f":
                faces.append([int(t.split("/")[0]) - 1 for t in tok[1:]])
    return np.array(verts), np.array(faces, dtype=np.int32)


def _normalize_verts(verts):
    """Centre + uniform scale to 95% of contest bounding box (matches msh_to_npy.py)."""
    verts = verts - verts.mean(axis=0)
    scale = 0.95 / (np.abs(verts).max(axis=0) / np.array([2.0, 2.0, 1.0])).max()
    return verts * scale


def _get_true_mesh(ast_id):
    """
    Return (verts, faces) of the reference .msh for the given asteroid ID (e.g. '01').
    Returns (None, None) if not found (e.g. Asteroid04).
    """
    info = training_catalog.get(f"Asteroid{ast_id}")
    if info is None:
        return None, None
    msh_path = info.get("true_msh")
    if msh_path is None or not Path(msh_path).exists():
        return None, None
    try:
        verts, faces = _read_msh_file(Path(msh_path))
        if len(faces) == 0:
            return None, None
        return _normalize_verts(verts), faces
    except Exception:
        return None, None


def _vox_rmse(a, b):
    """RMSE between two binary / float32 voxel grids."""
    a = (np.asarray(a) > 0.5).astype(np.float32)
    b = (np.asarray(b) > 0.5).astype(np.float32)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def _plot_triple_png(meshes, save_path, title="", error_labels=None):
    """
    Static 3-panel matplotlib figure.
    meshes:       list of (label, verts, faces)
    error_labels: optional list of error strings, one per panel
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    n      = len(meshes)
    colors = ["lightyellow", "lightblue", "lightgreen", "lightsalmon"]
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 6),
                             subplot_kw={"projection": "3d"})
    if n == 1:
        axes = [axes]

    for idx, (ax, (label, verts, faces), color) in enumerate(
            zip(axes, meshes, colors)):
        polys = [[verts[i] for i in f] for f in faces]
        col   = Poly3DCollection(polys, alpha=0.85, facecolor=color,
                                 edgecolor="k", linewidth=0.05)
        ax.add_collection3d(col)
        lo, hi = verts.min(0), verts.max(0)
        c, r   = (lo + hi) / 2, (hi - lo).max() / 2 * 1.2
        ax.set_xlim(c[0]-r, c[0]+r)
        ax.set_ylim(c[1]-r, c[1]+r)
        ax.set_zlim(c[2]-r, c[2]+r)
        ax.set_box_aspect([1, 1, 1])
        err_str = f"\n{error_labels[idx]}" if error_labels and idx < len(error_labels) else ""
        ax.set_title(f"{label}{err_str}", fontsize=9)
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")

    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  comparison png:  {save_path}")


def _plot_triple_html(meshes, save_path, title="", lc_obs=None, lc_pred=None,
                      error_labels=None):
    """
    Interactive Plotly HTML.
    4 meshes → 2×2 grid.  2-3 meshes → 1 row.
    Optional bottom row: observed vs predicted lightcurves.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    n           = len(meshes)
    colorscales = ["Viridis", "Blues", "Plasma", "Oranges"]
    has_lc      = lc_obs is not None and lc_pred is not None

    # 4 panels → 2×2;  otherwise 1 row
    use_2x2 = (n == 4)
    mesh_rows = 2 if use_2x2 else 1
    mesh_cols = 2 if use_2x2 else n

    def _panel_title(idx, label):
        if error_labels and idx < len(error_labels):
            return f"{label}<br><sup>{error_labels[idx]}</sup>"
        return label

    panel_titles = [_panel_title(i, m[0]) for i, m in enumerate(meshes)]

    total_rows = mesh_rows + (1 if has_lc else 0)
    lc_row     = mesh_rows + 1

    mesh_specs = [[{"type": "scene"}, {"type": "scene"}]] * mesh_rows \
                 if use_2x2 else \
                 [[{"type": "scene"}] * mesh_cols]
    lc_specs   = [[{"type": "xy", "colspan": mesh_cols}] + [None] * (mesh_cols - 1)]

    specs = mesh_specs + (lc_specs if has_lc else [])
    row_heights = ([0.45, 0.45] if use_2x2 else [0.62]) + ([0.38 if use_2x2 else 0.38] if has_lc else [])
    # normalise
    s = sum(row_heights); row_heights = [h / s for h in row_heights]

    subplot_titles = panel_titles + (["Observed vs Predicted Lightcurves"] if has_lc else [])

    fig = make_subplots(
        rows=total_rows, cols=mesh_cols,
        specs=specs,
        subplot_titles=subplot_titles,
        row_heights=row_heights,
        horizontal_spacing=0.04,
        vertical_spacing=0.08,
    )

    # --- 3D mesh panels ---
    for idx, ((label, verts, faces), cscale) in enumerate(zip(meshes, colorscales)):
        r = idx // mesh_cols + 1 if use_2x2 else 1
        c = idx %  mesh_cols + 1
        faces = np.asarray(faces)
        fig.add_trace(
            go.Mesh3d(
                x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
                i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                intensity=verts[:, 2],
                colorscale=cscale,
                showscale=False,
                lighting=dict(ambient=0.5, diffuse=0.8, specular=0.2, roughness=0.5),
                lightposition=dict(x=1, y=1, z=2),
                hovertemplate="x:%{x:.3f} y:%{y:.3f} z:%{z:.3f}<extra></extra>",
                name=label,
            ),
            row=r, col=c,
        )

    # --- Lightcurve panel ---
    if has_lc:
        t = np.arange(len(lc_obs))
        for ch in range(lc_obs.shape[1]):
            fig.add_trace(go.Scatter(
                x=t, y=lc_obs[:, ch],
                mode="lines", line=dict(color="rgba(0,100,200,0.15)", width=0.7),
                showlegend=False, hoverinfo="skip",
            ), row=lc_row, col=1)
        for ch in range(lc_pred.shape[1]):
            fig.add_trace(go.Scatter(
                x=t, y=lc_pred[:, ch],
                mode="lines", line=dict(color="rgba(220,60,60,0.15)", width=0.7),
                showlegend=False, hoverinfo="skip",
            ), row=lc_row, col=1)
        fig.add_trace(go.Scatter(
            x=t, y=lc_obs.mean(axis=1),
            mode="lines", line=dict(color="rgb(0,80,180)", width=2.5),
            name="Observed (mean)",
        ), row=lc_row, col=1)
        fig.add_trace(go.Scatter(
            x=t, y=lc_pred.mean(axis=1),
            mode="lines", line=dict(color="rgb(200,40,40)", width=2.5),
            name="Predicted (mean)",
        ), row=lc_row, col=1)
        fig.update_xaxes(title_text="Time step", row=lc_row, col=1,
                         showgrid=True, gridcolor="#e0e0e0")
        fig.update_yaxes(title_text="Normalised intensity", row=lc_row, col=1,
                         showgrid=True, gridcolor="#e0e0e0")

    # --- Layout ---
    axis_cfg   = dict(showgrid=True, showbackground=True, backgroundcolor="#f0f0f0")
    scene_base = dict(xaxis=axis_cfg, yaxis=axis_cfg, zaxis=axis_cfg, aspectmode="data")

    layout_kw = {"title": dict(text=title, x=0.5, xanchor="center", font=dict(size=14))}
    scene_idx = 1
    for idx in range(n):
        key = "scene" if scene_idx == 1 else f"scene{scene_idx}"
        layout_kw[key] = scene_base
        scene_idx += 1

    total_height = 1100 if (use_2x2 and has_lc) else (900 if use_2x2 or has_lc else 650)
    fig.update_layout(
        height=total_height,
        paper_bgcolor="white",
        plot_bgcolor="white",
        legend=dict(x=1.01, y=0.1),
        margin=dict(l=5, r=80, t=60, b=5),
        **layout_kw,
    )

    html_path = Path(save_path).with_suffix(".html")
    fig.write_html(str(html_path), include_plotlyjs=True, full_html=True)
    print(f"  comparison html: {html_path}")


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

def train(model_path=model_path):
    print(f"device: {device}")
    print("loading training data...")

    all_samples = []
    for name, info in training_catalog.items():
        obj_path  = info["sim_dir"] / "asteroid.obj"
        pred_path = info["sim_dir"] / "predicted_lightcurve.txt"
        for p in (info["true_npy"], info["lc_real"], obj_path, pred_path):
            if not Path(p).exists():
                print(f"  {name}: missing {p}, skipping")
                break
        else:
            print(f"  {name}: voxelising ...", end=" ", flush=True)
            true_vox = load_voxel_target(info["true_npy"])     # (VX,VY,VZ) float32
            sim_vox  = voxelize_obj(obj_path)                  # (VX,VY,VZ) float32

            lc_obs  = np.loadtxt(info["lc_real"], delimiter=",")[:, 1:]
            lc_pred = np.loadtxt(pred_path,       delimiter=",")[:, 1:]
            n = min(len(lc_obs), len(lc_pred))
            lc_obs, lc_pred = lc_obs[:n], lc_pred[:n]

            samples = build_delta_samples(lc_obs, lc_pred, true_vox, sim_vox, n_aug=n_aug)
            all_samples.extend(samples)
            print(f"{len(samples)} augmented samples")

    if not all_samples:
        print("no training data found")
        return

    print(f"\n{len(all_samples)} samples total")

    rng   = np.random.default_rng(42)
    idx   = rng.permutation(len(all_samples))
    split = int(0.85 * len(idx))
    train_s = [all_samples[i] for i in idx[:split]]
    val_s   = [all_samples[i] for i in idx[split:]]

    train_loader = DataLoader(make_delta_dataset(train_s),
                              batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(make_delta_dataset(val_s),
                              batch_size=batch_size, shuffle=False, num_workers=0)

    n_lc_channels = all_samples[0][0].shape[1]
    model    = DeltaCNN(n_lc_channels=n_lc_channels).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{n_params:,} parameters  (n_lc_channels={n_lc_channels})")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    criterion = nn.MSELoss()

    best_val  = float("inf")
    history   = {"train": [], "val": []}

    print(f"\ntraining for {n_epochs} epochs...")
    for epoch in range(1, n_epochs + 1):
        model.train()
        train_loss = 0.0
        for x_lc, x_sim, y_delta in train_loader:
            x_lc = x_lc + torch.randn_like(x_lc) * noise_std
            x_lc, x_sim, y_delta = x_lc.to(device), x_sim.to(device), y_delta.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x_lc, x_sim), y_delta)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * len(x_lc)
        train_loss /= len(train_s)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_lc, x_sim, y_delta in val_loader:
                x_lc, x_sim, y_delta = x_lc.to(device), x_sim.to(device), y_delta.to(device)
                val_loss += criterion(model(x_lc, x_sim), y_delta).item() * len(x_lc)
        val_loss /= len(val_s)

        scheduler.step()
        history["train"].append(train_loss)
        history["val"].append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            Path(model_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model_state":   model.state_dict(),
                        "n_lc_channels": n_lc_channels,
                        "vox_shape":     list(VOX_SHAPE)}, model_path)

        if epoch % 50 == 0 or epoch == 1:
            print(f"  epoch {epoch:4d}/{n_epochs}  "
                  f"train={train_loss:.5f}  val={val_loss:.5f}  "
                  f"best_val={best_val:.5f}  "
                  f"lr={scheduler.get_last_lr()[0]:.2e}")

    print(f"\nbest val loss: {best_val:.5f}")
    print(f"saved to {model_path}")
    _plot_loss(history, Path(model_path).parent / "delta_training_loss.png")
    return model_path


# ---------------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------------

def _vox_to_mesh(V_binary, vox_shape):
    try:
        from skimage.measure import marching_cubes
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install",
                               "scikit-image", "-q"])
        from skimage.measure import marching_cubes

    from scipy.ndimage import label as _label, binary_closing, binary_fill_holes

    NX, NY, NZ = vox_shape

    # Fill enclosed 2D holes in each XY slice, then smooth with 3D closing.
    V = V_binary.astype(bool)
    for z in range(NZ):
        V[:, :, z] = binary_fill_holes(V[:, :, z])
    V = binary_closing(V, structure=np.ones((3, 3, 3), dtype=bool))
    V_binary = V.astype(np.uint8)

    labels, _ = _label(V_binary)
    if labels.max() > 0:
        sizes   = np.bincount(labels.ravel())
        largest = sizes[1:].argmax() + 1
        V_binary = (labels == largest).astype(np.uint8)

    dx, dy, dz = 4.0 / NX, 4.0 / NY, 2.0 / NZ

    V_pad = np.pad(V_binary, 1, mode="constant", constant_values=0)
    verts, faces, _, _ = marching_cubes(V_pad.astype(float), level=0.5)
    verts[:, 0] = -2.0 + (verts[:, 0] - 1) * dx
    verts[:, 1] = -2.0 + (verts[:, 1] - 1) * dy
    verts[:, 2] = -1.0 + (verts[:, 2] - 1) * dz
    return verts, faces


def predict(sim_dir, model_path=model_path, out_root=None, lc_obs_path=None, out_dir=None):
    sim_dir  = Path(sim_dir)
    _out_root = Path(out_root) if out_root else _BASE / "prediction"
    out_dir  = Path(out_dir) if out_dir else _out_root / sim_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)

    if not Path(model_path).exists():
        print(f"model not found at {model_path}, run train first")
        return None

    ckpt      = torch.load(model_path, map_location=device, weights_only=False)
    vox_shape = tuple(ckpt["vox_shape"])
    model     = DeltaCNN(n_lc_channels=ckpt["n_lc_channels"],
                         vox_shape=vox_shape).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    obj_path  = sim_dir / "asteroid.obj"
    pred_path = sim_dir / "predicted_lightcurve.txt"
    if not obj_path.exists() or not pred_path.exists():
        print(f"missing asteroid.obj or predicted_lightcurve.txt in {sim_dir}")
        return None

    if lc_obs_path is not None:
        lc_file = Path(lc_obs_path)
    else:
        ast_id  = sim_dir.name.replace("Asteroid", "")
        matches = sorted(data_dir.glob(
            f"AsteroidModel{ast_id}_*/**/Asteroid{ast_id}_lightcurve_intensity.txt"))
        if not matches:
            matches = sorted(data_dir.glob(
                f"AsteroidModel{ast_id}_*/**/Asteroid{ast_id}_lightcurve_intensity_blender.txt"))
        if not matches:
            print(f"no lightcurve data found for Asteroid{ast_id}")
            return None
        lc_file = matches[0]

    lc_obs  = np.loadtxt(lc_file,   delimiter=",")[:, 1:]
    lc_pred = np.loadtxt(pred_path,  delimiter=",")[:, 1:]
    n       = min(len(lc_obs), len(lc_pred))
    lc_obs, lc_pred = lc_obs[:n], lc_pred[:n]

    sim_vox = voxelize_obj(obj_path, vox_shape)   # float32 (VX,VY,VZ)

    x_lc  = torch.from_numpy(
        np.concatenate([lc_obs, lc_pred], axis=1).astype(np.float32)
    ).unsqueeze(0).to(device)
    x_sim = torch.from_numpy(sim_vox).unsqueeze(0).to(device)

    with torch.no_grad():
        delta_pred = model(x_lc, x_sim).squeeze(0).cpu().numpy()

    # Apply correction: sim + delta, clamp to [0,1], threshold
    corrected_float = np.clip(sim_vox + delta_pred, 0.0, 1.0)
    V_corrected     = (corrected_float > 0.5).astype(np.uint8)

    try:
        verts, faces = _vox_to_mesh(V_corrected, vox_shape)
    except Exception as e:
        print(f"  mesh reconstruction failed: {e}")
        return None

    print(f"  {sim_dir.name}: {len(verts)} vertices, {len(faces)} faces")

    ast_id     = sim_dir.name.replace("Asteroid", "").zfill(2)
    cyl_R      = CYLINDER_RADII.get(ast_id)
    verts_norm = normalize_mesh(verts, cylinder_R=cyl_R)

    write_obj(out_dir / "delta_asteroid.obj", verts_norm, faces)
    write_ply(out_dir / "delta_asteroid.ply", verts_norm, faces)
    write_msh(out_dir / "delta_asteroid.msh", verts_norm, faces)
    stl_out = out_dir / "delta_asteroid.stl"
    write_stl(stl_out, verts_norm, faces)

    # --- Error metrics ---
    lc_rmse    = float(np.sqrt(np.mean((lc_obs.mean(1) - lc_pred.mean(1)) ** 2)))
    delta_mean = float(np.abs(delta_pred).mean())
    delta_max  = float(np.abs(delta_pred).max())

    # Dice coefficients vs true shape (available for Asteroid01-03)
    true_vox = None
    true_info = training_catalog.get(f"Asteroid{ast_id}")
    if true_info and Path(true_info["true_npy"]).exists():
        true_vox = load_voxel_target(true_info["true_npy"])

    def _rmse_str(pred_vox):
        if true_vox is None:
            return ""
        return f"  RMSE: {_vox_rmse(pred_vox, true_vox):.4f}"

    # --- Build mesh panels ---
    phys_verts, phys_faces = _read_obj_mesh(obj_path)
    phys_vox = voxelize_obj(obj_path, vox_shape)

    meshes       = [("Physics", phys_verts, phys_faces)]
    error_labels = [f"LC RMSE: {lc_rmse:.4f}{_rmse_str(phys_vox)}"]

    cnn_obj = out_dir / "cnn_asteroid.obj"
    if cnn_obj.exists():
        cnn_verts, cnn_faces = _read_obj_mesh(cnn_obj)
        cnn_vox = voxelize_obj(cnn_obj, vox_shape)
        meshes.append(("VoxelCNN", cnn_verts, cnn_faces))
        error_labels.append(_rmse_str(cnn_vox).strip() or f"{len(cnn_verts)} verts")

    meshes.append(("DeltaCNN", verts_norm, np.asarray(faces)))
    error_labels.append(_rmse_str(V_corrected).strip() or "")

    # True reference shape (4th panel, Asteroid01-03 only)
    true_verts, true_faces = _get_true_mesh(ast_id)
    if true_verts is not None:
        meshes.append(("True Shape", true_verts, true_faces))
        error_labels.append("reference")

    _title = f"Shape Comparison  —  {sim_dir.name}"
    _plot_triple_png(meshes, out_dir / "delta_shape_comparison.png",
                     title=_title, error_labels=error_labels)
    _plot_triple_html(meshes, out_dir / "delta_shape_comparison.html",
                      title=_title, lc_obs=lc_obs, lc_pred=lc_pred,
                      error_labels=error_labels)

    print(f"  saved to {out_dir}")
    return stl_out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Delta CNN asteroid shape correction")
    sub    = parser.add_subparsers(dest="cmd")

    sub.add_parser("train", help="Train DeltaCNN on Asteroid01-03")

    p = sub.add_parser("predict", help="Apply delta correction")
    p.add_argument("target",
                   help="Simulation directory path, or 'all' to process every folder")
    p.add_argument("--model", default=str(model_path))
    p.add_argument("--out",   default=str(_BASE / "prediction"))

    args = parser.parse_args()

    if args.cmd == "train":
        train()
    elif args.cmd == "predict":
        mp = args.model
        if args.target.lower() == "all":
            for d in sorted(Path(sim_dir).iterdir()):
                if d.is_dir() and (d / "asteroid.obj").exists():
                    predict(d, model_path=mp, out_root=args.out)
        else:
            predict(args.target, model_path=mp, out_root=args.out)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
