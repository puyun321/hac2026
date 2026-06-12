"""
Voxel CNN: lightcurve → 3D occupancy grid → STL

Usage:
  python CNN.py train
  python CNN.py predict <sim_dir>
  python CNN.py predict all
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

_CODE_DIR = Path(__file__).resolve().parent
_BASE     = _CODE_DIR.parent
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from train_config import (
    training_catalog, VOX_SHAPE, n_aug, n_epochs, batch_size,
    lr, weight_decay, noise_std, device,
    data_dir, sim_dir, pred_dir, CYLINDER_RADII,
    load_voxel_target, build_voxel_samples, make_voxel_dataset,
    read_obj_vertices,
    _plot_loss, _plot_result,
)
from shape_to_mesh import write_obj, write_ply, write_msh, write_stl, normalize_mesh

model_path = _CODE_DIR / "models" / "cnn_shape_model.pth"


class VoxelCNN(nn.Module):
    """
    1D LC encoder  →  3D transposed-conv decoder  →  voxel occupancy logits.

    Input:  x_lc   (B, T, C)         light-curve time series
    Output: logits  (B, VX, VY, VZ)  apply sigmoid for occupancy probability
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

        _seed_ch = 128
        self.fc = nn.Sequential(
            nn.Linear(512, 256), nn.GELU(), nn.Dropout(0.4),
            nn.Linear(256, _seed_ch * 2 * 2 * 1), nn.GELU(),
        )
        # → reshape → (B, 128, 2, 2, 1)

        def deconv_blk(cin, cout):
            return nn.Sequential(
                nn.ConvTranspose3d(cin, cout, 4, stride=2, padding=1, bias=False),
                nn.BatchNorm3d(cout), nn.GELU(),
            )

        self.decoder = nn.Sequential(
            deconv_blk(_seed_ch, 64),               # → (B, 64,  4,  4,  2)
            deconv_blk(64, 32),                     # → (B, 32,  8,  8,  4)
            deconv_blk(32, 16),                     # → (B, 16, 16, 16,  8)
            nn.ConvTranspose3d(16, 1, 4, stride=2, padding=1),  # → (B, 1, 32, 32, 16)
        )

    def forward(self, x_lc):
        x = self.lc_enc(x_lc.permute(0, 2, 1)).flatten(1)   # (B, 512)
        x = self.fc(x).view(x.size(0), 128, 2, 2, 1)
        return self.decoder(x).squeeze(1)                     # (B, VX, VY, VZ)


CNN = VoxelCNN   # alias

def train(model_path=model_path):
    print(f"device: {device}")
    print("loading training data...")

    all_samples = []
    for name, info in training_catalog.items():
        pred_path = info["sim_dir"] / "predicted_lightcurve.txt"
        for p in (info["true_npy"], info["lc_real"], pred_path):
            if not Path(p).exists():
                print(f"  {name}: missing {p}, skipping")
                break
        else:
            print(f"  {name}: loading voxel target ...", end=" ", flush=True)
            voxel_target = load_voxel_target(info["true_npy"])
            lc_obs  = np.loadtxt(info["lc_real"], delimiter=",")[:, 1:]
            lc_pred = np.loadtxt(pred_path,       delimiter=",")[:, 1:]
            n = min(len(lc_obs), len(lc_pred))
            lc_obs, lc_pred = lc_obs[:n], lc_pred[:n]

            samples = build_voxel_samples(lc_obs, lc_pred, voxel_target, n_aug=n_aug)
            all_samples.extend(samples)
            print(f"{len(samples)} augmented samples")

    if not all_samples:
        print("no training data found")
        return

    print(f"\n{len(all_samples)} samples total")

    rng   = np.random.default_rng(42)
    idx   = rng.permutation(len(all_samples))
    split = int(0.85 * len(idx))
    train_samples = [all_samples[i] for i in idx[:split]]
    val_samples   = [all_samples[i] for i in idx[split:]]

    train_ds     = make_voxel_dataset(train_samples)
    val_ds       = make_voxel_dataset(val_samples)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)

    n_lc_channels = all_samples[0][0].shape[1]
    model    = VoxelCNN(n_lc_channels=n_lc_channels).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{n_params:,} parameters  (n_lc_channels={n_lc_channels}, vox_shape={VOX_SHAPE})")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    criterion = nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")
    history       = {"train": [], "val": []}

    print(f"\ntraining for {n_epochs} epochs...")
    for epoch in range(1, n_epochs + 1):
        model.train()
        train_loss = 0.0
        for x_lc, y_vox in train_loader:
            x_lc = x_lc + torch.randn_like(x_lc) * noise_std
            x_lc, y_vox = x_lc.to(device), y_vox.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x_lc), y_vox)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * len(x_lc)
        train_loss /= len(train_ds)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_lc, y_vox in val_loader:
                x_lc, y_vox = x_lc.to(device), y_vox.to(device)
                val_loss += criterion(model(x_lc), y_vox).item() * len(x_lc)
        val_loss /= len(val_ds)

        scheduler.step()
        history["train"].append(train_loss)
        history["val"].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            Path(model_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model_state":    model.state_dict(),
                        "n_lc_channels":  n_lc_channels,
                        "vox_shape":      list(VOX_SHAPE)}, model_path)

        if epoch % 50 == 0 or epoch == 1:
            print(f"  epoch {epoch:4d}/{n_epochs}  "
                  f"train={train_loss:.5f}  val={val_loss:.5f}  "
                  f"best_val={best_val_loss:.5f}  "
                  f"lr={scheduler.get_last_lr()[0]:.2e}")

    print(f"\nbest val loss: {best_val_loss:.5f}")
    print(f"saved to {model_path}")
    _plot_loss(history, Path(model_path).parent / "cnn_training_loss.png")
    return model_path


# Predict

def _plot_interactive(sim_verts, cnn_verts, cnn_faces, save_path, title=""):
    """Save a side-by-side interactive Plotly HTML of two meshes."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from scipy.spatial import ConvexHull

    sim_hull  = ConvexHull(sim_verts)
    sim_faces = sim_hull.simplices.tolist()

    def trace(verts, faces, colorscale):
        fi, fj, fk = zip(*faces)
        return go.Mesh3d(
            x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
            i=fi, j=fj, k=fk,
            intensity=verts[:, 2],
            colorscale=colorscale,
            showscale=False,
            lighting=dict(ambient=0.5, diffuse=0.8, specular=0.2, roughness=0.5),
            lightposition=dict(x=1, y=1, z=2),
            hovertemplate="x:%{x:.3f} y:%{y:.3f} z:%{z:.3f}<extra></extra>",
        )

    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "scene"}, {"type": "scene"}]],
        subplot_titles=["Physics (convexinv)", "VoxelCNN"],
        horizontal_spacing=0.05,
    )
    fig.add_trace(trace(sim_verts, sim_faces, "Viridis"), row=1, col=1)
    fig.add_trace(trace(cnn_verts, cnn_faces, "Plasma"),  row=1, col=2)

    axis  = dict(showgrid=True, showbackground=True, backgroundcolor="#f0f0f0")
    scene = dict(xaxis=axis, yaxis=axis, zaxis=axis, aspectmode="data")
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center"),
        scene=scene, scene2=scene,
        height=680, paper_bgcolor="white",
        margin=dict(l=10, r=10, t=60, b=10),
    )

    html_path = Path(save_path).with_suffix(".html")
    fig.write_html(str(html_path), include_plotlyjs=True, full_html=True)
    print(f"  interactive: {html_path}")


def _vox_to_mesh(V_binary, vox_shape):
    """Marching cubes on a binary voxel grid → (verts, faces) in physical coords."""
    try:
        from skimage.measure import marching_cubes
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install",
                               "scikit-image", "-q"])
        from skimage.measure import marching_cubes

    from scipy.ndimage import label as _label, binary_closing, binary_fill_holes

    VX, VY, VZ = vox_shape

    # Fill enclosed 2D holes in each XY slice, then smooth with 3D closing.
    # This removes concave artifacts caused by sparse/noisy voxels at grid
    # boundaries (e.g. when the predicted shape is clipped at z=0 or z=VZ-1).
    V = V_binary.astype(bool)
    for z in range(VZ):
        V[:, :, z] = binary_fill_holes(V[:, :, z])
    V = binary_closing(V, structure=np.ones((3, 3, 3), dtype=bool))
    V_binary = V.astype(np.uint8)

    # Keep largest connected component to remove stray voxels
    labels, _ = _label(V_binary)
    if labels.max() > 0:
        sizes   = np.bincount(labels.ravel())
        largest = sizes[1:].argmax() + 1
        V_binary = (labels == largest).astype(np.uint8)

    dx, dy, dz  = 4.0 / VX, 4.0 / VY, 2.0 / VZ

    # Pad with one layer of zeros so marching cubes closes the surface
    # at the grid boundary instead of leaving holes.
    V_padded = np.pad(V_binary, pad_width=1, mode='constant', constant_values=0)
    verts, faces, _, _ = marching_cubes(V_padded.astype(float), level=0.5)
    # Subtract padding offset before converting to physical coordinates
    verts[:, 0] = -2.0 + (verts[:, 0] - 1) * dx
    verts[:, 1] = -2.0 + (verts[:, 1] - 1) * dy
    verts[:, 2] = -1.0 + (verts[:, 2] - 1) * dz
    return verts, faces


def predict(sim_dir, model_path=model_path, out_root=pred_dir,
            lc_obs_path=None, out_dir=None):
    sim_dir = Path(sim_dir)
    out_dir = Path(out_dir) if out_dir else Path(out_root) / sim_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)

    if not Path(model_path).exists():
        print(f"model not found at {model_path}, run train first")
        return None

    ckpt      = torch.load(model_path, map_location=device, weights_only=False)
    vox_shape = tuple(ckpt["vox_shape"])
    model     = VoxelCNN(n_lc_channels=ckpt["n_lc_channels"],
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

    x_lc = torch.from_numpy(
        np.concatenate([lc_obs, lc_pred], axis=1).astype(np.float32)
    ).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x_lc).squeeze(0).cpu().numpy()   # (VX, VY, VZ)

    prob     = 1.0 / (1.0 + np.exp(-logits))            # sigmoid
    V_pred   = (prob > 0.5).astype(np.uint8)

    try:
        verts, faces = _vox_to_mesh(V_pred, vox_shape)
    except Exception as e:
        print(f"  mesh reconstruction failed: {e}")
        return None

    print(f"  {sim_dir.name}: {len(verts)} vertices, {len(faces)} faces")

    ast_id     = sim_dir.name.replace("Asteroid", "").zfill(2)
    cyl_R      = CYLINDER_RADII.get(ast_id)
    verts_norm = normalize_mesh(verts, cylinder_R=cyl_R)

    write_obj(out_dir / "cnn_asteroid.obj", verts_norm, faces)
    write_ply(out_dir / "cnn_asteroid.ply", verts_norm, faces)
    write_msh(out_dir / "cnn_asteroid.msh", verts_norm, faces)
    stl_out = out_dir / "cnn_asteroid.stl"
    write_stl(stl_out, verts_norm, faces)

    sim_verts = read_obj_vertices(obj_path)
    _plot_result(sim_verts, verts_norm, faces,
                 sim_dir / "params_result.out",
                 out_dir / "cnn_shape_comparison.png",
                 title_extra=f"VoxelCNN  {sim_dir.name}")
    _plot_interactive(sim_verts, verts_norm, faces,
                      out_dir / "cnn_shape_comparison.html",
                      title=f"VoxelCNN  {sim_dir.name}")

    print(f"  saved to {out_dir}")
    return stl_out


# CLI
def main():
    parser = argparse.ArgumentParser(description="Voxel CNN asteroid shape reconstruction")
    sub    = parser.add_subparsers(dest="cmd")

    sub.add_parser("train", help="Train VoxelCNN on Asteroid01-03")

    p = sub.add_parser("predict", help="Predict shape from simulation directory")
    p.add_argument("target",
                   help="Simulation directory path, or 'all' to process every folder")
    p.add_argument("--model", default=str(model_path))
    p.add_argument("--out",   default=str(pred_dir))

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
