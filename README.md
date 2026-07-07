# HAC 2026 — Asteroid Shape Reconstruction

> [!TIP] 
> For proper equation rendering, please view this documentation in day mode instead of night mode. 

## Evaluation 
Before describing the asteroid reconstruction process, we first introduce the evaluation method for the resulting STL file. An STL file represents a three-dimensional object as a collection of triangular facets, where each facet is specified by its normal vector and the coordinates of its three vertices. For example, if we open the `asteroid1.stl` file provided by the organizers in a text editor, its contents begin as follows: 

```text
solid Created by Gmsh
facet normal -0.159289 0.167131 0.972982
  outer loop
    vertex -1.888296041844072 0.0002460429996352797 3.161097003616941
    vertex -1.890329489593098 0.01126283538653845 3.158871721174612
    vertex -1.90946363000239 0.0006845222986165005 3.157556283277667
  endloop
endfacet
facet normal 0.0283625 -0.328219 0.944176
  outer loop
    vertex -0.174033817520803 -1.862797688459199 3.159907628518454
    vertex -0.1963030849178117 -1.864990084954105 3.159814451667421
    vertex -0.1809672714359435 -1.879416053890588 3.154338941421392
  endloop
endfacet
facet normal -0.460625 -0.794076 0.39657
  outer loop
    vertex -2.445548420936831 -3.041386196190822 0.7772878517318224
    vertex -2.430985427219417 -3.046417746146632 0.7841281287959293
    vertex -2.445625154814153 -3.036135406585522 0.7877126970651008
  endloop
endfacet
(etc)
```

In addition, the asteroids provided by the organizers were produced using a 3D printer and then coated with a uniform color. Therefore, the albedo ![\overline{\omega}](https://latex.codecogs.com/png.image?\dpi{110}\overline{\omega}) is a constant. 


TBA 

## Method 1: Voxelization and Deep Learning-Based Reconstruction 

### Requirements

| Package | Version used |
|---------|-------------|
| Python | 3.14.3 |
| torch (CUDA 12.8) | 2.11.0+cu128 |
| numpy | 2.4.4 |
| scipy | 1.17.1 |
| matplotlib | 3.10.8 |
| plotly | 6.7.0 |
| trimesh | 4.12.2 |
| scikit-image | 0.26.0 |

**GPU:** NVIDIA GeForce RTX 5090 (CUDA 12.8)

**Executables** (included under `convexinv_suite/`):
- `convexinv/convexinv.exe`
- `lcgenerator/lcgenerator.exe`

---

### Execution of the model 

Before running the code, please update the directory paths in all `.py` files under the `code` folder according to your local environment. The current configuration points to:

```text
C:\Users\TKU\Desktop\lab\research\hac2026\code
```

First, we convert the reference `.msh` shapes into `.npy` voxel grids. This can be performed execute the `msh_to_npy.py` script located in the `code` folder. We assume that all asteroid are bounded by the rectangle ![[-2,2]\times[-2,2]\times[-1,1]](https://latex.codecogs.com/png.image?\dpi{110}[-2,2]\times[-2,2]\times[-1,1]). We approximate the object by cube voxels of side length 0.01, in other words, each asteroid will be approximately represented by a ![400\times400\times200](https://latex.codecogs.com/png.image?\dpi{110}400\times400\times200), with entries 0 or 1. If the voxel grid is stored using the `bool` or `uint8` data type, each `.npy` file occupies 32 MiB of storage (as expected).

Rather than training the model using only the voxelized data and the light curve, we additionally include the convex hull as an input. The convex hull can be approximated using the software provided by the [Database of Asteroid Models from Inversion Techniques (DAMIT)](https://damit.cuni.cz/projects/damit/pages/software_download), see also the following paper: 

- Ďurech et al. (2010), DAMIT: a database of asteroid models, A&A, 513, A46 (ADS: [2010A&A...513A..46D](https://ui.adsabs.harvard.edu/abs/2010A%26A...513A..46D), preprint: [PDF 1.4 MiB](https://damit.cuni.cz/projects/damit/files/durech_et_at_2010_damit_preprint.pdf)) 

Specifically, we use the `convexinv` module, which is based on the following references:

- Kaasalainen, J., Torppa, J. "Optimization Methods for Asteroid Lightcurve Inversion: I. Shape Determination". 2001. Icarus 153, 24-36.
- Kaasalainen, J., Torppa, J., Muinonen, K., "Optimization Methods for Asteroid Lightcurve Inversion: II. The Complete Inverse Problem". 2001. Icarus 153, 37-51.
- Kaasalainen, M., Mottola, S. Fulchignoni, M., "Asteroid Models from Disk-integrated Data" in Asteroids III. 2002, 139-150.
- Kaasalainen, M., Durech, J., "Inverse Problems of NEO Photometry: Imaging the NEO Population", Proceedings IAU Symposium No. 236, in press.

For convenience, we have packaged the C programs into an `.exe` file, which can be found in the `convexinv_suite/convexinv` folder. If you prefer to execute the program through Python, we have also prepared `convexinv_pipeline.py`, which can be run using the following command:

```cmd
REM Run convexinv for Asteroid 01, 02, 03
python convexinv_pipeline.py 1 2 3
```




We are now ready to train the model by running the following program:

```cmd
Train VoxelCNN
python CNN.py train
```

After training the model, we can proceed with the prediction, and the results will appear in the `output` folder: 

```cmd
REM Prediction of a single asteroid
cd ..
python main.py data\AsteroidModel04_shape_secret output
REM Prediction all asteroids in the data directory
cd ..
python main.py data output
```

Use `--jobs N` to control parallel workers (defaults to your CPU count):

```cmd
python main.py data output --jobs 4
```

---

### Model Architecture 

#### Input

The model receives a single tensor built by concatenating two lightcurve sources:

| Component | Shape | Description |
|-----------|-------|-------------|
| `lc_obs` | (360, 28) | Real observed intensity lightcurves (downsampled from 841 steps to 360) |
| `lc_pred` | (360, 28) | Physics-predicted lightcurves from convexinv + lcgenerator |
| **Combined `x_lc`** | **(360, 56)** | Final CNN input: 360 time steps × 56 channels |

The 28 channels per source correspond to **7 measurement angles × 4 camera positions**.

#### Output

| | Value |
|---|---|
| Raw output | Logits of shape **(32, 32, 16)** |
| After sigmoid | Voxel occupancy probability [0, 1] |
| After threshold (> 0.5) | Binary occupancy grid |
| Voxel pitch | 0.125 per cell (4.0 / 32 in x/y, 2.0 / 16 in z) |

#### Training labels

Reference `.msh` shapes are voxelised at pitch 0.01 → 400 × 400 × 200 grid, then
max-pooled down to **32 × 32 × 16** to match the model output.
Loss function: **BCEWithLogitsLoss**.
Augmentation: 72 cyclic phase shifts per asteroid, with matching voxel Z-rotations.

#### Post-processing (prediction)

```
logits (32×32×16)
  → sigmoid → threshold at 0.5 → binary voxel grid
  → keep largest connected component (removes stray voxels)
  → pad with zeros → marching cubes → mesh (verts, faces)
  → normalize_mesh() → scale to competition bounding box
  → STL output
```

---

### Output Files

All outputs are written under `output/Asteroid0X/`.

| File | Description |
|------|-------------|
| `Asteroid0X.stl` | VoxelCNN final submission STL |
| `simulation/asteroid.obj` | Physics mesh from convexinv |
| `simulation/predicted_lightcurve.txt` | Forward-simulated lightcurve |
| `prediction/cnn_asteroid.stl` | VoxelCNN corrected shape |
| `prediction/cnn_shape_comparison.html` | Interactive 3D comparison: Physics vs VoxelCNN |
