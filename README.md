# Helsinki Asteroid Challenge (HAC 2026) — Asteroid Shape Reconstruction

> [!TIP] 
> For proper equation rendering, please view this documentation in day mode instead of night mode. 

For the organizers' convenience, we have placed our final reconstruction results (`Asteroid04.stl`, `Asteroid05.stl`, ..., `Asteroid10.stl`) directly in the main directory for evaluation.


## Rendering Light Curves from 3D Objects 
Before describing the asteroid reconstruction process, we first explain the mechanism for generating light curves from 3D objects represented by STL files. An STL file represents a three-dimensional object as a collection of triangular facets, where each facet is specified by its normal vector and the coordinates of its three vertices. In addition, the asteroids provided by the organizers were produced using a 3D printer and then coated with a uniform color. Therefore, the albedo ![\overline{\omega}](https://latex.codecogs.com/png.image?\dpi{110}\overline{\omega}) is a constant. Let ![\mathbf{n}_{\Delta}](https://latex.codecogs.com/png.image?\dpi{110}\mathbf{n}_{\Delta}) denote the unit normal of the facet ![\triangle](https://latex.codecogs.com/png.image?\dpi{110}\triangle). Let ![\mathbf{E}_{0}](https://latex.codecogs.com/png.image?\dpi{110}\mathbf{E}_{0})  and ![\mathbf{E}](https://latex.codecogs.com/png.image?\dpi{110}\mathbf{E}) represent the directions of the light source and the observer (camera), respectively. The lightcurve can be approximated by 

![L=\overline{\omega}\sum_{\triangle\in\{{\rm%20facet}\}}S(\mu,\mu_{0}){\rm%20area}(\triangle)+O(\overline{\omega}^{2})](https://latex.codecogs.com/png.image?\dpi{110}L=\overline{\omega}\sum_{\triangle\in\{{\rm%20facet}\}}S(\mu,\mu_{0}){\rm%20area}(\triangle)+O(\overline{\omega}&Hat;{2}))

where ![\mu=(\mathbf{E}\cdot\mathbf{n}_{\triangle})_{+}](https://latex.codecogs.com/png.image?\dpi{110}\mu=(\mathbf{E}\cdot\mathbf{n}_{\triangle})_{+}) and ![\mu_{0}=(\mathbf{E}_{0}\cdot\mathbf{n}_{\triangle})_{+}](https://latex.codecogs.com/png.image?\dpi{110}\mu_{0}=(\mathbf{E}_{0}\cdot\mathbf{n}_{\triangle})_{+}). We assume the Lambert law ![S(\mu,\mu_{0})=\mu\mu_{0}](https://latex.codecogs.com/png.image?\dpi{110}S(\mu,\mu_{0})=\mu\mu_{0}). 

The organizers fix the illumination direction to ![\mathbf{E}_{0}=(-1,0,0)](https://latex.codecogs.com/png.image?\dpi{110}\mathbf{E}_{0}=(-1,0,0)). The positions of the two cameras can then be computed using matrix multiplication. This evaluation can be performed in MATLAB by running the script `light_curve_evaluation.m`, which output the average light curve ![L/{\rm%20average}(L)](https://latex.codecogs.com/png.image?\dpi{110}L/{\rm%20average}(L)) of the object. 

> [!NOTE] 
> The input light curve must be provided in `.csv` format. The `.txt` files supplied by the organizers can be easily converted to this format.

> [!NOTE] 
> In principle, this can also be accomplished using the `lcgenerator` module, which is available from the [Database of Asteroid Models from Inversion Techniques (DAMIT)](https://damit.cuni.cz/projects/damit/pages/software_download). We will provide the appropriate references in a later section. We developed our own implementation to ensure that it is fully compatible with the organizers' experimental setup.

---

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
> [!WARNING] 
> To avoid non-uniqueness arising from arbitrary scaling, we rescaled the reconstructed object to fit the bounding cylinder provided by the organizers. 


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

#### Evaluation training

<img width="2527" height="1314" alt="evaluation_Asteroid01" src="https://github.com/user-attachments/assets/787b2a9f-2d2c-4da2-94e8-c97fc7b98537" />
<img width="2527" height="1314" alt="evaluation_Asteroid02" src="https://github.com/user-attachments/assets/56493b5b-941f-499a-a131-36494a567341" />
<img width="2527" height="1314" alt="evaluation_Asteroid03" src="https://github.com/user-attachments/assets/90cdc6d9-6645-4528-8ec4-2ff02a9635b6" />

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
| `simulation/asteroid.obj` | object produced by convexinv |
| `simulation/predicted_lightcurve.txt` | Forward-simulated lightcurve |
| `prediction/cnn_asteroid.stl` | VoxelCNN corrected shape |
| `prediction/cnn_shape_comparison.html` | Interactive 3D comparison: Physics vs VoxelCNN |

#### Evaluation output 

<img width="2527" height="1314" alt="evaluation_Asteroid04" src="https://github.com/user-attachments/assets/800d46ea-d564-4b9e-ae2c-3f2037851e6a" />
<img width="2527" height="1314" alt="evaluation_Asteroid05" src="https://github.com/user-attachments/assets/2ef9ff57-cc46-4f3d-93d2-9baf293fe4d4" />
<img width="2527" height="1314" alt="evaluation_Asteroid06" src="https://github.com/user-attachments/assets/6cca53f5-6835-4849-a499-564101c72ff0" />
<img width="2527" height="1314" alt="evaluation_Asteroid07" src="https://github.com/user-attachments/assets/8bb4891c-361e-4edb-bb86-5a00685a92f5" />
<img width="2527" height="1314" alt="evaluation_Asteroid08" src="https://github.com/user-attachments/assets/6478358a-5abf-4161-b96b-de0b782eb519" />
<img width="2527" height="1314" alt="evaluation_Asteroid09" src="https://github.com/user-attachments/assets/fd906973-2276-4cae-9b05-d1cdc4bd0ac7" />
<img width="2527" height="1314" alt="evaluation_Asteroid010" src="https://github.com/user-attachments/assets/4d4fc157-4538-4b18-83f8-62f5207c8890" />

--- 

## Method 2: ConvexInv with Manual Scaling 

It is difficult to determine whether the voxelized objects have been reconstructed correctly. At least, Asteroids 5, 7, 8, and 9 still appear to have a chance of being reconstructed successfully. We also wonder whether Asteroids 4-10 are convex. If so, it would likely be preferable to reconstruct them using an existing convex inversion method, such as ConvexInv, as mentioned above. However, even if the asteroid is convex, the reconstruction is unique only up to translation, rotation, and scaling, since we consider only the average light curve. Therefore, these transformations must be determined and applied manually. Fortunately, inspection of the ConvexInv output in [Blender](https://www.blender.org/) suggests that the reconstructed object is already properly centered. Therefore, only the rotation and scaling need to be determined manually.

> [!NOTE] 
> The output of ConvexInv is provided in the `.obj` format, which can be readily converted to the `.stl` format using [Blender](https://www.blender.org/). 

### Asteroid04 

The MATLAB code is available in `scaling_manually/asteroid04_convex`. The resulting reconstruction appears to be in good agreement with the target object (compare to Method 1):

<img width="2527" height="1314" alt="Asteroid04_evaluation" src="https://github.com/user-attachments/assets/44df390f-6530-4ba0-96d5-a1c04c2dfcb7" />




