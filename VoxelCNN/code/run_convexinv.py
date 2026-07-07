
import math
import os
import subprocess
from pathlib import Path

# =============================================================================
# PATHS  (edit these if your files are in different locations)
# =============================================================================
DATA_FILE   = Path(r"C:\Users\TKU\Desktop\lab\research\hac2026\data" 
                   r"\AsteroidModel01_shape_public\Asteroid1_lightcurve_data"
                   r"\Asteroid01_lightcurve_intensity.txt")

PARAMS_FILE = Path(r"C:\Users\TKU\Desktop\lab\research\hac2026"
                   r"\convexinv_suite\input_convexinv")

CONVEXINV_EXE = Path(r"C:\Users\TKU\Desktop\lab\research\hac2026"
                     r"\convexinv_suite\convexinv\convexinv.exe")

OUTPUT_DIR  = Path(r"C:\Users\TKU\Desktop\lab\research\hac2026\simulation")

# =============================================================================
# SIMULATION PARAMETERS
# =============================================================================
PERIOD_HOURS = 5.76198   # rotation period (from input_convexinv)
JD_START     = 2451545.0 # Julian Date of first observation (J2000.0)
SUN_DIR      = (1.0, 0.0, 0.0)  # fixed sun direction (unit vector, +X axis)
LC_TYPE      = 1         # 0 = absolute intensity, 1 = relative (normalized)

# =============================================================================
# CAMERA GEOMETRY
# 28 columns = 4 camera tracks × 7 initial asteroid rotation angles
#
# Column order within each angle group (from Readme):
#   col 0: HORIZ_A  – horizontal camera A (beam-splitter / opposition)
#   col 1: HORIZ_B  – horizontal camera B (side view)
#   col 2: TOP      – top camera looking down   (elevation +90°)
#   col 3: BOTTOM   – virtual bottom camera     (elevation −90°)
#
# Horizontal cameras have elevation = 0° by definition.
# The Blender data confirms HORIZ_A and HORIZ_B produce IDENTICAL lightcurves
# for all 7 angle groups (max_diff = 0.0), so HORIZ_B is at azimuth 0° = same
# position as HORIZ_A.  Setting HORIZ_B_AZIMUTH != 0 gives convexinv wrong
# observer directions for half the horizontal lightcurves.
# =============================================================================
HORIZ_B_AZIMUTH = 0.0    # same azimuth as HORIZ_A (confirmed by Blender data: col2==col3)

CAMERAS = [
    {"name": "HORIZ_A", "azimuth_deg":   0.0,            "elevation_deg":   0.0},
    {"name": "HORIZ_B", "azimuth_deg":   HORIZ_B_AZIMUTH,"elevation_deg":   0.0},
    {"name": "TOP",     "azimuth_deg":   0.0,            "elevation_deg":  90.0},
    {"name": "BOTTOM",  "azimuth_deg":   0.0,            "elevation_deg": -90.0},
]

INITIAL_ANGLES_DEG = [0, 45, 90, 135, 225, 270, 315]  # 7 starting phases

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def unit_vector(azimuth_deg, elevation_deg):
    """Convert azimuth + elevation (degrees) to a unit 3D vector."""
    az = math.radians(azimuth_deg)
    el = math.radians(elevation_deg)
    x  = math.cos(el) * math.cos(az)
    y  = math.cos(el) * math.sin(az)
    z  = math.sin(el)
    return (x, y, z)


_MAX_STEPS = 360   # lcgenerator MAX_N_OBS = 20000; with 28 LCs, 360*28=10080 is safe


def load_data(filepath):
    data = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            values = [float(v) for v in line.split(",")]
            data.append(values[1:])  # drop the leading row-index column

    # Real LCs have 841 steps (841*28=23548 > MAX_N_OBS=20000).
    # Uniformly downsample to _MAX_STEPS so lcgenerator doesn't overflow.
    if len(data) > _MAX_STEPS:
        import numpy as np
        arr     = np.array(data)
        indices = np.linspace(0, len(arr) - 1, _MAX_STEPS, dtype=int)
        data    = arr[indices].tolist()
        print(f"  Downsampled LC: {len(indices)} steps (was {len(arr)})")

    return data


def build_geometry():
    """
    Return 28 (observer_unit_vector, jd_phase_offset) pairs (4 cameras × 7 angles).

    Column order is angle-major, camera-minor (from Readme):
      [angle0-cam0, angle0-cam1, ..., angle1-cam0, ...]
    phase_offset = 0 for all LCs: the deg000/deg045/... groups rotate the CAMERA rig
    around Z, not the asteroid, so only the observer azimuth changes.
    """
    geometry = []
    for angle_deg in INITIAL_ANGLES_DEG:
        for cam in CAMERAS:
            # Rotate the entire camera rig by angle_deg around Z.
            # TOP/BOTTOM cameras are on the Z-axis so their azimuth is irrelevant.
            az = (cam["azimuth_deg"] + angle_deg) % 360.0
            obs = unit_vector(az, cam["elevation_deg"])
            geometry.append((obs, 0.0))   # all lightcurves share the same phase zero
    return geometry


def write_convexinv_input(data, output_file):
    """Write the lightcurve data as a convexinv-format input file."""
    n_steps = len(data)
    n_lcs   = len(data[0])

    geometry     = build_geometry()
    period_days  = PERIOD_HOURS / 24.0
    dt           = period_days / n_steps  # time between phase steps

    assert len(geometry) == n_lcs, (
        f"Geometry has {len(geometry)} entries but data has {n_lcs} columns. "
        "Check CAMERAS and INITIAL_ANGLES_DEG."
    )

    with open(output_file, "w") as f:
        f.write(f"{n_lcs}\n")

        for lc_i, (obs, jd_offset) in enumerate(geometry):
            f.write(f"{n_steps}  {LC_TYPE}\n")
            for step in range(n_steps):
                jd         = JD_START + jd_offset + step * dt
                brightness = data[step][lc_i]
                f.write(
                    f"{jd:.6f}  {brightness:.10e}"
                    f"  {SUN_DIR[0]:.6f} {SUN_DIR[1]:.6f} {SUN_DIR[2]:.6f}"
                    f"  {obs[0]:.6f} {obs[1]:.6f} {obs[2]:.6f}\n"
                )


def run_convexinv(lcs_file, params_file, shape_out, params_out, fitted_lcs_out):
    """Run the convexinv executable with the prepared input files."""
    cmd = [
        str(CONVEXINV_EXE), "-v",
        "-o", str(shape_out),
        "-p", str(params_out),
        str(params_file),
        str(fitted_lcs_out),
    ]
    print("Command:", " ".join(cmd))
    env = os.environ.copy()
    msys2_bin = r"C:\msys64\ucrt64\bin"
    if msys2_bin not in env.get("PATH", ""):
        env["PATH"] = msys2_bin + os.pathsep + env.get("PATH", "")
    stdin_data = Path(lcs_file).read_bytes()
    result = subprocess.run(cmd, input=stdin_data, capture_output=True, env=env)

    if result.stdout:
        print(result.stdout.decode("utf-8", errors="replace"))
    if result.returncode != 0:
        print(f"Exit code {result.returncode}")
        if result.stderr:
            print("Stderr:", result.stderr.decode("utf-8", errors="replace"))

    return result.returncode

# =============================================================================
# MAIN (callable as module or standalone script)
# =============================================================================


def run(data_file=None, output_dir=None):
    lc_file  = Path(data_file)  if data_file  else DATA_FILE
    out_dir  = Path(output_dir) if output_dir else OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    lcs_input  = out_dir / "lcs_input.txt"
    shape_out  = out_dir / "shape_result.out"
    params_out = out_dir / "params_result.out"
    fitted_lcs = out_dir / "fitted_lcs.dat"

    # 1. Load brightness data
    data = load_data(lc_file)
    print(f"Loaded: {len(data)} time steps × {len(data[0])} lightcurves")

    # 2. Write convexinv input file
    write_convexinv_input(data, lcs_input)
    print(f"Written convexinv input → {lcs_input}")

    # 3. Run convexinv
    if not CONVEXINV_EXE.exists():
        print(f"\nconvexinv not found at: {CONVEXINV_EXE}")
        print("Compile it first with 'make' inside the convexinv/ directory.")
        print(f"\nInput file is ready at: {lcs_input}")
        print("You can run manually with:")
        print(f"  cat {lcs_input} | convexinv -v -o {shape_out} -p {params_out} {PARAMS_FILE} {fitted_lcs}")
        return None

    rc = run_convexinv(lcs_input, PARAMS_FILE, shape_out, params_out, fitted_lcs)

    if rc == 0:
        print(f"\nShape model → {shape_out}")
        print(f"Parameters  → {params_out}")
        print(f"Fitted LCs  → {fitted_lcs}")
        return shape_out

    return None


def main():
    run()

if __name__ == "__main__":
    main()
