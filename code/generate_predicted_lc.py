
import os
import subprocess
import numpy as np
from pathlib import Path

LCGEN_EXE = Path(r"C:\Users\TKU\Desktop\lab\research\hac2026"
                 r"\version_0.2.1\lcgenerator\lcgenerator.exe")


def read_obj(obj_path):
    """Return (vertices, faces) from a Wavefront OBJ file."""
    verts, faces = [], []
    with open(obj_path) as f:
        for line in f:
            tok = line.split()
            if not tok:
                continue
            if tok[0] == 'v':
                verts.append((float(tok[1]), float(tok[2]), float(tok[3])))
            elif tok[0] == 'f':
                # OBJ face indices are 1-based; keep 1-based for lcgenerator
                faces.append((int(tok[1]), int(tok[2]), int(tok[3])))
    return verts, faces


def write_lcgen_shape(path, verts, faces):
    """Write shape in lcgenerator format: n_ver n_fac, vertices, faces."""
    with open(path, 'w') as f:
        f.write(f"{len(verts)} {len(faces)}\n")
        for v in verts:
            f.write(f"{v[0]:.10f} {v[1]:.10f} {v[2]:.10f}\n")
        for face in faces:
            f.write(f"{face[0]} {face[1]} {face[2]}\n")


def read_lcs_input_times(lcs_path):
    timestamps = []
    with open(lcs_path) as f:
        n_lcs = int(f.readline())
        for _ in range(n_lcs):
            n_pts, _lc_type = f.readline().split()
            n_pts = int(n_pts)
            lc_times = []
            for _ in range(n_pts):
                parts = f.readline().split()
                lc_times.append(float(parts[0]))
            timestamps.append(lc_times)
    return n_lcs, timestamps


def run(sim_dir=None):
    out_dir = Path(sim_dir)

    obj_path    = out_dir / "asteroid.obj"
    params_path = out_dir / "params_result.out"
    lcs_path    = out_dir / "lcs_input.txt"
    shape_tmp   = out_dir / "_lcgen_shape.txt"
    lcgen_out   = out_dir / "_lcgen_brightness.txt"
    result_path = out_dir / "predicted_lightcurve.txt"

    for p in (obj_path, params_path, lcs_path):
        if not p.exists():
            print(f"[WARN] Missing file: {p}")
            return None

    if not LCGEN_EXE.exists():
        print(f"[ERROR] lcgenerator not found at {LCGEN_EXE}")
        return None

    verts, faces = read_obj(obj_path)
    write_lcgen_shape(shape_tmp, verts, faces)
    print(f"  Shape: {len(verts)} vertices, {len(faces)} faces")

    cmd = [str(LCGEN_EXE), "-v", str(params_path), str(shape_tmp), str(lcgen_out)]
    env = os.environ.copy()
    msys2_bin = r"C:\msys64\ucrt64\bin"
    if msys2_bin not in env.get("PATH", ""):
        env["PATH"] = msys2_bin + os.pathsep + env["PATH"]

    result = subprocess.run(cmd, input=Path(lcs_path).read_bytes(), capture_output=True, env=env)

    if result.stdout:
        print(result.stdout.decode("utf-8", errors="replace").strip())
    if result.returncode != 0:
        print(f"  [ERROR] lcgenerator exit {result.returncode}")
        if result.stderr:
            print(result.stderr.decode("utf-8", errors="replace"))
        return None

    brightness_flat = np.array([float(l) for l in
                                 lcgen_out.read_text().splitlines() if l.strip()])

    n_lcs, timestamps = read_lcs_input_times(lcs_path)
    n_pts = len(timestamps[0])

    if len(brightness_flat) != n_lcs * n_pts:
        print(f"  [WARN] Expected {n_lcs * n_pts} brightness values, got {len(brightness_flat)}")

    # reshape lcgenerator output: (n_lcs, n_pts) → transpose to (n_pts, n_lcs)
    brightness = brightness_flat.reshape(n_lcs, n_pts).T
    col_means  = brightness.mean(axis=0, keepdims=True)
    col_means[col_means == 0] = 1.0
    brightness = brightness / col_means

    time_col   = np.array(timestamps[0]).reshape(-1, 1)
    out_matrix = np.hstack([time_col, brightness])
    np.savetxt(result_path, out_matrix, delimiter=",", fmt=["%.6f"] + ["%.10e"] * n_lcs)
    print(f"  Predicted LC  -> {result_path}  ({n_pts} steps × {n_lcs} LCs)")

    shape_tmp.unlink(missing_ok=True)
    lcgen_out.unlink(missing_ok=True)

    return result_path


def main():
    import sys
    if len(sys.argv) < 2:
        print("Usage: python generate_predicted_lc.py <sim_dir> [sim_dir2 ...]")
        sys.exit(1)
    for d in sys.argv[1:]:
        print(f"\n--- {d} ---")
        run(d)

if __name__ == "__main__":
    main()
