"""
Usage:
  python full_pipeline.py              # interactive menu
  python full_pipeline.py 1            # Asteroid01
  python full_pipeline.py 1 3          # Asteroid01 and Asteroid03
  python full_pipeline.py all
  python full_pipeline.py --cnn-only 4
  python full_pipeline.py --physics-only 1
"""

import re
import sys
import importlib.util
import subprocess
import argparse
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Auto-install missing dependencies
# ---------------------------------------------------------------------------
_REQUIRED = [
    ("numpy",      "numpy"),
    ("scipy",      "scipy"),
    ("matplotlib", "matplotlib"),
    ("plotly",     "plotly"),
    ("torch",      "torch"),
]

def _ensure_packages():
    missing = [pip for imp, pip in _REQUIRED if importlib.util.find_spec(imp) is None]
    if not missing:
        return
    print(f"installing missing packages: {', '.join(missing)}")
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install"] + missing,
        stdout=subprocess.DEVNULL,
    )
    print("done.\n")

_ensure_packages()
# ---------------------------------------------------------------------------

CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import run_convexinv         as rc_mod
import shape_to_mesh         as s2m_mod
import generate_predicted_lc as gplc_mod
import CNN                   as cnn_mod
import CNN_delta              as cnn_delta_mod

from shape_to_mesh import normalize_mesh

# Dev-mode defaults (not used by run())
_BASE     = CODE_DIR.parent
_DATA_DIR = _BASE / "data"
_SIM_DIR  = _BASE / "simulation"
_PRED_DIR = _BASE / "prediction"


# ---------------------------------------------------------------------------
# Mesh utilities
# ---------------------------------------------------------------------------


def _read_obj_mesh(path):
    verts, faces = [], []
    with open(path) as f:
        for line in f:
            tok = line.split()
            if not tok:
                continue
            if tok[0] == "v":
                verts.append([float(tok[1]), float(tok[2]), float(tok[3])])
            elif tok[0] == "f":
                # handle v, v/vt, v/vt/vn — OBJ is 1-indexed
                faces.append([int(t.split("/")[0]) - 1 for t in tok[1:]])
    return np.array(verts), faces


def save_normalized_stl(obj_path, stl_path):
    Path(stl_path).parent.mkdir(parents=True, exist_ok=True)
    verts, faces = _read_obj_mesh(obj_path)
    s2m_mod.write_stl(stl_path, verts, faces)
    print(f"  STL  -> {stl_path}")


# ---------------------------------------------------------------------------
# Input discovery
# ---------------------------------------------------------------------------

def discover_lightcurves(input_dir):
    d = Path(input_dir)
    int_files = sorted(d.rglob("*_lightcurve_intensity.txt"))
    if not int_files:
        int_files = sorted(d.rglob("*_lightcurve_intensity_blender.txt"))
    result = []
    for lc in int_files:
        sibling = lc.parent / lc.name.replace("intensity", "binary")
        result.append({
            "lc_intensity": lc,
            "lc_binary":    sibling if sibling.exists() else None,
        })
    return result


def _ast_id_from_lc(lc_path):
    m = re.search(r'Asteroid(\d+)', Path(lc_path).stem, re.IGNORECASE)
    return m.group(1).zfill(2) if m else "00"


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def run_physics(lc_file, sim_dir):
    sim_dir = Path(sim_dir)
    sep = "=" * 62
    print(f"\n{sep}")
    print(f"  physics  input : {lc_file}")
    print(f"           output: {sim_dir}")
    print(sep)

    shape_file = rc_mod.run(data_file=lc_file, output_dir=sim_dir)
    if shape_file is None or not Path(shape_file).exists():
        print("  convexinv failed — no shape file produced")
        return False

    s2m_mod.run(shape_file=shape_file, output_dir=sim_dir)
    gplc_mod.run(sim_dir=sim_dir)
    print(f"\n  physics done  →  {sim_dir}")
    return True


def run_cnn(sim_dir, lc_obs_path, pred_dir, stl_path=None):
    sim_dir  = Path(sim_dir)
    pred_dir = Path(pred_dir)

    sep = "=" * 62
    print(f"\n{sep}")
    print(f"  CNN  {sim_dir.parent.name}")
    print(f"       sim  : {sim_dir}")
    print(f"       pred : {pred_dir}")
    print(sep)

    if not Path(cnn_mod.model_path).exists():
        print(f"  model not found: {cnn_mod.model_path}")
        print("  run  python CNN.py train  first")
        return False

    obj_path  = sim_dir / "asteroid.obj"
    pred_path = sim_dir / "predicted_lightcurve.txt"
    if not obj_path.exists() or not pred_path.exists():
        print(f"  physics outputs missing in {sim_dir}")
        return False

    msh_out = cnn_mod.predict(
        sim_dir=sim_dir,
        model_path=cnn_mod.model_path,
        out_dir=pred_dir,
        lc_obs_path=lc_obs_path,
    )
    if msh_out is None:
        return False

    if stl_path is not None:
        if msh_out and Path(msh_out).exists():
            import shutil
            Path(stl_path).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(msh_out, stl_path)
            print(f"  STL  -> {stl_path}")
        else:
            print(f"  STL not found at {msh_out}")
            return False

    print(f"\n  CNN done  →  {pred_dir}")
    return True


def run_cnn_delta(sim_dir, lc_obs_path, pred_dir, stl_path=None):
    sim_dir  = Path(sim_dir)
    pred_dir = Path(pred_dir)

    sep = "=" * 62
    print(f"\n{sep}")
    print(f"  CNN-delta  {sim_dir.parent.name}")
    print(f"             sim  : {sim_dir}")
    print(f"             pred : {pred_dir}")
    print(sep)

    if not Path(cnn_delta_mod.model_path).exists():
        print(f"  delta model not found: {cnn_delta_mod.model_path}")
        print("  run  python CNN_delta.py train  first")
        return False

    obj_path  = sim_dir / "asteroid.obj"
    pred_path = sim_dir / "predicted_lightcurve.txt"
    if not obj_path.exists() or not pred_path.exists():
        print(f"  physics outputs missing in {sim_dir}")
        return False

    stl_out = cnn_delta_mod.predict(
        sim_dir=sim_dir,
        model_path=cnn_delta_mod.model_path,
        out_dir=pred_dir,
        lc_obs_path=lc_obs_path,
    )
    if stl_out is None:
        return False

    if stl_path is not None:
        if Path(stl_out).exists():
            Path(stl_path).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(stl_out, stl_path)
            print(f"  STL  -> {stl_path}")
        else:
            print(f"  delta STL not found at {stl_out}")
            return False

    print(f"\n  CNN-delta done  →  {pred_dir}")
    return True


# ---------------------------------------------------------------------------
# Evaluation entry point
# ---------------------------------------------------------------------------

def run(input_dir, output_dir, jobs=None):
    input_dir  = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    lc_entries = discover_lightcurves(input_dir)
    if not lc_entries:
        print(f"no lightcurve files found in {input_dir}")
        sys.exit(1)

    tasks = []
    for entry in lc_entries:
        lc_file = entry["lc_intensity"]
        ast_id  = _ast_id_from_lc(lc_file)
        ast_dir = output_dir / f"Asteroid{ast_id}"
        tasks.append({
            "id":        ast_id,
            "lc_file":   lc_file,
            "lc_bin":    entry["lc_binary"],
            "sim_dir":   ast_dir / "simulation",
            "pred_dir":  ast_dir / "prediction",
            "stl":       ast_dir / f"Asteroid{ast_id}.stl",
            "delta_stl": ast_dir / f"Asteroid{ast_id}_delta.stl",
        })

    # --- Phase 1: physics — run all asteroids in parallel ---
    print(f"\n{'#'*62}")
    print(f"  Phase 1: physics  ({len(tasks)} asteroids, jobs={jobs or 'auto'})")
    print(f"{'#'*62}")

    physics_ok = {}
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        future_to_task = {
            pool.submit(run_physics, t["lc_file"], t["sim_dir"]): t
            for t in tasks
        }
        for future in as_completed(future_to_task):
            t = future_to_task[future]
            ok = future.result()
            physics_ok[t["id"]] = ok
            print(f"  [{'ok' if ok else 'FAIL'}] Asteroid{t['id']} physics done")

    # --- Phase 2: CNN — sequential (GPU memory) ---
    print(f"\n{'#'*62}")
    print(f"  Phase 2: CNN correction")
    print(f"{'#'*62}")

    results = []
    for t in tasks:
        cnn_ok = False
        if physics_ok.get(t["id"]):
            cnn_ok = run_cnn(t["sim_dir"], t["lc_file"], t["pred_dir"],
                             stl_path=t["stl"])
        results.append({
            "id":        t["id"],
            "stl":       t["stl"],
            "delta_stl": t["delta_stl"],
            "success":   physics_ok.get(t["id"], False) and cnn_ok,
            "delta_ok":  False,
        })

    # --- Phase 3: CNN-delta — sequential (GPU memory) ---
    print(f"\n{'#'*62}")
    print(f"  Phase 3: CNN-delta correction")
    print(f"{'#'*62}")

    for t, r in zip(tasks, results):
        if r["success"]:
            delta_ok = run_cnn_delta(t["sim_dir"], t["lc_file"], t["pred_dir"],
                                     stl_path=t["delta_stl"])
            r["delta_ok"] = delta_ok

    # Move root-level simulation/ and prediction/ (left by the dev CLI) into training_result/
    _BASE = Path(__file__).resolve().parent.parent
    training_result = _BASE / "training_result"
    for d_name in ("simulation", "prediction"):
        src = _BASE / d_name
        if src.exists():
            dst = training_result / d_name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))
            print(f"  moved {src} → {dst}")

    print(f"\n{'='*62}")
    print("  evaluation summary")
    print(f"{'='*62}")
    for r in results:
        tag = "ok" if r["success"] else "FAILED"
        stl_tag       = "  " if r["stl"].exists()       else "  [missing] "
        delta_stl_tag = "  " if r["delta_stl"].exists() else "  [missing] "
        print(f"  {tag}  Asteroid{r['id']}")
        print(f"         CNN   {stl_tag}{r['stl']}")
        print(f"         delta {delta_stl_tag}{r['delta_stl']}")
    n_ok       = sum(1 for r in results if r["success"])
    n_delta_ok = sum(1 for r in results if r["delta_ok"])
    print(f"\n  CNN: {n_ok}/{len(results)}  |  CNN-delta: {n_delta_ok}/{len(results)}")
    print(f"{'='*62}")


# ---------------------------------------------------------------------------
# Development CLI support (uses local data/ simulation/ prediction/ dirs)
# ---------------------------------------------------------------------------

def discover_datasets(data_dir=None):
    d = Path(data_dir) if data_dir else _DATA_DIR
    datasets = []
    for model_dir in sorted(d.glob("AsteroidModel*")):
        lc_files = sorted(model_dir.glob("**/Asteroid*_lightcurve_intensity.txt"))
        if not lc_files:
            lc_files = sorted(model_dir.glob("**/Asteroid*_lightcurve_intensity_blender.txt"))
        if not lc_files:
            continue
        raw_id  = model_dir.name.replace("AsteroidModel", "")[:2]
        lc_int  = lc_files[0]
        lc_bin  = lc_int.parent / lc_int.name.replace("intensity", "binary")
        datasets.append({
            "name":     model_dir.name,
            "label":    f"Asteroid {raw_id}  ({model_dir.name})",
            "id":       raw_id,
            "lc_file":  lc_int,
            "lc_bin":   lc_bin if lc_bin.exists() else None,
            "sim_dir":  _SIM_DIR  / f"Asteroid{raw_id}",
            "pred_dir": _PRED_DIR / f"Asteroid{raw_id}",
        })
    return datasets


def _print_menu(datasets):
    print("|  Full Pipeline  (Physics + CNN correction) |")
    for i, ds in enumerate(datasets, 1):
        print(f"|  [{i}] {ds['label']:<42}|")
    print("|  [0] Run ALL datasets                      |")


def pick_datasets(datasets, tokens):
    if any(t.lower() in ("all", "0") for t in tokens):
        return datasets
    selected = []
    for tok in tokens:
        if tok.isdigit():
            idx = int(tok) - 1
            if 0 <= idx < len(datasets):
                selected.append(datasets[idx])
            else:
                print(f"#{tok} out of range, skipping")
        else:
            key = tok.replace(" ", "").lower()
            match = [ds for ds in datasets
                     if ds["label"].replace(" ", "").lower().startswith(key)
                     or ds["name"].lower().startswith(key)]
            if match:
                selected.extend(match)
            else:
                print(f"'{tok}' didn't match anything, skipping")
    return selected


def pick_datasets_interactive(datasets):
    _print_menu(datasets)
    raw = input("\nSelect dataset(s)  [number, comma-separated, or 0 for all]: ").strip()
    if not raw:
        return []
    if raw == "0":
        return datasets
    selected = []
    for tok in raw.split(","):
        tok = tok.strip()
        if tok.isdigit():
            idx = int(tok) - 1
            if 0 <= idx < len(datasets):
                selected.append(datasets[idx])
            else:
                print(f"#{tok} out of range, skipping")
    return selected


def print_summary(results):
    sep = "=" * 62
    print(f"\n{sep}")
    print("  results")
    print(sep)
    for r in results:
        status = "ok" if r["success"] else "FAILED"
        print(f"\n  {status}  {r['name']}")
        for fname in ("asteroid.obj", "asteroid.msh",
                      "predicted_lightcurve.txt", "asteroid_shape.png"):
            p = Path(r["sim_dir"]) / fname
            tag = "  " if p.exists() else "  [missing]  "
            print(f"       {tag}{p}")
        for fname in ("cnn_asteroid.obj", "cnn_asteroid.msh",
                      "cnn_shape_comparison.png"):
            p = Path(r["pred_dir"]) / fname
            tag = "  " if p.exists() else "  [missing]  "
            print(f"       {tag}{p}")
    n_ok = sum(1 for r in results if r["success"])
    print(f"\n  {n_ok}/{len(results)} finished")
    print(sep)


def main():
    parser = argparse.ArgumentParser(
        description="Full pipeline: physics inversion + CNN shape correction (dev mode)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "targets", nargs="*",
        help="Dataset numbers or names (e.g. 1 3, Asteroid04, all). "
             "Omit for interactive menu.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--cnn-only",     action="store_true",
                      help="Skip physics; run CNN on existing simulation/ outputs only.")
    mode.add_argument("--physics-only", action="store_true",
                      help="Run physics only; skip CNN correction.")
    args = parser.parse_args()

    all_ds = discover_datasets()
    if not all_ds:
        print(f"no datasets found under {_DATA_DIR}")
        sys.exit(1)

    selected = pick_datasets(all_ds, args.targets) if args.targets \
               else pick_datasets_interactive(all_ds)

    if not selected:
        print("nothing selected, exiting")
        sys.exit(0)

    results = []
    for ds in selected:
        r = {"name": ds["name"], "sim_dir": ds["sim_dir"],
             "pred_dir": ds["pred_dir"], "success": False}

        physics_ok = True
        if not args.cnn_only:
            physics_ok = run_physics(ds["lc_file"], ds["sim_dir"])

        cnn_ok = True
        if physics_ok and not args.physics_only:
            cnn_ok = run_cnn(ds["sim_dir"], ds["lc_file"], ds["pred_dir"])

        r["success"] = physics_ok and cnn_ok
        results.append(r)

    print_summary(results)


if __name__ == "__main__":
    main()
