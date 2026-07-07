"""
convexinv_pipeline.py
Physics-only pipeline: convex lightcurve inversion (Kaasalainen method) → mesh.

This runs the classical convex inversion approach with no CNN correction.
Use full_pipeline.py (or main.py) for the full physics + CNN submission pipeline.

Steps for each selected dataset:
    1. run_convexinv  – lightcurves → convexinv shape (Gaussian image)
    2. shape_to_mesh  – Gaussian image → convex polyhedron mesh (OBJ/PLY/MSH + plot)
    3. generate_predicted_lc – forward-render predicted lightcurves from the mesh

Usage:
    python convexinv_pipeline.py              # interactive menu
    python convexinv_pipeline.py 1            # run Asteroid01
    python convexinv_pipeline.py 1 3          # run Asteroid01 and Asteroid03
    python convexinv_pipeline.py all          # run every dataset (parallel, all CPU cores)
    python convexinv_pipeline.py all --jobs 2 # limit to 2 parallel workers
"""

import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# =============================================================================
# DATASET SELECTION  (edit this line to choose which datasets to run)
#
#   Examples:
#     RUN_DATASETS = ["Asteroid04"]           # run only Asteroid 04
#     RUN_DATASETS = ["Asteroid01", "Asteroid03"]  # run 01 and 03
#     RUN_DATASETS = "all"                    # run every dataset
#     RUN_DATASETS = []                       # show interactive menu
# =============================================================================
RUN_DATASETS = []   # <-- change this

_CODE_DIR  = Path(__file__).resolve().parent
_BASE      = _CODE_DIR.parent
DATA_DIR   = _BASE / "data"
SIM_DIR    = _BASE / "simulation"
_CPU_COUNT = os.cpu_count() or 1

if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

import run_convexinv as rc_mod
import shape_to_mesh as s2m_mod
import generate_predicted_lc as gplc_mod


def discover_datasets():
    datasets = []
    for model_dir in sorted(DATA_DIR.glob("AsteroidModel*")):
        lc_files = sorted(model_dir.glob("**/Asteroid*_lightcurve_intensity.txt"))
        if not lc_files:
            lc_files = sorted(model_dir.glob("**/Asteroid*_lightcurve_intensity_blender.txt"))
        if not lc_files:
            continue
        raw_id = model_dir.name.replace("AsteroidModel", "")[:2]
        datasets.append({
            "name":    model_dir.name,
            "label":   f"Asteroid {raw_id}  ({model_dir.name})",
            "lc_file": lc_files[0],
            "out_dir": SIM_DIR / f"Asteroid{raw_id}",
        })
    return datasets


def _print_menu(datasets):
    print()
    print("+-------------------------------------------+")
    print("|   Asteroid Shape Inversion Pipeline       |")
    print("+-------------------------------------------+")
    for i, ds in enumerate(datasets, 1):
        print(f"|  [{i}] {ds['label']:<39}|")
    print("|  [0] Run ALL datasets                     |")
    print("+-------------------------------------------+")


def pick_datasets(datasets, cli_args=None):
    if cli_args:
        tokens = [a.strip() for a in cli_args if not a.startswith("--")]
        if any(t.lower() in ("all", "0") for t in tokens):
            return datasets
        selected = []
        for tok in tokens:
            if tok.isdigit():
                idx = int(tok) - 1
                if 0 <= idx < len(datasets):
                    selected.append(datasets[idx])
                else:
                    print(f"[WARN] No dataset #{tok}, skipping.")
            else:
                key = tok.replace(" ", "").lower()
                match = [ds for ds in datasets
                         if ds["label"].replace(" ", "").lower().startswith(key)]
                if match:
                    selected.extend(match)
                else:
                    print(f"[WARN] '{tok}' did not match any dataset, skipping.")
        return selected

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
                print(f"[WARN] No dataset #{tok}, skipping.")
    return selected


def run_dataset(ds):
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  Dataset : {ds['name']}")
    print(f"  Input   : {ds['lc_file']}")
    print(f"  Output  : {ds['out_dir']}")
    print(sep)

    shape_file = rc_mod.run(data_file=ds["lc_file"], output_dir=ds["out_dir"])
    if shape_file is None or not Path(shape_file).exists():
        print("[ERROR] convexinv did not produce a shape file.")
        return False

    s2m_mod.run(shape_file=shape_file, output_dir=ds["out_dir"])
    gplc_mod.run(sim_dir=ds["out_dir"])

    print(f"\n[DONE] {ds['name']}  →  {ds['out_dir']}")
    return True


def run_parallel(selected, jobs=_CPU_COUNT):
    print(f"\n  Running {len(selected)} dataset(s) in parallel  (jobs={jobs})")
    ok = 0
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        future_to_ds = {pool.submit(run_dataset, ds): ds for ds in selected}
        for future in as_completed(future_to_ds):
            ds = future_to_ds[future]
            if future.result():
                ok += 1
            else:
                print(f"[FAIL] {ds['name']}")
    return ok


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Convex inversion physics pipeline (parallel by default)",
    )
    parser.add_argument(
        "targets", nargs="*",
        help="Dataset numbers, names, or 'all'. Omit for interactive menu.",
    )
    parser.add_argument(
        "--jobs", type=int, default=_CPU_COUNT,
        help=f"Parallel workers (default: {_CPU_COUNT} — all CPU cores). Use 1 to disable.",
    )
    args = parser.parse_args()

    datasets = discover_datasets()
    if not datasets:
        print(f"[ERROR] No datasets found under {DATA_DIR}")
        sys.exit(1)

    if args.targets:
        selected = pick_datasets(datasets, args.targets)
    elif RUN_DATASETS == "all":
        selected = datasets
    elif RUN_DATASETS:
        selected = []
        for name in (RUN_DATASETS if isinstance(RUN_DATASETS, list) else [RUN_DATASETS]):
            match = [ds for ds in datasets
                     if ds["label"].replace(" ", "").startswith(name.replace(" ", ""))]
            if match:
                selected.extend(match)
            else:
                print(f"[WARN] '{name}' did not match any dataset, skipping.")
    else:
        selected = pick_datasets(datasets, None)

    if not selected:
        print("No datasets selected. Exiting.")
        sys.exit(0)

    ok = run_parallel(selected, jobs=args.jobs)

    print(f"\n{'='*60}")
    print(f"Finished: {ok}/{len(selected)} dataset(s) completed successfully.")


if __name__ == "__main__":
    main()
