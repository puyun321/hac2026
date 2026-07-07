"""
One-shot development pipeline.

Runs only the steps that are missing:
  1. msh_to_npy     -- .msh -> npy_data/*.npy
  2. convexinv      -- physics for Asteroid01-03
  3. CNN train      -- train VoxelCNN
  4. CNN predict    -- predict all asteroids

Usage:
    python run_dev.py              # run all missing steps
    python run_dev.py --retrain    # force re-train even if model exists
    python run_dev.py --predict-only  # skip to predict (model must exist)
"""

import argparse
import subprocess
import sys
from pathlib import Path

CODE   = Path(__file__).resolve().parent
BASE   = CODE.parent
NPY    = BASE / "npy_data"
SIM    = BASE / "simulation"
MODEL  = CODE / "models" / "cnn_shape_model.pth"
PYTHON = sys.executable


def run(args, label):
    print(f"\n{'='*62}")
    print(f"  {label}")
    print(f"{'='*62}")
    result = subprocess.run([PYTHON] + args, cwd=str(CODE))
    if result.returncode != 0:
        print(f"\n[FAILED] {label}")
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser(description="HAC2026 development pipeline")
    parser.add_argument("--retrain",      action="store_true",
                        help="Force re-train even if model already exists")
    parser.add_argument("--predict-only", action="store_true",
                        help="Skip training steps; only run CNN predict")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Step 1: msh -> npy
    # ------------------------------------------------------------------
    npy_count = len(list(NPY.glob("AsteroidModel*.npy"))) if NPY.exists() else 0
    if args.predict_only or npy_count >= 3:
        print(f"Step 1: skip  ({npy_count} .npy files found)")
    else:
        run(["msh_to_npy.py"], "Step 1: Convert .msh → .npy")

    # ------------------------------------------------------------------
    # Step 2: convexinv for Asteroid01-03
    # ------------------------------------------------------------------
    missing_ids = [
        str(int(i)) for i in ("01", "02", "03")
        if not (SIM / f"Asteroid{i}" / "predicted_lightcurve.txt").exists()
    ]
    if args.predict_only or not missing_ids:
        print(f"Step 2: skip  (all simulation outputs present)")
    else:
        run(["convexinv_pipeline.py"] + missing_ids,
            f"Step 2: Physics  (Asteroid {', '.join(missing_ids)})")

    # ------------------------------------------------------------------
    # Step 3: CNN train
    # ------------------------------------------------------------------
    if args.predict_only:
        if not MODEL.exists():
            print("ERROR: --predict-only set but no model found. Run training first.")
            sys.exit(1)
        print(f"Step 3: skip  (--predict-only)")
    elif MODEL.exists() and not args.retrain:
        print(f"Step 3: skip  (model exists: {MODEL.name})")
    else:
        run(["CNN.py", "train"], "Step 3: Train VoxelCNN")

    # ------------------------------------------------------------------
    # Step 4: CNN predict
    # ------------------------------------------------------------------
    run(["CNN.py", "predict", "all"], "Step 4: CNN predict (all asteroids)")

    print(f"\n{'='*62}")
    print(f"  Done.  Results -> {BASE / 'prediction'}")
    print(f"{'='*62}")


if __name__ == "__main__":
    main()
