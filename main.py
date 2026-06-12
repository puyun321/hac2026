"""
HAC 2026 entry point — run from the hac2026/ directory.

Usage:
    python main.py data/AsteroidModel04_shape_secret output
    python main.py data output --jobs 4
"""

import sys
import argparse
import os
from pathlib import Path

# Add code/ to path so all modules can be imported
_CODE_DIR = Path(__file__).resolve().parent / "code"
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

import full_pipeline

_CPU_COUNT = os.cpu_count() or 1


def main():
    parser = argparse.ArgumentParser(
        description="HAC 2026: asteroid shape reconstruction",
    )
    parser.add_argument("input_dir",  help="Directory containing lightcurve .txt files")
    parser.add_argument("output_dir", help="Directory for output .stl files")
    parser.add_argument(
        "--jobs", type=int, default=_CPU_COUNT,
        help=f"Parallel physics workers (default: {_CPU_COUNT})",
    )
    args = parser.parse_args()

    _HERE = Path(__file__).resolve().parent   # always hac2026/

    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    if not input_dir.is_absolute():
        input_dir = _HERE / input_dir
    if not output_dir.is_absolute():
        output_dir = _HERE / output_dir

    full_pipeline.run(
        input_dir=input_dir,
        output_dir=output_dir,
        jobs=args.jobs,
    )


if __name__ == "__main__":
    main()
