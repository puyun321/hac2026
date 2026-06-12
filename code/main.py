"""
HAC 2026 submission entry point.

Usage:
    python main.py path/to/input/files path/to/output/files
    python main.py path/to/input/files path/to/output/files --jobs 2
"""

import argparse
import os
from pathlib import Path

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
        help=f"Parallel physics workers (default: {_CPU_COUNT} — all CPU cores)",
    )
    args = parser.parse_args()

    full_pipeline.run(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        jobs=args.jobs,
    )


if __name__ == "__main__":
    main()
