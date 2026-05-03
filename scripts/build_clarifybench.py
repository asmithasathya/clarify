"""Build the fixed ClarifyBench v1 benchmark files."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.data.clarifybench_v1 import write_clarifybench_v1


def main() -> None:
    parser = argparse.ArgumentParser(description="Build ClarifyBench v1 benchmark files.")
    parser.add_argument("--output-dir", default="data", help="Directory to write the benchmark files.")
    args = parser.parse_args()

    summary = write_clarifybench_v1(Path(args.output_dir))
    print(
        "Built ClarifyBench v1 with "
        f"{summary['n_examples']} examples "
        f"({summary['dev_examples']} dev / {summary['test_examples']} test)."
    )


if __name__ == "__main__":
    main()
