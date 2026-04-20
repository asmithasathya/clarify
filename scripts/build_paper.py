"""Build the manuscript if pdflatex is available."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the LaTeX manuscript.")
    parser.add_argument("--paper-dir", default="paper", help="Paper directory containing main.tex.")
    args = parser.parse_args()

    paper_dir = Path(args.paper_dir)
    if shutil.which("pdflatex") is None:
        raise SystemExit("pdflatex is not installed or not on PATH.")

    for _ in range(2):
        subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "main.tex"],
            cwd=paper_dir,
            check=True,
        )
    print(f"Built {paper_dir / 'main.pdf'}")


if __name__ == "__main__":
    main()
