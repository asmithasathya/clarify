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
        if shutil.which("tectonic") is None:
            raise SystemExit("Neither pdflatex nor tectonic is installed or on PATH.")
        subprocess.run(
            ["tectonic", "--synctex", "--keep-logs", "--keep-intermediates", "main.tex"],
            cwd=paper_dir,
            check=True,
        )
        print(f"Built {paper_dir / 'main.pdf'}")
        return

    subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "main.tex"],
        cwd=paper_dir,
        check=True,
    )
    if (paper_dir / "references.bib").exists() and shutil.which("bibtex") is None:
        raise SystemExit("bibtex is not installed or not on PATH.")
    if (paper_dir / "references.bib").exists():
        subprocess.run(["bibtex", "main"], cwd=paper_dir, check=True)
    for _ in range(2):
        subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "main.tex"],
            cwd=paper_dir,
            check=True,
        )
    print(f"Built {paper_dir / 'main.pdf'}")


if __name__ == "__main__":
    main()
