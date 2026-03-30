"""Run multi-method evaluation and save comparison artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.eval.runner import run_experiment
from src.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run evaluation across multiple methods.")
    parser.add_argument(
        "--methods",
        nargs="+",
        required=True,
        help="Methods to evaluate, e.g. --methods closed_book rag_direct hedge revise_verify",
    )
    parser.add_argument("--config", default="configs/default.yaml", help="Base config path.")
    parser.add_argument("--limit", type=int, default=None, help="Optional example limit.")
    parser.add_argument("--output-dir", default=None, help="Optional output directory.")
    parser.add_argument("--backend", default=None, help="Optional generator backend override.")
    args = parser.parse_args()

    resolved = load_config(args.config)
    if args.backend is not None:
        resolved["model"]["generator"]["backend"] = args.backend
    payload = run_experiment(
        config=resolved,
        methods=args.methods,
        limit=args.limit,
        output_root=Path(args.output_dir) if args.output_dir else None,
    )
    print(f"Completed evaluation. Outputs saved to {payload['output_dir']}")


if __name__ == "__main__":
    main()
