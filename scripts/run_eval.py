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
        help="Methods to evaluate, e.g. --methods direct_answer generic_hedge generic_clarify targeted_clarify resample_clarify",
    )
    parser.add_argument("--config", default="configs/default.yaml", help="Base config path.")
    parser.add_argument("--dataset", default=None, help="Optional dataset override.")
    parser.add_argument("--split", default=None, help="Optional split override.")
    parser.add_argument("--limit", type=int, default=None, help="Optional example limit.")
    parser.add_argument("--output-dir", default=None, help="Optional output directory.")
    args = parser.parse_args()

    resolved = load_config(args.config)
    if args.dataset is not None:
        resolved["data"]["primary_dataset"] = args.dataset
    if args.split is not None:
        resolved["data"]["split"] = args.split
    payload = run_experiment(
        config=resolved,
        methods=args.methods,
        limit=args.limit,
        output_root=Path(args.output_dir) if args.output_dir else None,
    )
    print(f"Completed evaluation. Outputs saved to {payload['output_dir']}")


if __name__ == "__main__":
    main()
