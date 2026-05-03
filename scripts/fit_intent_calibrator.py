"""Fit a lightweight intent-confidence calibrator from dev predictions."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.data.schema import MethodResult
from src.understand.confidence_calibrator import ConfidenceExample, IntentConfidenceCalibrator


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit a bucketed intent-confidence calibrator.")
    parser.add_argument("--predictions", required=True, help="Path to predictions.jsonl from a dev-set run.")
    parser.add_argument("--output", required=True, help="Path to write the fitted calibrator JSON.")
    parser.add_argument("--medium-threshold", type=float, default=0.55)
    parser.add_argument("--high-threshold", type=float, default=0.80)
    parser.add_argument("--n-buckets", type=int, default=10)
    args = parser.parse_args()

    rows: list[ConfidenceExample] = []
    with Path(args.predictions).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            result = MethodResult.model_validate_json(line)
            rows.append(
                ConfidenceExample(
                    raw_confidence=float(result.intent_confidence or result.confidence),
                    success=bool(result.correct),
                )
            )

    calibrator = IntentConfidenceCalibrator(
        medium_threshold=args.medium_threshold,
        high_threshold=args.high_threshold,
        n_buckets=args.n_buckets,
    ).fit(rows)
    calibrator.save(args.output)
    print(f"Saved calibrator to {args.output}")


if __name__ == "__main__":
    main()
