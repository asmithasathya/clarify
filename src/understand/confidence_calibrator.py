"""Lightweight confidence calibration for intent-stabilization signals."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Iterable

from src.utils.io import read_json, write_json


@dataclass
class ConfidenceExample:
    raw_confidence: float
    success: bool


class IntentConfidenceCalibrator:
    """Bucketed calibrator over raw disagreement-based confidence signals."""

    def __init__(
        self,
        *,
        bins: list[dict[str, float]] | None = None,
        medium_threshold: float = 0.55,
        high_threshold: float = 0.80,
        n_buckets: int = 10,
    ) -> None:
        self.bins = bins or []
        self.medium_threshold = medium_threshold
        self.high_threshold = high_threshold
        self.n_buckets = n_buckets

    @staticmethod
    def _clamp(value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    def fit(self, examples: Iterable[ConfidenceExample]) -> "IntentConfidenceCalibrator":
        rows = list(examples)
        if not rows:
            self.bins = []
            return self

        buckets: list[list[ConfidenceExample]] = [[] for _ in range(self.n_buckets)]
        for row in rows:
            index = min(self.n_buckets - 1, int(math.floor(self._clamp(row.raw_confidence) * self.n_buckets)))
            buckets[index].append(row)

        fitted_bins: list[dict[str, float]] = []
        running = 0.0
        for index, bucket in enumerate(buckets):
            if bucket:
                empirical = sum(1.0 for row in bucket if row.success) / len(bucket)
                running = max(running, empirical)
            upper = (index + 1) / self.n_buckets
            fitted_bins.append({"upper": upper, "value": running})
        self.bins = fitted_bins
        return self

    def predict(self, raw_confidence: float) -> float:
        score = self._clamp(raw_confidence)
        if self.bins:
            index = min(len(self.bins) - 1, int(math.floor(score * len(self.bins))))
            return self._clamp(self.bins[index]["value"])
        return score

    def band(self, confidence: float) -> str:
        score = self._clamp(confidence)
        if score >= self.high_threshold:
            return "high"
        if score >= self.medium_threshold:
            return "medium"
        return "low"

    def to_dict(self) -> dict[str, Any]:
        return {
            "bins": self.bins,
            "medium_threshold": self.medium_threshold,
            "high_threshold": self.high_threshold,
            "n_buckets": self.n_buckets,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "IntentConfidenceCalibrator":
        return cls(
            bins=list(payload.get("bins", [])),
            medium_threshold=float(payload.get("medium_threshold", 0.55)),
            high_threshold=float(payload.get("high_threshold", 0.80)),
            n_buckets=int(payload.get("n_buckets", 10)),
        )

    @classmethod
    def from_path(cls, path: str | Path) -> "IntentConfidenceCalibrator":
        return cls.from_dict(read_json(path))

    def save(self, path: str | Path) -> None:
        write_json(self.to_dict(), path)
