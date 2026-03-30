"""Cross-encoder reranking wrappers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Sequence

import numpy as np

from src.data.schema import RetrievedPassage


class BaseReranker(ABC):
    @abstractmethod
    def score(self, query: str, passages: Sequence[RetrievedPassage]) -> list[float]:
        raise NotImplementedError


class CrossEncoderReranker(BaseReranker):
    def __init__(self, model_name: str) -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for the reranker backend."
            ) from exc
        self.model = CrossEncoder(model_name)

    def score(self, query: str, passages: Sequence[RetrievedPassage]) -> list[float]:
        if not passages:
            return []
        pairs = [(query, passage.text) for passage in passages]
        scores = self.model.predict(pairs)
        return [float(score) for score in np.asarray(scores).tolist()]


class NoOpReranker(BaseReranker):
    def score(self, query: str, passages: Sequence[RetrievedPassage]) -> list[float]:
        return [0.0 for _ in passages]


def build_reranker(config: dict[str, Any]) -> BaseReranker | None:
    if not config.get("ablations", {}).get("reranker", True):
        return None
    return CrossEncoderReranker(config.get("retrieval", {})["reranker_model"])

