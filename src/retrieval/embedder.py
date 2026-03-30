"""Embedding model wrappers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Sequence

import numpy as np


class BaseEmbedder(ABC):
    @abstractmethod
    def encode(self, texts: Sequence[str], batch_size: int | None = None) -> np.ndarray:
        raise NotImplementedError


class SentenceTransformerEmbedder(BaseEmbedder):
    def __init__(self, model_name: str, normalize_embeddings: bool = True) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for the embedding backend."
            ) from exc
        self.model = SentenceTransformer(model_name)
        self.normalize_embeddings = normalize_embeddings

    def encode(self, texts: Sequence[str], batch_size: int | None = None) -> np.ndarray:
        embeddings = self.model.encode(
            list(texts),
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=False,
        )
        return np.asarray(embeddings, dtype="float32")


def build_embedder(config: dict[str, Any]) -> BaseEmbedder:
    retrieval_cfg = config.get("retrieval", {})
    return SentenceTransformerEmbedder(
        model_name=retrieval_cfg["embedder_model"],
        normalize_embeddings=retrieval_cfg.get("normalize_embeddings", True),
    )

