"""FAISS index helpers for statute retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.data.schema import StatutePassage
from src.utils.io import read_json, read_jsonl, write_json


INDEX_FILENAME = "index.faiss"
DOCS_FILENAME = "documents.jsonl"
META_FILENAME = "index_meta.json"


@dataclass
class FaissIndexBundle:
    index: Any
    documents: list[StatutePassage]
    metadata: dict[str, Any]


def _faiss_import() -> Any:
    try:
        import faiss
    except ImportError as exc:
        raise ImportError("faiss-cpu is required for FAISS retrieval.") from exc
    return faiss


def create_index(
    dim: int,
    *,
    index_type: str = "ivfflat",
    metric: str = "ip",
    nlist: int = 2048,
) -> Any:
    faiss = _faiss_import()
    metric_type = faiss.METRIC_INNER_PRODUCT if metric == "ip" else faiss.METRIC_L2
    index_type = index_type.lower()
    if index_type == "flatip":
        if metric != "ip":
            raise ValueError("flatip index_type only supports inner-product metric.")
        return faiss.IndexFlatIP(dim)
    if index_type == "ivfflat":
        quantizer = faiss.IndexFlatIP(dim) if metric == "ip" else faiss.IndexFlatL2(dim)
        return faiss.IndexIVFFlat(quantizer, dim, nlist, metric_type)
    raise ValueError(f"Unsupported FAISS index type: {index_type}")


def maybe_normalize(vectors: np.ndarray) -> np.ndarray:
    faiss = _faiss_import()
    normalized = np.asarray(vectors, dtype="float32").copy()
    faiss.normalize_L2(normalized)
    return normalized


def save_index(index: Any, index_dir: str | Path, metadata: dict[str, Any]) -> None:
    faiss = _faiss_import()
    directory = Path(index_dir)
    directory.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(directory / INDEX_FILENAME))
    write_json(metadata, directory / META_FILENAME)


def load_index_bundle(index_dir: str | Path) -> FaissIndexBundle:
    faiss = _faiss_import()
    directory = Path(index_dir)
    index = faiss.read_index(str(directory / INDEX_FILENAME))
    documents = [
        StatutePassage.model_validate(record)
        for record in read_jsonl(directory / DOCS_FILENAME)
    ]
    metadata = read_json(directory / META_FILENAME)
    return FaissIndexBundle(index=index, documents=documents, metadata=metadata)


def search(index: Any, query_vectors: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
    query_vectors = np.asarray(query_vectors, dtype="float32")
    return index.search(query_vectors, top_k)

