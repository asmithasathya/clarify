"""Dense retrieval with optional reranking and multi-query fusion."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Sequence

import numpy as np

from src.data.schema import RetrievedPassage
from src.retrieval.embedder import BaseEmbedder, build_embedder
from src.retrieval.faiss_index import load_index_bundle, search
from src.retrieval.rerank import BaseReranker, build_reranker


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


class Retriever:
    def __init__(
        self,
        config: dict[str, Any],
        embedder: BaseEmbedder,
        documents: list[Any],
        index: Any,
        reranker: BaseReranker | None = None,
    ) -> None:
        self.config = config
        self.embedder = embedder
        self.documents = documents
        self.index = index
        self.reranker = reranker

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any],
        *,
        embedder: BaseEmbedder | None = None,
        reranker: BaseReranker | None = None,
    ) -> "Retriever":
        bundle = load_index_bundle(config["paths"]["index_dir"])
        return cls(
            config=config,
            embedder=embedder or build_embedder(config),
            documents=bundle.documents,
            index=bundle.index,
            reranker=reranker if reranker is not None else build_reranker(config),
        )

    def _decorate_query(self, query: str, state: str | None) -> str:
        pieces = [state or "", "housing law", str(self.config.get("project", {}).get("year", 2021)), query]
        return " ".join(piece for piece in pieces if piece).strip()

    def _dense_search(
        self,
        query: str,
        *,
        state: str | None = None,
        top_k: int | None = None,
        fetch_k: int | None = None,
    ) -> list[RetrievedPassage]:
        retrieval_cfg = self.config.get("retrieval", {})
        top_k = top_k or retrieval_cfg.get("top_k", 8)
        fetch_k = fetch_k or retrieval_cfg.get("initial_fetch_k", 64)
        query_vector = self.embedder.encode([self._decorate_query(query, state)])
        scores, indices = search(self.index, query_vector, fetch_k)

        same_state: list[RetrievedPassage] = []
        fallback: list[RetrievedPassage] = []
        for rank, (doc_idx, score) in enumerate(zip(indices[0], scores[0]), start=1):
            if doc_idx < 0:
                continue
            doc = self.documents[int(doc_idx)]
            passage = RetrievedPassage(
                doc_id=doc.doc_id,
                text=doc.text,
                state=doc.state,
                citation=doc.citation,
                dense_score=float(score),
                fused_score=float(score),
                rank=rank,
            )
            if state and doc.state and doc.state.strip().lower() == state.strip().lower():
                same_state.append(passage)
            else:
                fallback.append(passage)

        candidates = same_state if same_state else []
        if len(candidates) < top_k:
            candidates.extend(fallback[: max(0, top_k - len(candidates))])
        return candidates[:top_k]

    def _apply_reranker(
        self,
        query: str,
        passages: Sequence[RetrievedPassage],
    ) -> list[RetrievedPassage]:
        ranked = [RetrievedPassage.model_validate(p.model_dump()) for p in passages]
        if not ranked:
            return ranked
        if self.reranker is None:
            for idx, passage in enumerate(ranked, start=1):
                passage.rank = idx
            return ranked

        raw_scores = self.reranker.score(query, ranked)
        for passage, rerank_score in zip(ranked, raw_scores):
            passage.rerank_score = rerank_score
            dense_component = passage.dense_score or 0.0
            passage.fused_score = 0.5 * dense_component + 0.5 * _sigmoid(rerank_score)
        ranked.sort(key=lambda item: item.fused_score or float("-inf"), reverse=True)
        for idx, passage in enumerate(ranked, start=1):
            passage.rank = idx
        return ranked

    def retrieve(
        self,
        query: str,
        *,
        state: str | None = None,
        top_k: int | None = None,
    ) -> list[RetrievedPassage]:
        dense_hits = self._dense_search(query, state=state, top_k=top_k)
        return self._apply_reranker(query, dense_hits)

    def retrieve_multi(
        self,
        queries: Sequence[str],
        *,
        state: str | None = None,
        top_k: int | None = None,
        rerank_query: str | None = None,
    ) -> list[RetrievedPassage]:
        top_k = top_k or self.config.get("retrieval", {}).get("top_k", 8)
        buckets: dict[str, dict[str, Any]] = defaultdict(dict)

        for query in queries:
            hits = self._dense_search(query, state=state, top_k=top_k, fetch_k=top_k * 8)
            for rank, hit in enumerate(hits, start=1):
                bucket = buckets[hit.doc_id]
                bucket["passage"] = hit
                bucket["rrf"] = bucket.get("rrf", 0.0) + 1.0 / (60.0 + rank)
                bucket["dense"] = max(bucket.get("dense", float("-inf")), hit.dense_score or float("-inf"))

        fused: list[RetrievedPassage] = []
        for bucket in buckets.values():
            passage = bucket["passage"]
            passage.fused_score = 0.5 * bucket["rrf"] + 0.5 * bucket["dense"]
            fused.append(passage)

        fused.sort(key=lambda item: item.fused_score or float("-inf"), reverse=True)
        rerank_seed = rerank_query or queries[0]
        reranked = self._apply_reranker(rerank_seed, fused[: max(top_k * 2, top_k)])
        return reranked[:top_k]


def build_retriever(config: dict[str, Any]) -> Retriever:
    return Retriever.from_config(config)

