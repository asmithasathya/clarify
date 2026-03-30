"""Retrieval metrics against HousingQA gold statutes."""

from __future__ import annotations

from typing import Sequence

from src.data.schema import MethodResult


def _match_identifiers(result: MethodResult) -> list[str]:
    identifiers = [passage.doc_id for passage in result.retrieved_passages]
    identifiers.extend(passage.citation for passage in result.retrieved_passages if passage.citation)
    return [identifier.strip().lower() for identifier in identifiers if identifier]


def hit_at_k(result: MethodResult, gold_statutes: Sequence[str], k: int) -> float:
    hits = _match_identifiers(result)[:k]
    gold = {value.strip().lower() for value in gold_statutes}
    return 1.0 if any(hit in gold for hit in hits) else 0.0


def recall_at_k(result: MethodResult, gold_statutes: Sequence[str], k: int) -> float:
    gold = {value.strip().lower() for value in gold_statutes}
    if not gold:
        return 0.0
    hits = {value for value in _match_identifiers(result)[:k] if value in gold}
    return len(hits) / len(gold)


def mrr(result: MethodResult, gold_statutes: Sequence[str]) -> float:
    gold = {value.strip().lower() for value in gold_statutes}
    if not gold:
        return 0.0
    for rank, identifier in enumerate(_match_identifiers(result), start=1):
        if identifier in gold:
            return 1.0 / rank
    return 0.0


def compute_retrieval_metrics(
    results: Sequence[MethodResult],
    gold_lookup: dict[str, Sequence[str]],
    ks: Sequence[int],
) -> dict[str, float]:
    if not results:
        return {}
    metrics: dict[str, float] = {}
    for k in ks:
        metrics[f"retrieval_recall@{k}"] = sum(
            recall_at_k(result, gold_lookup.get(result.example_id, []), k) for result in results
        ) / len(results)
        metrics[f"retrieval_hit@{k}"] = sum(
            hit_at_k(result, gold_lookup.get(result.example_id, []), k) for result in results
        ) / len(results)
    metrics["retrieval_mrr"] = sum(
        mrr(result, gold_lookup.get(result.example_id, [])) for result in results
    ) / len(results)
    return metrics
