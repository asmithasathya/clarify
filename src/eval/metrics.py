"""Aggregate metric computation."""

from __future__ import annotations

from typing import Sequence

from src.data.schema import MethodResult
from src.eval.attribution_metrics import (
    average_claim_support_score,
    fraction_explanation_sentences_with_supporting_citation,
    fraction_final_answers_with_gold_overlap,
    unsupported_answer_rate,
)
from src.eval.retrieval_metrics import compute_retrieval_metrics
from src.eval.selective_metrics import (
    abstention_rate,
    bad_acceptance_proxy,
    confusion_matrix,
    coverage,
    risk_coverage_curve,
    selective_accuracy,
    useful_answer_rate,
)


def exact_match_accuracy(results: Sequence[MethodResult]) -> float:
    if not results:
        return 0.0
    return sum(1 for result in results if result.correct) / len(results)


def compute_all_metrics(
    results: Sequence[MethodResult],
    *,
    gold_lookup: dict[str, Sequence[str]],
    retrieval_ks: Sequence[int],
    unsupported_threshold: float = 0.55,
) -> dict[str, object]:
    metrics: dict[str, object] = {
        "exact_match_accuracy": exact_match_accuracy(results),
        "abstention_rate": abstention_rate(results),
        "selective_accuracy": selective_accuracy(results),
        "coverage": coverage(results),
        "bad_acceptance_proxy": bad_acceptance_proxy(results),
        "useful_answer_rate": useful_answer_rate(results),
        "confusion_matrix": confusion_matrix(results),
        "risk_coverage_curve": risk_coverage_curve(results),
        "average_claim_support_score": average_claim_support_score(results),
        "fraction_explanation_sentences_with_supporting_citation": (
            fraction_explanation_sentences_with_supporting_citation(results)
        ),
        "fraction_final_answers_with_gold_overlap": (
            fraction_final_answers_with_gold_overlap(results, gold_lookup)
        ),
        "unsupported_answer_rate": unsupported_answer_rate(
            results,
            threshold=unsupported_threshold,
        ),
    }
    metrics.update(compute_retrieval_metrics(results, gold_lookup, retrieval_ks))
    return metrics

