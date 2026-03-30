"""Selective prediction metrics for abstention-aware evaluation."""

from __future__ import annotations

from typing import Sequence

from src.data.schema import MethodResult


def coverage(results: Sequence[MethodResult]) -> float:
    if not results:
        return 0.0
    answered = sum(1 for result in results if not result.abstained)
    return answered / len(results)


def abstention_rate(results: Sequence[MethodResult]) -> float:
    if not results:
        return 0.0
    return sum(1 for result in results if result.abstained) / len(results)


def selective_accuracy(results: Sequence[MethodResult]) -> float:
    answered = [result for result in results if not result.abstained]
    if not answered:
        return 0.0
    return sum(1 for result in answered if result.correct) / len(answered)


def bad_acceptance_proxy(results: Sequence[MethodResult]) -> float:
    answered = [result for result in results if not result.abstained]
    if not answered:
        return 0.0
    return sum(1 for result in answered if not result.correct) / len(answered)


def useful_answer_rate(results: Sequence[MethodResult]) -> float:
    if not results:
        return 0.0
    return sum(1 for result in results if result.correct and not result.abstained) / len(results)


def risk_coverage_curve(results: Sequence[MethodResult]) -> list[dict[str, float]]:
    answered = [result for result in results if not result.abstained]
    if not results or not answered:
        return []
    ranked = sorted(answered, key=lambda item: item.confidence, reverse=True)
    curve: list[dict[str, float]] = []
    incorrect = 0
    total = len(results)
    for idx, result in enumerate(ranked, start=1):
        if not result.correct:
            incorrect += 1
        curve.append(
            {
                "coverage": idx / total,
                "risk": incorrect / idx,
                "selective_accuracy": 1.0 - (incorrect / idx),
            }
        )
    return curve


def confusion_matrix(results: Sequence[MethodResult]) -> dict[str, dict[str, int]]:
    labels = ["Yes", "No", "Abstain"]
    matrix = {gold: {pred: 0 for pred in labels} for gold in labels}
    for result in results:
        matrix[result.gold_answer][result.predicted_answer] += 1
    return matrix

