"""Aggregate metric computation for clarification-oriented evaluation."""

from __future__ import annotations

from typing import Sequence

from src.data.schema import MethodResult
from src.eval.selective_metrics import (
    abstention_rate,
    answer_rate,
    appropriate_action_rate,
    clarification_rate,
    final_answer_rate,
    missed_ambiguity_rate,
    multi_turn_completion_rate,
    unnecessary_clarification_rate,
    wrong_answer_under_ambiguity,
)


def task_success_rate(results: Sequence[MethodResult]) -> float:
    """Fraction of examples where the final answer was marked correct."""
    if not results:
        return 0.0
    return sum(1 for r in results if r.correct) / len(results)


def clarification_precision(results: Sequence[MethodResult]) -> float:
    """When clarification was asked, how often was it actually needed?"""
    asked = [r for r in results if r.asked_clarification]
    if not asked:
        return 0.0
    return sum(1 for r in asked if r.gold_clarification_needed) / len(asked)


def clarification_recall(results: Sequence[MethodResult]) -> float:
    """Of cases that needed clarification, how often did the method ask?"""
    needed = [r for r in results if r.gold_clarification_needed]
    if not needed:
        return 0.0
    return sum(1 for r in needed if r.asked_clarification) / len(needed)


def ambiguity_detection_accuracy(results: Sequence[MethodResult]) -> float:
    """How often did the detector agree with the gold label?"""
    if not results:
        return 0.0
    correct = 0
    for r in results:
        if r.is_ambiguous_detected == r.gold_clarification_needed:
            correct += 1
    return correct / len(results)


def strategy_distribution(results: Sequence[MethodResult]) -> dict[str, int]:
    """Count how often each strategy was chosen."""
    dist: dict[str, int] = {}
    for r in results:
        dist[r.response_strategy] = dist.get(r.response_strategy, 0) + 1
    return dist


def average_score(results: Sequence[MethodResult], attr: str) -> float:
    values = [getattr(r, attr) for r in results if getattr(r, attr) is not None]
    if not values:
        return 0.0
    return sum(values) / len(values)


def intent_stability_rate(results: Sequence[MethodResult]) -> float:
    if not results:
        return 0.0
    return sum(1 for r in results if (r.intent_confidence or 0.0) >= 0.8) / len(results)


def resample_rate(results: Sequence[MethodResult]) -> float:
    if not results:
        return 0.0
    return sum(1 for r in results if r.resample_rounds > 0) / len(results)


def clarification_after_resampling_rate(results: Sequence[MethodResult]) -> float:
    if not results:
        return 0.0
    return sum(1 for r in results if r.asked_clarification and r.resample_rounds > 0) / len(results)


def resolved_without_clarification_rate(results: Sequence[MethodResult]) -> float:
    if not results:
        return 0.0
    return sum(1 for r in results if not r.asked_clarification and (r.final_answer or r.response_strategy != "abstain")) / len(results)


def compute_all_metrics(results: Sequence[MethodResult]) -> dict[str, object]:
    """Compute all evaluation metrics for a set of method results."""
    return {
        "task_success_rate": task_success_rate(results),
        "appropriate_action_rate": appropriate_action_rate(results),
        "clarification_rate": clarification_rate(results),
        "answer_rate": answer_rate(results),
        "final_answer_rate": final_answer_rate(results),
        "abstention_rate": abstention_rate(results),
        "clarification_precision": clarification_precision(results),
        "clarification_recall": clarification_recall(results),
        "ambiguity_detection_accuracy": ambiguity_detection_accuracy(results),
        "unnecessary_clarification_rate": unnecessary_clarification_rate(results),
        "missed_ambiguity_rate": missed_ambiguity_rate(results),
        "wrong_answer_under_ambiguity": wrong_answer_under_ambiguity(results),
        "multi_turn_completion_rate": multi_turn_completion_rate(results),
        "average_answer_score": average_score(results, "answer_score"),
        "average_clarification_quality": average_score(results, "clarification_quality_score"),
        "average_alternatives_quality": average_score(results, "alternatives_quality_score"),
        "mean_intent_confidence": average_score(results, "intent_confidence"),
        "intent_stability_rate": intent_stability_rate(results),
        "resample_rate": resample_rate(results),
        "avg_resample_rounds": average_score(results, "resample_rounds"),
        "clarification_after_resampling_rate": clarification_after_resampling_rate(results),
        "resolved_without_clarification_rate": resolved_without_clarification_rate(results),
        "avg_task_model_calls": average_score(results, "task_model_calls"),
        "avg_latency": average_score(results, "latency_seconds"),
        "avg_estimated_cost": average_score(results, "estimated_cost"),
        "strategy_distribution": strategy_distribution(results),
        "n_examples": len(results),
    }
