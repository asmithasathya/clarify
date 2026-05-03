"""Action-level metrics: rates of clarification, answering, and appropriate action."""

from __future__ import annotations

from typing import Sequence

from src.data.schema import MethodResult


def answer_rate(results: Sequence[MethodResult]) -> float:
    """Fraction of examples where the method answered directly."""
    if not results:
        return 0.0
    return sum(1 for r in results if r.answered_directly) / len(results)


def final_answer_rate(results: Sequence[MethodResult]) -> float:
    """Fraction of examples that end with a final answer after any dialogue."""
    if not results:
        return 0.0
    return sum(1 for r in results if r.final_answer) / len(results)


def clarification_rate(results: Sequence[MethodResult]) -> float:
    """Fraction of examples where the method asked a clarification question."""
    if not results:
        return 0.0
    return sum(1 for r in results if r.asked_clarification) / len(results)


def abstention_rate(results: Sequence[MethodResult]) -> float:
    """Fraction of examples where the method abstained."""
    if not results:
        return 0.0
    return sum(1 for r in results if r.response_strategy == "abstain") / len(results)


def appropriate_action_rate(results: Sequence[MethodResult]) -> float:
    """Fraction of examples where the method took the right type of action.

    - If the gold says clarification was needed: asking clarification or
      presenting alternatives counts as appropriate.
    - If the gold says clarification was NOT needed: answering directly counts
      as appropriate.
    """
    if not results:
        return 0.0
    correct = 0
    for r in results:
        if r.gold_clarification_needed and r.response_strategy in {
            "ask_clarification",
            "present_alternatives",
        }:
            correct += 1
        elif not r.gold_clarification_needed and r.answered_directly:
            correct += 1
    return correct / len(results)


def unnecessary_clarification_rate(results: Sequence[MethodResult]) -> float:
    """Fraction of examples where clarification was asked but was not needed."""
    if not results:
        return 0.0
    return sum(
        1 for r in results
        if r.asked_clarification and not r.gold_clarification_needed
    ) / len(results)


def missed_ambiguity_rate(results: Sequence[MethodResult]) -> float:
    """Fraction of examples where the method answered directly despite ambiguity."""
    if not results:
        return 0.0
    return sum(
        1 for r in results
        if r.answered_directly and r.gold_clarification_needed
    ) / len(results)


def wrong_answer_under_ambiguity(results: Sequence[MethodResult]) -> float:
    """Fraction of examples answered without clarifying AND got wrong."""
    if not results:
        return 0.0
    return sum(
        1 for r in results
        if r.answered_directly and r.gold_clarification_needed and not r.correct
    ) / len(results)


def multi_turn_completion_rate(results: Sequence[MethodResult]) -> float:
    """Fraction of examples that reached a final answer after clarification."""
    if not results:
        return 0.0
    return sum(1 for r in results if r.answered_after_clarification) / len(results)
