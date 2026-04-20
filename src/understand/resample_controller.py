"""Control loop for selective resampling over unstable intent stages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from src.llm.schemas import IntentStabilityReportSchema
from src.understand.intent_stability import select_weakest_stage


@dataclass
class RepairDecision:
    should_continue: bool
    should_clarify: bool
    next_stage: str | None
    reason: str


def decide_repair_action(
    report: IntentStabilityReportSchema,
    *,
    round_index: int,
    max_rounds: int,
    confidence_threshold: float,
    clarification_fallback_threshold: float,
    resample_stages: Sequence[str],
    clarify_if_low_confidence: bool = True,
) -> RepairDecision:
    if report.overall_confidence >= confidence_threshold and not report.weak_points:
        return RepairDecision(
            should_continue=False,
            should_clarify=False,
            next_stage=None,
            reason="Intent confidence is high enough to stop resampling.",
        )

    if round_index >= max_rounds:
        should_clarify = clarify_if_low_confidence and (
            report.overall_confidence < clarification_fallback_threshold or bool(report.weak_points)
        )
        return RepairDecision(
            should_continue=False,
            should_clarify=should_clarify,
            next_stage=None,
            reason="Reached the configured repair-round budget.",
        )

    next_stage = select_weakest_stage(report, allowed_stages=resample_stages)
    if next_stage is None:
        should_clarify = clarify_if_low_confidence and report.overall_confidence < clarification_fallback_threshold
        return RepairDecision(
            should_continue=False,
            should_clarify=should_clarify,
            next_stage=None,
            reason="No allowed weak stage remained for selective resampling.",
        )

    return RepairDecision(
        should_continue=True,
        should_clarify=False,
        next_stage=next_stage,
        reason=f"Resample the weakest stage: {next_stage}.",
    )

