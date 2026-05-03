"""Aggregation and weak-point analysis for sampled intent reasoning stages."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import re
from typing import Iterable, Sequence

from src.llm.schemas import (
    AmbiguityAssessmentSchema,
    ClarificationQuestionSchema,
    IntentModelSchema,
    IntentSampleBundleSchema,
    IntentStabilityReportSchema,
    IntentWeakPointSchema,
    StrategyDecisionSchema,
)
from src.understand.confidence_calibrator import IntentConfidenceCalibrator


_WORD_RE = re.compile(r"[A-Za-z0-9]+")


def _normalize_label(value: str) -> str:
    return " ".join(_WORD_RE.findall(value.lower()))


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _dominant(counter: Counter[str]) -> tuple[str, int]:
    if not counter:
        return "", 0
    return counter.most_common(1)[0]


def _agreement(counter: Counter[str], total: int) -> float:
    if total <= 0 or not counter:
        return 0.0
    return counter.most_common(1)[0][1] / total


@dataclass
class AmbiguityAggregation:
    assessment: AmbiguityAssessmentSchema
    stage_report: IntentSampleBundleSchema


@dataclass
class IntentAggregation:
    intent: IntentModelSchema
    stage_report: IntentSampleBundleSchema


@dataclass
class StrategyAggregation:
    decision: StrategyDecisionSchema
    stage_report: IntentSampleBundleSchema


@dataclass
class ClarificationTargetAggregation:
    clarification: ClarificationQuestionSchema
    stage_report: IntentSampleBundleSchema


def aggregate_ambiguity_samples(samples: Iterable[AmbiguityAssessmentSchema]) -> AmbiguityAggregation:
    rows = list(samples)
    if not rows:
        raise ValueError("aggregate_ambiguity_samples() requires at least one sample.")

    ambiguous_votes = sum(1 for sample in rows if sample.is_ambiguous)
    ambiguity_types = Counter(sample.ambiguity_type for sample in rows if sample.ambiguity_type != "none")
    missing_counter = Counter(
        _normalize_label(mv.variable)
        for sample in rows
        for mv in sample.missing_variables
        if _normalize_label(mv.variable)
    )
    dominant_missing, _ = _dominant(missing_counter)
    avg_conf = _mean([sample.confidence for sample in rows])
    is_ambiguous = ambiguous_votes >= max(1, len(rows) // 2 + len(rows) % 2)
    dominant_type, _ = _dominant(ambiguity_types)
    rationale_parts = []
    if dominant_missing:
        rationale_parts.append(f"Most samples flagged missing information about {dominant_missing}.")
    rationale_parts.append(f"{ambiguous_votes}/{len(rows)} ambiguity votes.")
    assessment = AmbiguityAssessmentSchema(
        is_ambiguous=is_ambiguous,
        ambiguity_type=dominant_type or ("underspecified" if is_ambiguous else "none"),
        missing_variables=[
            {
                "variable": dominant_missing or "user intent / missing details",
                "why_missing": "Most frequent missing variable across sampled ambiguity assessments.",
                "importance": max(0.5, avg_conf),
            }
        ]
        if is_ambiguous
        else [],
        confidence=max(0.0, min(1.0, 0.5 * (ambiguous_votes / len(rows)) + 0.5 * avg_conf)),
        rationale=" ".join(rationale_parts),
    )
    stage_report = IntentSampleBundleSchema(
        stage_name="ambiguity_detection",
        raw_confidence=assessment.confidence,
        sample_size=len(rows),
        agreement=ambiguous_votes / len(rows) if is_ambiguous else (len(rows) - ambiguous_votes) / len(rows),
        dominant_label="ambiguous" if is_ambiguous else "not ambiguous",
        supporting_labels=[label for label, _ in ambiguity_types.most_common(3)],
    )
    return AmbiguityAggregation(assessment=assessment, stage_report=stage_report)


def aggregate_intent_samples(samples: Iterable[IntentModelSchema]) -> IntentAggregation:
    rows = list(samples)
    if not rows:
        raise ValueError("aggregate_intent_samples() requires at least one sample.")

    top_counter = Counter()
    gap_counter = Counter()
    entropy_counter = Counter()
    representative_by_top: dict[str, IntentModelSchema] = {}
    for sample in rows:
        idx = sample.most_likely_index if 0 <= sample.most_likely_index < len(sample.interpretations) else 0
        top = _normalize_label(sample.interpretations[idx].description)
        top_counter[top] += 1
        representative_by_top.setdefault(top, sample)
        gap_counter[_normalize_label(sample.gap_description)] += 1
        entropy_counter[sample.entropy_estimate] += 1

    dominant_top, _ = _dominant(top_counter)
    representative = representative_by_top[dominant_top]
    ordered_interps = sorted(
        representative.interpretations,
        key=lambda interp: (-interp.plausibility, _normalize_label(interp.description)),
    )
    stage_report = IntentSampleBundleSchema(
        stage_name="intent_modeling",
        raw_confidence=max(
            0.0,
            min(
                1.0,
                0.6 * _agreement(top_counter, len(rows))
                + 0.4 * _mean([representative.interpretations[representative.most_likely_index].plausibility]),
            ),
        ),
        sample_size=len(rows),
        agreement=_agreement(top_counter, len(rows)),
        dominant_label=dominant_top or _normalize_label(ordered_interps[0].description),
        supporting_labels=[label for label, _ in top_counter.most_common(4)],
    )
    return IntentAggregation(
        intent=IntentModelSchema(
            interpretations=ordered_interps,
            most_likely_index=0,
            entropy_estimate=_dominant(entropy_counter)[0] or representative.entropy_estimate,
            gap_description=_dominant(gap_counter)[0] or representative.gap_description,
        ),
        stage_report=stage_report,
    )


def aggregate_strategy_samples(samples: Iterable[StrategyDecisionSchema]) -> StrategyAggregation:
    rows = list(samples)
    if not rows:
        raise ValueError("aggregate_strategy_samples() requires at least one sample.")

    strategy_counter = Counter(sample.strategy for sample in rows)
    dominant_strategy, _ = _dominant(strategy_counter)
    representative = next(sample for sample in rows if sample.strategy == dominant_strategy)
    agreement = _agreement(strategy_counter, len(rows))
    confidence = max(0.0, min(1.0, 0.6 * agreement + 0.4 * _mean([sample.confidence for sample in rows])))
    return StrategyAggregation(
        decision=StrategyDecisionSchema(
            strategy=representative.strategy,
            rationale=representative.rationale,
            confidence=confidence,
        ),
        stage_report=IntentSampleBundleSchema(
            stage_name="strategy_selection",
            raw_confidence=confidence,
            sample_size=len(rows),
            agreement=agreement,
            dominant_label=dominant_strategy,
            supporting_labels=[label for label, _ in strategy_counter.most_common(5)],
        ),
    )


def aggregate_clarification_targets(
    samples: Iterable[ClarificationQuestionSchema],
) -> ClarificationTargetAggregation:
    rows = list(samples)
    if not rows:
        raise ValueError("aggregate_clarification_targets() requires at least one sample.")

    target_counter = Counter(_normalize_label(sample.target_variable) for sample in rows)
    question_counter = Counter(_normalize_label(sample.question) for sample in rows)
    dominant_target, _ = _dominant(target_counter)
    dominant_question, _ = _dominant(question_counter)
    representative = next(
        sample for sample in rows
        if _normalize_label(sample.target_variable) == dominant_target
        and _normalize_label(sample.question) == dominant_question
    )
    agreement = min(_agreement(target_counter, len(rows)), _agreement(question_counter, len(rows)))
    return ClarificationTargetAggregation(
        clarification=representative,
        stage_report=IntentSampleBundleSchema(
            stage_name="clarification_target",
            raw_confidence=agreement,
            sample_size=len(rows),
            agreement=agreement,
            dominant_label=dominant_target,
            supporting_labels=[label for label, _ in target_counter.most_common(4)],
        ),
    )


def build_stability_report(
    stage_reports: Sequence[IntentSampleBundleSchema],
    *,
    weak_point_threshold: float,
    calibrator: IntentConfidenceCalibrator | None = None,
) -> IntentStabilityReportSchema:
    if not stage_reports:
        raise ValueError("build_stability_report() requires at least one stage report.")

    raw_scores = [report.raw_confidence for report in stage_reports]
    raw_confidence = max(0.0, min(1.0, 0.5 * min(raw_scores) + 0.5 * _mean(raw_scores)))
    calibrated = calibrator.predict(raw_confidence) if calibrator is not None else raw_confidence

    weak_points = [
        IntentWeakPointSchema(
            stage_name=report.stage_name,
            confidence=report.raw_confidence,
            reason=f"Agreement on {report.stage_name} remained below the weak-point threshold.",
        )
        for report in stage_reports
        if report.raw_confidence < weak_point_threshold
    ]
    if weak_points:
        weak_points = sorted(weak_points, key=lambda point: point.confidence)
        explanation = (
            "Intent understanding is unstable, primarily at "
            + ", ".join(point.stage_name for point in weak_points[:3])
            + "."
        )
    else:
        explanation = "Intent understanding is stable across the sampled reasoning stages."

    if calibrator is not None:
        band = calibrator.band(calibrated)
    elif calibrated >= 0.80:
        band = "high"
    elif calibrated >= 0.55:
        band = "medium"
    else:
        band = "low"

    return IntentStabilityReportSchema(
        overall_confidence=calibrated,
        confidence_band=band,
        uncertainty_explanation=explanation,
        stage_reports=list(stage_reports),
        weak_points=weak_points,
        should_clarify=bool(weak_points),
    )


def select_weakest_stage(
    report: IntentStabilityReportSchema,
    *,
    allowed_stages: Sequence[str],
) -> str | None:
    candidates = [point for point in report.weak_points if point.stage_name in allowed_stages]
    if not candidates:
        return None
    return min(candidates, key=lambda point: point.confidence).stage_name

