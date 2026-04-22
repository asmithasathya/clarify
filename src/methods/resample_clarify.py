"""Intent stabilization via selective resampling with clarification fallback."""

from __future__ import annotations

from copy import deepcopy
import time
from typing import Any, Iterable

from src.data.schema import DialogueExample, MethodResult
from src.llm.prompts import (
    ambiguity_detection_prompt,
    generic_clarification_prompt,
    intent_modeling_prompt,
    schema_instruction,
    strategy_selection_prompt,
)
from src.llm.schemas import (
    AmbiguityAssessmentSchema,
    ClarificationQuestionSchema,
    IntentModelSchema,
    StrategyDecisionSchema,
)
from src.understand.ambiguity_detector import AmbiguityDetector
from src.understand.clarification_generator import ClarificationGenerator
from src.understand.confidence_calibrator import IntentConfidenceCalibrator
from src.understand.intent_model import IntentModeler
from src.understand.intent_stability import (
    aggregate_ambiguity_samples,
    aggregate_clarification_targets,
    aggregate_intent_samples,
    aggregate_strategy_samples,
    build_stability_report,
)
from src.understand.resample_controller import decide_repair_action
from src.understand.strategy_selector import StrategySelector


def _load_calibrator(config: dict[str, Any]) -> IntentConfidenceCalibrator | None:
    calibration_cfg = config.get("intent_calibration", {})
    if not calibration_cfg.get("enabled", True):
        return None
    path = calibration_cfg.get("calibrator_path")
    if not path:
        return IntentConfidenceCalibrator(
            medium_threshold=float(calibration_cfg.get("medium_threshold", 0.55)),
            high_threshold=float(calibration_cfg.get("high_threshold", 0.80)),
            n_buckets=int(calibration_cfg.get("n_buckets", 10)),
        )
    try:
        return IntentConfidenceCalibrator.from_path(path)
    except FileNotFoundError:
        return IntentConfidenceCalibrator(
            medium_threshold=float(calibration_cfg.get("medium_threshold", 0.55)),
            high_threshold=float(calibration_cfg.get("high_threshold", 0.80)),
            n_buckets=int(calibration_cfg.get("n_buckets", 10)),
        )


def _call_units(config: dict[str, Any], task_model_calls: int) -> float:
    unit = float(config.get("report", {}).get("cost_per_call_unit", 1.0))
    return task_model_calls * unit


def _sample_ambiguity(
    request: str,
    *,
    count: int,
    temperature: float,
    detector: AmbiguityDetector,
) -> list[AmbiguityAssessmentSchema]:
    if count <= 0:
        return []
    if detector.generator is None or detector.config.get("ablations", {}).get("heuristic_ambiguity_detection", False):
        return [detector.detect(request, temperature=temperature) for _ in range(count)]
    return detector.generator.generate_structured_n(
        ambiguity_detection_prompt(
            request=request,
            schema_text=schema_instruction(AmbiguityAssessmentSchema),
        ),
        AmbiguityAssessmentSchema,
        n=count,
        temperature=temperature,
    )


def _sample_intent(
    request: str,
    assessment: AmbiguityAssessmentSchema,
    *,
    count: int,
    temperature: float,
    intent_modeler: IntentModeler,
) -> list[IntentModelSchema]:
    if count <= 0:
        return []
    if intent_modeler.generator is None or intent_modeler.config.get("ablations", {}).get("heuristic_intent_modeling", False):
        return [intent_modeler.model(request, assessment, temperature=temperature) for _ in range(count)]

    missing_vars = [mv.variable for mv in assessment.missing_variables]
    return intent_modeler.generator.generate_structured_n(
        intent_modeling_prompt(
            request=request,
            ambiguity_rationale=assessment.rationale,
            missing_variables=missing_vars,
            schema_text=schema_instruction(IntentModelSchema),
        ),
        IntentModelSchema,
        n=count,
        temperature=temperature,
    )


def _sample_strategy(
    request: str,
    assessment: AmbiguityAssessmentSchema,
    intent: IntentModelSchema,
    *,
    count: int,
    temperature: float,
    strategy_selector: StrategySelector,
) -> list[StrategyDecisionSchema]:
    if count <= 0:
        return []
    if strategy_selector.generator is None or strategy_selector.config.get("ablations", {}).get("heuristic_strategy_selection", False):
        return [strategy_selector.select(request, assessment, intent, temperature=temperature) for _ in range(count)]

    return strategy_selector.generator.generate_structured_n(
        strategy_selection_prompt(
            request=request,
            is_ambiguous=assessment.is_ambiguous,
            num_missing_variables=len(assessment.missing_variables),
            entropy=intent.entropy_estimate,
            gap_description=intent.gap_description,
            schema_text=schema_instruction(StrategyDecisionSchema),
        ),
        StrategyDecisionSchema,
        n=count,
        temperature=temperature,
    )


def _sample_generic_clarifications(
    request: str,
    *,
    count: int,
    temperature: float,
    clarification_generator: ClarificationGenerator,
) -> list[ClarificationQuestionSchema]:
    if count <= 0:
        return []
    generator = clarification_generator.generator
    if generator is None:
        return [
            ClarificationQuestionSchema(
                question="Could you share a bit more detail about what you need?",
                target_variable="general context",
                why_this_helps="More context would make the request easier to answer well.",
            )
            for _ in range(count)
        ]
    return generator.generate_structured_n(
        generic_clarification_prompt(
            request=request,
            schema_text=schema_instruction(ClarificationQuestionSchema),
        ),
        ClarificationQuestionSchema,
        n=count,
        temperature=temperature,
    )


def _sample_targeted_clarifications(
    example: DialogueExample,
    target_variable: str,
    intent: IntentModelSchema,
    *,
    count: int,
    temperature: float,
    clarification_generator: ClarificationGenerator,
) -> list[ClarificationQuestionSchema]:
    if count <= 0:
        return []
    return [
        clarification_generator.generate_clarification(
            example.user_request,
            target_variable,
            intent,
            temperature=temperature,
        )
        for _ in range(count)
    ]


def _build_trace_samples(samples: Iterable[Any]) -> list[dict[str, Any]]:
    return [sample.model_dump() for sample in samples]


def _ablation_enabled(config: dict[str, Any], name: str) -> bool:
    return bool(config.get("ablations", {}).get(name, False))


def _apply_resample_ablation_defaults(config: dict[str, Any]) -> dict[str, Any]:
    updated = deepcopy(config)
    rs_cfg = updated.setdefault("intent_resampling", {})
    if _ablation_enabled(updated, "single_sample_only"):
        rs_cfg["initial_samples"] = 1
        rs_cfg["repair_samples"] = 1
        rs_cfg["max_rounds"] = 0
    if _ablation_enabled(updated, "one_round_only"):
        rs_cfg["max_rounds"] = 1
    if _ablation_enabled(updated, "no_selective_resample") or _ablation_enabled(updated, "resample_all_stages"):
        rs_cfg["resample_stages"] = [
            "ambiguity_detection",
            "intent_modeling",
            "strategy_selection",
            "clarification_target",
        ]
    return updated


def run_resample_clarify(
    example: DialogueExample,
    *,
    config: dict[str, Any],
    ambiguity_detector: AmbiguityDetector,
    intent_modeler: IntentModeler,
    strategy_selector: StrategySelector,
    clarification_generator: ClarificationGenerator,
) -> MethodResult:
    started = time.perf_counter()
    config = _apply_resample_ablation_defaults(config)
    trace: dict[str, Any] = {"method": "resample_clarify"}

    rs_cfg = config.get("intent_resampling", {})
    temperature = float(rs_cfg.get("stage_temperature", 0.7))
    initial_samples = int(rs_cfg.get("initial_samples", 5))
    repair_samples = int(rs_cfg.get("repair_samples", 3))
    max_rounds = int(rs_cfg.get("max_rounds", 2))
    weak_point_threshold = float(rs_cfg.get("weak_point_threshold", 0.75))
    confidence_threshold = float(rs_cfg.get("confidence_threshold", 0.80))
    clarification_fallback_threshold = float(rs_cfg.get("clarification_fallback_threshold", 0.80))
    clarify_if_low_confidence = bool(rs_cfg.get("clarify_if_low_confidence", True))
    resample_stages = list(rs_cfg.get("resample_stages", []))
    selective_repair_enabled = bool(rs_cfg.get("selective_repair_enabled", False))
    fallback_question_mode = str(rs_cfg.get("fallback_question_mode", "generic")).lower()
    use_calibration_for_decisions = bool(rs_cfg.get("use_calibration_for_decisions", False))

    calibrator = None if _ablation_enabled(config, "no_calibration") else _load_calibrator(config)
    task_model_calls = 0
    sample_count = 0

    ambiguity_samples = _sample_ambiguity(
        example.user_request,
        count=initial_samples,
        temperature=temperature,
        detector=ambiguity_detector,
    )
    task_model_calls += len(ambiguity_samples)
    sample_count += len(ambiguity_samples)
    ambiguity_agg = aggregate_ambiguity_samples(ambiguity_samples)

    intent_samples: list[IntentModelSchema] = []
    strategy_samples: list[StrategyDecisionSchema] = []
    clarification_samples: list[ClarificationQuestionSchema] = []
    clarification_agg = None

    if ambiguity_agg.assessment.is_ambiguous:
        intent_samples = _sample_intent(
            example.user_request,
            ambiguity_agg.assessment,
            count=initial_samples,
            temperature=temperature,
            intent_modeler=intent_modeler,
        )
        task_model_calls += len(intent_samples)
        sample_count += len(intent_samples)
        intent_agg = aggregate_intent_samples(intent_samples)
        strategy_samples = _sample_strategy(
            example.user_request,
            ambiguity_agg.assessment,
            intent_agg.intent,
            count=initial_samples,
            temperature=temperature,
            strategy_selector=strategy_selector,
        )
        task_model_calls += len(strategy_samples)
        sample_count += len(strategy_samples)
        strategy_agg = aggregate_strategy_samples(strategy_samples)
    else:
        intent_agg = None
        strategy_agg = None

    current_round = 0
    while True:
        stage_reports = [ambiguity_agg.stage_report]
        if intent_agg is not None:
            stage_reports.append(intent_agg.stage_report)
        if strategy_agg is not None:
            stage_reports.append(strategy_agg.stage_report)
        if clarification_agg is not None:
            stage_reports.append(clarification_agg.stage_report)

        decision_stability_report = build_stability_report(
            stage_reports,
            weak_point_threshold=weak_point_threshold,
            calibrator=calibrator if use_calibration_for_decisions else None,
        )
        stability_report = build_stability_report(
            stage_reports,
            weak_point_threshold=weak_point_threshold,
            calibrator=calibrator,
        )
        trace["latest_stability_report"] = stability_report.model_dump()
        trace["latest_decision_stability_report"] = decision_stability_report.model_dump()

        if _ablation_enabled(config, "clarify_immediately") and ambiguity_agg.assessment.is_ambiguous:
            repair_decision = decide_repair_action(
                decision_stability_report,
                round_index=max_rounds,
                max_rounds=max_rounds,
                confidence_threshold=confidence_threshold,
                clarification_fallback_threshold=clarification_fallback_threshold,
                resample_stages=resample_stages,
                clarify_if_low_confidence=True,
            )
        else:
            repair_decision = decide_repair_action(
                decision_stability_report,
                round_index=current_round,
                max_rounds=max_rounds,
                confidence_threshold=confidence_threshold,
                clarification_fallback_threshold=clarification_fallback_threshold,
                resample_stages=resample_stages,
                clarify_if_low_confidence=clarify_if_low_confidence and not _ablation_enabled(config, "no_clarification_fallback"),
            )

        trace.setdefault("repair_rounds", []).append(
            {
                "round_index": current_round,
                "decision": repair_decision.__dict__,
                "stage_reports": [report.model_dump() for report in stage_reports],
            }
        )

        if not repair_decision.should_continue:
            break

        current_round += 1
        resample_all = (
            not selective_repair_enabled
            or _ablation_enabled(config, "no_selective_resample")
            or _ablation_enabled(config, "resample_all_stages")
        )
        stages_to_refresh: set[str]
        if resample_all:
            stages_to_refresh = {"ambiguity_detection", "intent_modeling", "strategy_selection"}
        elif repair_decision.next_stage == "ambiguity_detection":
            stages_to_refresh = {"ambiguity_detection", "intent_modeling", "strategy_selection"}
        elif repair_decision.next_stage == "intent_modeling":
            stages_to_refresh = {"intent_modeling", "strategy_selection"}
        elif repair_decision.next_stage == "strategy_selection":
            stages_to_refresh = {"strategy_selection"}
        else:
            stages_to_refresh = {"clarification_target"}

        if "ambiguity_detection" in stages_to_refresh:
            extra_samples = _sample_ambiguity(
                example.user_request,
                count=repair_samples,
                temperature=temperature,
                detector=ambiguity_detector,
            )
            ambiguity_samples.extend(extra_samples)
            task_model_calls += len(extra_samples)
            sample_count += len(extra_samples)
            ambiguity_agg = aggregate_ambiguity_samples(ambiguity_samples)

        if ambiguity_agg.assessment.is_ambiguous and "intent_modeling" in stages_to_refresh:
            extra_samples = _sample_intent(
                example.user_request,
                ambiguity_agg.assessment,
                count=repair_samples,
                temperature=temperature,
                intent_modeler=intent_modeler,
            )
            intent_samples.extend(extra_samples)
            task_model_calls += len(extra_samples)
            sample_count += len(extra_samples)
            intent_agg = aggregate_intent_samples(intent_samples)

        if ambiguity_agg.assessment.is_ambiguous and intent_agg is not None and "strategy_selection" in stages_to_refresh:
            extra_samples = _sample_strategy(
                example.user_request,
                ambiguity_agg.assessment,
                intent_agg.intent,
                count=repair_samples,
                temperature=temperature,
                strategy_selector=strategy_selector,
            )
            strategy_samples.extend(extra_samples)
            task_model_calls += len(extra_samples)
            sample_count += len(extra_samples)
            strategy_agg = aggregate_strategy_samples(strategy_samples)

    stage_reports = [ambiguity_agg.stage_report]
    if intent_agg is not None:
        stage_reports.append(intent_agg.stage_report)
    if strategy_agg is not None:
        stage_reports.append(strategy_agg.stage_report)
    final_stability_report = build_stability_report(
        stage_reports,
        weak_point_threshold=weak_point_threshold,
        calibrator=calibrator,
    )
    weak_points = [point.stage_name for point in final_stability_report.weak_points]

    trace["ambiguity_samples"] = _build_trace_samples(ambiguity_samples)
    trace["stabilized_ambiguity"] = ambiguity_agg.assessment.model_dump()
    if intent_agg is not None:
        trace["intent_samples"] = _build_trace_samples(intent_samples)
        trace["stabilized_intent"] = intent_agg.intent.model_dump()
    if strategy_agg is not None:
        trace["strategy_samples"] = _build_trace_samples(strategy_samples)
        trace["stabilized_strategy"] = strategy_agg.decision.model_dump()
    trace["final_stability_report"] = final_stability_report.model_dump()

    if not ambiguity_agg.assessment.is_ambiguous:
        answer = clarification_generator.generate_direct_answer(example.user_request)
        task_model_calls += 1
        result = MethodResult(
            example_id=example.example_id,
            method="resample_clarify",
            user_request=example.user_request,
            hidden_context=example.hidden_context,
            gold_clarification_needed=example.gold_clarification_needed,
            gold_answer=example.gold_answer,
            response_strategy="answer_directly",
            response_text=answer.answer,
            final_answer=answer.answer,
            assumed_interpretation=answer.assumed_interpretation,
            is_ambiguous_detected=False,
            confidence=answer.confidence,
            answered_directly=True,
            trace=trace,
        )
    else:
        clarify_by_fallback = (
            decision_stability_report.overall_confidence < clarification_fallback_threshold
            or bool(decision_stability_report.weak_points)
        ) and not _ablation_enabled(config, "no_clarification_fallback")

        strategy = strategy_agg.decision.strategy if strategy_agg is not None else "ask_clarification"
        if clarify_by_fallback or strategy == "ask_clarification":
            use_generic = _ablation_enabled(config, "generic_question_fallback") or fallback_question_mode == "generic"
            target_variable = (
                ambiguity_agg.assessment.missing_variables[0].variable
                if ambiguity_agg.assessment.missing_variables
                else intent_agg.intent.gap_description if intent_agg is not None
                else "general context"
            )
            if use_generic or intent_agg is None:
                clarification_samples = _sample_generic_clarifications(
                    example.user_request,
                    count=max(1, repair_samples),
                    temperature=temperature,
                    clarification_generator=clarification_generator,
                )
            else:
                clarification_samples = _sample_targeted_clarifications(
                    example,
                    target_variable,
                    intent_agg.intent,
                    count=max(1, repair_samples),
                    temperature=temperature,
                    clarification_generator=clarification_generator,
                )
            task_model_calls += len(clarification_samples)
            sample_count += len(clarification_samples)
            clarification_agg = aggregate_clarification_targets(clarification_samples)
            trace["clarification_samples"] = _build_trace_samples(clarification_samples)
            trace["stabilized_clarification_target"] = clarification_agg.clarification.model_dump()
            result = MethodResult(
                example_id=example.example_id,
                method="resample_clarify",
                user_request=example.user_request,
                hidden_context=example.hidden_context,
                gold_clarification_needed=example.gold_clarification_needed,
                gold_answer=example.gold_answer,
                response_strategy="ask_clarification",
                response_text=clarification_agg.clarification.question,
                clarification_question=clarification_agg.clarification.question,
                is_ambiguous_detected=True,
                confidence=final_stability_report.overall_confidence,
                asked_clarification=True,
                trace=trace,
            )
        elif strategy == "narrow_and_answer" and intent_agg is not None:
            most_likely = intent_agg.intent.interpretations[0]
            narrowed = clarification_generator.generate_narrowed_answer(
                example.user_request,
                most_likely.description,
            )
            task_model_calls += 1
            result = MethodResult(
                example_id=example.example_id,
                method="resample_clarify",
                user_request=example.user_request,
                hidden_context=example.hidden_context,
                gold_clarification_needed=example.gold_clarification_needed,
                gold_answer=example.gold_answer,
                response_strategy="narrow_and_answer",
                response_text=f"Assuming: {narrowed.stated_assumption}\n\n{narrowed.answer}",
                final_answer=narrowed.answer,
                assumed_interpretation=narrowed.stated_assumption,
                is_ambiguous_detected=True,
                confidence=narrowed.confidence,
                trace=trace,
            )
        elif strategy == "present_alternatives" and intent_agg is not None:
            alternatives = clarification_generator.generate_alternatives(example.user_request, intent_agg.intent)
            task_model_calls += 1
            alt_text = alternatives.preamble + "\n"
            for index, alt in enumerate(alternatives.alternatives, start=1):
                alt_text += f"\n{index}. If you mean '{alt.interpretation}': {alt.answer}"
            result = MethodResult(
                example_id=example.example_id,
                method="resample_clarify",
                user_request=example.user_request,
                hidden_context=example.hidden_context,
                gold_clarification_needed=example.gold_clarification_needed,
                gold_answer=example.gold_answer,
                response_strategy="present_alternatives",
                response_text=alt_text,
                is_ambiguous_detected=True,
                confidence=final_stability_report.overall_confidence,
                trace=trace,
            )
        else:
            answer = clarification_generator.generate_direct_answer(example.user_request)
            task_model_calls += 1
            result = MethodResult(
                example_id=example.example_id,
                method="resample_clarify",
                user_request=example.user_request,
                hidden_context=example.hidden_context,
                gold_clarification_needed=example.gold_clarification_needed,
                gold_answer=example.gold_answer,
                response_strategy="answer_directly",
                response_text=answer.answer,
                final_answer=answer.answer,
                assumed_interpretation=answer.assumed_interpretation,
                is_ambiguous_detected=True,
                confidence=answer.confidence,
                answered_directly=True,
                trace=trace,
            )

    result.intent_confidence = final_stability_report.overall_confidence
    result.confidence_band = final_stability_report.confidence_band
    result.uncertainty_explanation = final_stability_report.uncertainty_explanation
    result.weak_points = weak_points
    result.resample_rounds = current_round
    result.sample_count = sample_count
    result.task_model_calls = task_model_calls
    result.estimated_cost = _call_units(config, task_model_calls)
    result.latency_seconds = time.perf_counter() - started
    result.num_missing_variables = len(ambiguity_agg.assessment.missing_variables)
    trace["runtime"] = {
        "task_model_calls": task_model_calls,
        "sample_count": sample_count,
        "resample_rounds": current_round,
        "estimated_cost": result.estimated_cost,
        "latency_seconds": result.latency_seconds,
    }
    return result
