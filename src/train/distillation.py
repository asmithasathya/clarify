"""Teacher rollout export and distillation-corpus helpers."""

from __future__ import annotations

import json
from typing import Any, Iterable

from src.data.schema import MethodResult
from src.llm.prompts import (
    alternatives_prompt,
    clarification_question_prompt,
    direct_answer_prompt,
    intent_modeling_prompt,
    narrowed_answer_prompt,
    schema_instruction,
    strategy_selection_prompt,
    ambiguity_detection_prompt,
)
from src.llm.schemas import (
    AlternativesResponseSchema,
    AmbiguityAssessmentSchema,
    AnswerSchema,
    ClarificationQuestionSchema,
    IntentModelSchema,
    NarrowedAnswerSchema,
    StrategyDecisionSchema,
)


def _record_id(result: MethodResult, rollout_id: str, stage_name: str) -> str:
    return f"{rollout_id}:{result.example_id}:{stage_name}"


def _assistant_payload(value: Any) -> str:
    if hasattr(value, "model_dump_json"):
        return value.model_dump_json()
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=True)


def build_sft_records(
    result: MethodResult,
    *,
    rollout_id: str,
) -> list[dict[str, Any]]:
    """Convert one teacher rollout into modular SFT stage records."""
    records: list[dict[str, Any]] = []
    trace = result.trace or {}

    ambiguity_payload = trace.get("stabilized_ambiguity")
    if ambiguity_payload:
        records.append(
            {
                "record_id": _record_id(result, rollout_id, "ambiguity_detection"),
                "rollout_id": rollout_id,
                "example_id": result.example_id,
                "dataset_name": result.dataset_name,
                "split_name": result.split_name,
                "teacher_model_id": result.task_model_id,
                "stage_name": "ambiguity_detection",
                "messages": ambiguity_detection_prompt(
                    request=result.user_request,
                    schema_text=schema_instruction(AmbiguityAssessmentSchema),
                ),
                "assistant_target": _assistant_payload(ambiguity_payload),
                "response_strategy": result.response_strategy,
                "intent_confidence": result.intent_confidence,
                "confidence_band": result.confidence_band,
            }
        )

    intent_payload = trace.get("stabilized_intent")
    if ambiguity_payload and intent_payload:
        missing_variables = [mv["variable"] for mv in ambiguity_payload.get("missing_variables", [])]
        records.append(
            {
                "record_id": _record_id(result, rollout_id, "intent_modeling"),
                "rollout_id": rollout_id,
                "example_id": result.example_id,
                "dataset_name": result.dataset_name,
                "split_name": result.split_name,
                "teacher_model_id": result.task_model_id,
                "stage_name": "intent_modeling",
                "messages": intent_modeling_prompt(
                    request=result.user_request,
                    ambiguity_rationale=ambiguity_payload.get("rationale", ""),
                    missing_variables=missing_variables,
                    schema_text=schema_instruction(IntentModelSchema),
                ),
                "assistant_target": _assistant_payload(intent_payload),
                "response_strategy": result.response_strategy,
                "intent_confidence": result.intent_confidence,
                "confidence_band": result.confidence_band,
            }
        )

    strategy_payload = trace.get("stabilized_strategy")
    if ambiguity_payload and intent_payload and strategy_payload:
        records.append(
            {
                "record_id": _record_id(result, rollout_id, "strategy_selection"),
                "rollout_id": rollout_id,
                "example_id": result.example_id,
                "dataset_name": result.dataset_name,
                "split_name": result.split_name,
                "teacher_model_id": result.task_model_id,
                "stage_name": "strategy_selection",
                "messages": strategy_selection_prompt(
                    request=result.user_request,
                    is_ambiguous=bool(ambiguity_payload.get("is_ambiguous", False)),
                    num_missing_variables=len(ambiguity_payload.get("missing_variables", [])),
                    entropy=intent_payload.get("entropy_estimate", "medium"),
                    gap_description=intent_payload.get("gap_description", ""),
                    schema_text=schema_instruction(StrategyDecisionSchema),
                ),
                "assistant_target": _assistant_payload(strategy_payload),
                "response_strategy": result.response_strategy,
                "intent_confidence": result.intent_confidence,
                "confidence_band": result.confidence_band,
            }
        )

    clarification_payload = trace.get("stabilized_clarification_target")
    if clarification_payload and intent_payload:
        interp_lines = []
        for index, interp in enumerate(intent_payload.get("interpretations", []), start=1):
            interp_lines.append(
                f"{index}. {interp['description']} (plausibility={float(interp.get('plausibility', 0.5)):.2f})"
            )
        target_variable = clarification_payload.get("target_variable") or "general context"
        records.append(
            {
                "record_id": _record_id(result, rollout_id, "clarification_generation"),
                "rollout_id": rollout_id,
                "example_id": result.example_id,
                "dataset_name": result.dataset_name,
                "split_name": result.split_name,
                "teacher_model_id": result.task_model_id,
                "stage_name": "clarification_generation",
                "messages": clarification_question_prompt(
                    request=result.user_request,
                    target_variable=target_variable,
                    interpretations_summary="\n".join(interp_lines),
                    schema_text=schema_instruction(ClarificationQuestionSchema),
                ),
                "assistant_target": _assistant_payload(clarification_payload),
                "response_strategy": result.response_strategy,
                "intent_confidence": result.intent_confidence,
                "confidence_band": result.confidence_band,
            }
        )

    if result.response_strategy == "answer_directly" and result.final_answer:
        records.append(
            {
                "record_id": _record_id(result, rollout_id, "final_answer_direct"),
                "rollout_id": rollout_id,
                "example_id": result.example_id,
                "dataset_name": result.dataset_name,
                "split_name": result.split_name,
                "teacher_model_id": result.task_model_id,
                "stage_name": "final_answer_direct",
                "messages": direct_answer_prompt(
                    request=result.user_request,
                    schema_text=schema_instruction(AnswerSchema),
                ),
                "assistant_target": _assistant_payload(
                    {
                        "answer": result.final_answer,
                        "assumed_interpretation": result.assumed_interpretation,
                        "confidence": result.confidence,
                        "caveats": None,
                    }
                ),
                "response_strategy": result.response_strategy,
                "intent_confidence": result.intent_confidence,
                "confidence_band": result.confidence_band,
            }
        )

    if result.response_strategy == "narrow_and_answer" and result.final_answer and result.assumed_interpretation:
        records.append(
            {
                "record_id": _record_id(result, rollout_id, "final_answer_narrowed"),
                "rollout_id": rollout_id,
                "example_id": result.example_id,
                "dataset_name": result.dataset_name,
                "split_name": result.split_name,
                "teacher_model_id": result.task_model_id,
                "stage_name": "final_answer_narrowed",
                "messages": narrowed_answer_prompt(
                    request=result.user_request,
                    assumed_interpretation=result.assumed_interpretation,
                    schema_text=schema_instruction(NarrowedAnswerSchema),
                ),
                "assistant_target": _assistant_payload(
                    {
                        "stated_assumption": result.assumed_interpretation,
                        "answer": result.final_answer,
                        "confidence": result.confidence,
                        "caveats": None,
                    }
                ),
                "response_strategy": result.response_strategy,
                "intent_confidence": result.intent_confidence,
                "confidence_band": result.confidence_band,
            }
        )

    if result.response_strategy == "present_alternatives" and intent_payload:
        interps = [
            {"interpretation": interp["description"], "assumed_context": interp.get("assumed_context", "")}
            for interp in intent_payload.get("interpretations", [])
        ]
        records.append(
            {
                "record_id": _record_id(result, rollout_id, "final_answer_alternatives"),
                "rollout_id": rollout_id,
                "example_id": result.example_id,
                "dataset_name": result.dataset_name,
                "split_name": result.split_name,
                "teacher_model_id": result.task_model_id,
                "stage_name": "final_answer_alternatives",
                "messages": alternatives_prompt(
                    request=result.user_request,
                    interpretations_json=json.dumps(interps, indent=2),
                    schema_text=schema_instruction(AlternativesResponseSchema),
                ),
                "assistant_target": result.response_text,
                "response_strategy": result.response_strategy,
                "intent_confidence": result.intent_confidence,
                "confidence_band": result.confidence_band,
            }
        )

    return records


def build_preference_pair(
    teacher_result: MethodResult,
    rejected_result: MethodResult,
    *,
    rollout_id: str,
) -> dict[str, Any] | None:
    if teacher_result.response_text.strip() == rejected_result.response_text.strip():
        return None
    return {
        "pair_id": f"{rollout_id}:{teacher_result.example_id}",
        "example_id": teacher_result.example_id,
        "dataset_name": teacher_result.dataset_name,
        "split_name": teacher_result.split_name,
        "teacher_model_id": teacher_result.task_model_id,
        "messages": [
            {
                "role": "system",
                "content": "Choose the best assistant response for understanding and addressing the user's intent.",
            },
            {"role": "user", "content": teacher_result.user_request},
        ],
        "chosen": teacher_result.response_text,
        "rejected": rejected_result.response_text,
        "chosen_strategy": teacher_result.response_strategy,
        "rejected_strategy": rejected_result.response_strategy,
        "teacher_correct": teacher_result.correct,
        "rejected_correct": rejected_result.correct,
    }


def index_predictions_by_example(rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {row["example_id"]: row for row in rows}

