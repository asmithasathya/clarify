"""Main method: detect ambiguity, model intent, choose strategy, act."""

from __future__ import annotations

from typing import Any

from src.data.schema import DialogueExample, MethodResult
from src.understand.ambiguity_detector import AmbiguityDetector
from src.understand.clarification_generator import ClarificationGenerator
from src.understand.intent_model import IntentModeler
from src.understand.strategy_selector import StrategySelector


def _resolve_interpretation_index(trace: dict[str, Any], raw_index: int, count: int) -> int:
    if count <= 0:
        raise ValueError("Intent model returned no interpretations to choose from.")
    if 0 <= raw_index < count:
        return raw_index
    if 1 <= raw_index <= count:
        resolved = raw_index - 1
        trace["most_likely_index_adjustment"] = {
            "raw_index": raw_index,
            "resolved_index": resolved,
            "reason": "converted_one_based_index",
        }
        return resolved

    trace["most_likely_index_adjustment"] = {
        "raw_index": raw_index,
        "resolved_index": 0,
        "reason": "out_of_range_fallback",
    }
    return 0


def run_targeted_clarify(
    example: DialogueExample,
    *,
    config: dict[str, Any],
    ambiguity_detector: AmbiguityDetector,
    intent_modeler: IntentModeler,
    strategy_selector: StrategySelector,
    clarification_generator: ClarificationGenerator,
) -> MethodResult:
    trace: dict[str, Any] = {}

    # Step 1: detect ambiguity
    assessment = ambiguity_detector.detect(example.user_request)
    trace["ambiguity_assessment"] = assessment.model_dump()

    # Step 2: fast path — if not ambiguous, answer directly
    if not assessment.is_ambiguous:
        answer = clarification_generator.generate_direct_answer(example.user_request)
        trace["direct_answer"] = answer.model_dump()
        return MethodResult(
            example_id=example.example_id,
            method="targeted_clarify",
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

    # Step 3: model candidate interpretations
    intent = intent_modeler.model(example.user_request, assessment)
    trace["intent_model"] = intent.model_dump()

    # Step 4: select strategy
    decision = strategy_selector.select(example.user_request, assessment, intent)
    trace["strategy_decision"] = decision.model_dump()

    strategy = decision.strategy
    n_missing = len(assessment.missing_variables)

    # Step 5: execute chosen strategy
    if strategy == "ask_clarification":
        target_var = (
            assessment.missing_variables[0].variable
            if assessment.missing_variables
            else intent.gap_description
        )
        cq = clarification_generator.generate_clarification(
            example.user_request, target_var, intent,
        )
        trace["clarification_question"] = cq.model_dump()
        return MethodResult(
            example_id=example.example_id,
            method="targeted_clarify",
            user_request=example.user_request,
            hidden_context=example.hidden_context,
            gold_clarification_needed=example.gold_clarification_needed,
            gold_answer=example.gold_answer,
            response_strategy="ask_clarification",
            response_text=cq.question,
            clarification_question=cq.question,
            is_ambiguous_detected=True,
            confidence=decision.confidence,
            num_missing_variables=n_missing,
            asked_clarification=True,
            trace=trace,
        )

    if strategy == "narrow_and_answer":
        resolved_index = _resolve_interpretation_index(
            trace,
            intent.most_likely_index,
            len(intent.interpretations),
        )
        most_likely = intent.interpretations[resolved_index]
        narrowed = clarification_generator.generate_narrowed_answer(
            example.user_request, most_likely.description,
        )
        trace["narrowed_answer"] = narrowed.model_dump()
        return MethodResult(
            example_id=example.example_id,
            method="targeted_clarify",
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
            num_missing_variables=n_missing,
            trace=trace,
        )

    if strategy == "present_alternatives":
        alts = clarification_generator.generate_alternatives(example.user_request, intent)
        trace["alternatives"] = alts.model_dump()
        alt_text = alts.preamble + "\n"
        for i, alt in enumerate(alts.alternatives, 1):
            alt_text += f"\n{i}. If you mean '{alt.interpretation}': {alt.answer}"
        return MethodResult(
            example_id=example.example_id,
            method="targeted_clarify",
            user_request=example.user_request,
            hidden_context=example.hidden_context,
            gold_clarification_needed=example.gold_clarification_needed,
            gold_answer=example.gold_answer,
            response_strategy="present_alternatives",
            response_text=alt_text,
            is_ambiguous_detected=True,
            confidence=decision.confidence,
            num_missing_variables=n_missing,
            trace=trace,
        )

    # strategy == "abstain"
    return MethodResult(
        example_id=example.example_id,
        method="targeted_clarify",
        user_request=example.user_request,
        hidden_context=example.hidden_context,
        gold_clarification_needed=example.gold_clarification_needed,
        gold_answer=example.gold_answer,
        response_strategy="abstain",
        response_text="I'm sorry, but your request is too vague for me to help meaningfully. Could you rephrase with more specific details?",
        is_ambiguous_detected=True,
        confidence=decision.confidence,
        num_missing_variables=n_missing,
        trace=trace,
    )
