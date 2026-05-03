"""Baseline: always ask a generic clarification question regardless of ambiguity."""

from __future__ import annotations

from typing import Any

from src.data.schema import DialogueExample, MethodResult
from src.llm.generator import BaseGenerator
from src.llm.prompts import generic_clarification_prompt, schema_instruction
from src.llm.schemas import ClarificationQuestionSchema


def run_generic_clarify(
    example: DialogueExample,
    *,
    generator: BaseGenerator,
    config: dict[str, Any],
) -> MethodResult:
    response = generator.generate_structured(
        generic_clarification_prompt(
            request=example.user_request,
            schema_text=schema_instruction(ClarificationQuestionSchema),
        ),
        ClarificationQuestionSchema,
        temperature=0.0,
    )

    return MethodResult(
        example_id=example.example_id,
        method="generic_clarify",
        user_request=example.user_request,
        hidden_context=example.hidden_context,
        gold_clarification_needed=example.gold_clarification_needed,
        gold_answer=example.gold_answer,
        response_strategy="ask_clarification",
        response_text=response.question,
        clarification_question=response.question,
        is_ambiguous_detected=True,  # assumes everything is ambiguous
        confidence=0.5,
        asked_clarification=True,
        trace={
            "generic_clarification": response.model_dump(),
        },
    )
