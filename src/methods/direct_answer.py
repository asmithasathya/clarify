"""Baseline: answer immediately with no ambiguity checking."""

from __future__ import annotations

from typing import Any

from src.data.schema import DialogueExample, MethodResult
from src.llm.generator import BaseGenerator
from src.llm.prompts import direct_answer_prompt, schema_instruction
from src.llm.schemas import AnswerSchema


def run_direct_answer(
    example: DialogueExample,
    *,
    generator: BaseGenerator,
    config: dict[str, Any],
) -> MethodResult:
    response = generator.generate_structured(
        direct_answer_prompt(
            request=example.user_request,
            schema_text=schema_instruction(AnswerSchema),
        ),
        AnswerSchema,
        temperature=0.0,
    )

    return MethodResult(
        example_id=example.example_id,
        method="direct_answer",
        user_request=example.user_request,
        hidden_context=example.hidden_context,
        gold_clarification_needed=example.gold_clarification_needed,
        gold_answer=example.gold_answer,
        response_strategy="answer_directly",
        response_text=response.answer,
        final_answer=response.answer,
        assumed_interpretation=response.assumed_interpretation,
        is_ambiguous_detected=False,
        confidence=response.confidence,
        answered_directly=True,
        trace={"raw_answer": response.model_dump()},
    )
