"""Baseline: answer with hedging language when confidence is low."""

from __future__ import annotations

from typing import Any

from src.data.schema import DialogueExample, MethodResult
from src.llm.generator import BaseGenerator
from src.llm.prompts import hedged_answer_prompt, schema_instruction
from src.llm.schemas import HedgedAnswerSchema


def run_generic_hedge(
    example: DialogueExample,
    *,
    generator: BaseGenerator,
    config: dict[str, Any],
) -> MethodResult:
    response = generator.generate_structured(
        hedged_answer_prompt(
            request=example.user_request,
            schema_text=schema_instruction(HedgedAnswerSchema),
        ),
        HedgedAnswerSchema,
        temperature=0.0,
    )

    return MethodResult(
        example_id=example.example_id,
        method="generic_hedge",
        user_request=example.user_request,
        hidden_context=example.hidden_context,
        gold_clarification_needed=example.gold_clarification_needed,
        gold_answer=example.gold_answer,
        response_strategy="answer_directly",
        response_text=response.answer,
        final_answer=response.answer,
        is_ambiguous_detected=False,
        confidence=response.confidence,
        answered_directly=True,
        trace={
            "hedged_answer": response.model_dump(),
        },
    )
