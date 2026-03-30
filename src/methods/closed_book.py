"""Closed-book legal QA baseline."""

from __future__ import annotations

from typing import Any

from src.data.schema import MethodResult, confidence_bucket
from src.llm.generator import BaseGenerator
from src.llm.prompts import closed_book_answer_prompt, schema_instruction
from src.llm.schemas import ClosedBookAnswerSchema


def run_closed_book(
    example: Any,
    *,
    generator: BaseGenerator,
    config: dict[str, Any],
) -> MethodResult:
    response = generator.generate_structured(
        closed_book_answer_prompt(
            question=example.question,
            state=example.state,
            year=config.get("project", {}).get("year", 2021),
            schema_text=schema_instruction(ClosedBookAnswerSchema),
        ),
        ClosedBookAnswerSchema,
        temperature=0.0,
    )
    answer = response.answer
    conf = float(response.confidence)
    return MethodResult(
        example_id=example.example_id,
        method="closed_book",
        question=example.question,
        gold_answer=example.answer,
        predicted_answer=answer,
        explanation=response.explanation,
        state=example.state,
        confidence=conf,
        confidence_bucket=response.confidence_bucket or confidence_bucket(conf),
        citations=[],
        statute_ids=[],
        abstained=(answer == "Abstain"),
        narrowed=False,
        correct=(answer == example.answer and answer != "Abstain"),
        support_score=0.0 if answer != "Abstain" else None,
        trace={"raw_closed_book": response.model_dump()},
    )

