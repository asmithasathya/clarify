"""Minimal answer revision around the weakest claim."""

from __future__ import annotations

import json
from typing import Any, Sequence

from src.data.schema import RetrievedPassage
from src.llm.generator import BaseGenerator
from src.llm.prompts import minimal_revision_prompt, schema_instruction
from src.llm.schemas import RagAnswerSchema, RevisionSchema


class RevisionEngine:
    def __init__(self, config: dict[str, Any], generator: BaseGenerator | None = None) -> None:
        self.config = config
        self.generator = generator

    def revise(
        self,
        *,
        question: str,
        state: str,
        weak_claim: str,
        initial_answer: RagAnswerSchema,
        new_evidence: Sequence[RetrievedPassage],
    ) -> RevisionSchema:
        if self.generator is None:
            return RevisionSchema(
                answer=initial_answer.answer,
                explanation=initial_answer.explanation,
                cited_statute_ids=initial_answer.cited_statute_ids,
                citations=initial_answer.citations,
                confidence=initial_answer.confidence,
                confidence_bucket=initial_answer.confidence_bucket,
                revision_notes="Generator unavailable; no revision applied.",
            )

        return self.generator.generate_structured(
            minimal_revision_prompt(
                question=question,
                state=state,
                year=self.config.get("project", {}).get("year", 2021),
                initial_answer_json=json.dumps(initial_answer.model_dump(), ensure_ascii=True),
                weak_claim=weak_claim,
                new_evidence=new_evidence,
                schema_text=schema_instruction(RevisionSchema),
            ),
            RevisionSchema,
            temperature=0.0,
        )

