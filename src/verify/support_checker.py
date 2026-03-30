"""Optional generator-assisted support checking."""

from __future__ import annotations

from typing import Any, Sequence

from src.data.schema import RetrievedPassage
from src.llm.generator import BaseGenerator
from src.llm.prompts import claim_support_judging_prompt, schema_instruction
from src.llm.schemas import SupportJudgmentSchema


class LLMSupportJudge:
    def __init__(self, config: dict[str, Any], generator: BaseGenerator | None = None) -> None:
        self.config = config
        self.generator = generator

    def judge(
        self,
        *,
        question: str,
        claim_text: str,
        state: str,
        evidence: Sequence[RetrievedPassage],
    ) -> SupportJudgmentSchema | None:
        if self.generator is None:
            return None
        return self.generator.generate_structured(
            claim_support_judging_prompt(
                question=question,
                claim_text=claim_text,
                state=state,
                year=self.config.get("project", {}).get("year", 2021),
                evidence=evidence,
                schema_text=schema_instruction(SupportJudgmentSchema),
            ),
            SupportJudgmentSchema,
            temperature=0.0,
        )

