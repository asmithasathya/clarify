"""Decision policy for answering, narrowing, or abstaining."""

from __future__ import annotations

from typing import Any

from src.data.schema import AbstentionDecision
from src.llm.generator import BaseGenerator
from src.llm.prompts import abstention_prompt, schema_instruction
from src.llm.schemas import AbstentionSchema


def _template_narrow_answer(predicted_answer: str, missing_verification: str) -> AbstentionDecision:
    return AbstentionDecision(
        predicted_answer=predicted_answer,
        explanation=(
            f"{predicted_answer}. The core answer is plausible, but the supporting explanation is only partial. "
            f"This should be treated narrowly pending verification of: {missing_verification}"
        ),
        narrowed=True,
        abstained=False,
        requested_verification=missing_verification,
    )


def _template_abstain(missing_verification: str) -> AbstentionDecision:
    return AbstentionDecision(
        predicted_answer="Abstain",
        explanation=(
            "Abstain. The available statute evidence does not sufficiently support the answer. "
            f"Needed verification: {missing_verification}"
        ),
        narrowed=False,
        abstained=True,
        requested_verification=missing_verification,
    )


class AbstentionPolicy:
    def __init__(self, config: dict[str, Any], generator: BaseGenerator | None = None) -> None:
        self.config = config
        self.generator = generator

    def decide(
        self,
        *,
        question: str,
        state: str,
        predicted_answer: str,
        current_explanation: str,
        final_support_score: float,
        missing_verification: str,
    ) -> AbstentionDecision:
        if not self.config.get("ablations", {}).get("abstain", True):
            return AbstentionDecision(
                predicted_answer=predicted_answer,
                explanation=current_explanation,
                narrowed=False,
                abstained=(predicted_answer == "Abstain"),
            )

        verify_cfg = self.config.get("verify", {})
        support_threshold = verify_cfg.get("support_threshold", 0.62)
        narrow_threshold = verify_cfg.get("narrow_threshold", 0.48)

        if final_support_score >= support_threshold:
            return AbstentionDecision(
                predicted_answer=predicted_answer,
                explanation=current_explanation,
                narrowed=False,
                abstained=(predicted_answer == "Abstain"),
            )

        if predicted_answer != "Abstain" and final_support_score >= narrow_threshold:
            if self.generator is None:
                return _template_narrow_answer(predicted_answer, missing_verification)
            response = self.generator.generate_structured(
                abstention_prompt(
                    question=question,
                    state=state,
                    year=self.config.get("project", {}).get("year", 2021),
                    predicted_answer=predicted_answer,
                    missing_verification=missing_verification,
                    schema_text=schema_instruction(AbstentionSchema),
                ),
                AbstentionSchema,
                temperature=0.0,
            )
            return AbstentionDecision(
                predicted_answer=response.answer,
                explanation=response.explanation,
                narrowed=response.narrow_answer,
                abstained=response.answer == "Abstain",
                requested_verification=response.requested_verification,
            )

        if self.generator is None:
            return _template_abstain(missing_verification)
        response = self.generator.generate_structured(
            abstention_prompt(
                question=question,
                state=state,
                year=self.config.get("project", {}).get("year", 2021),
                predicted_answer=predicted_answer,
                missing_verification=missing_verification,
                schema_text=schema_instruction(AbstentionSchema),
            ),
            AbstentionSchema,
            temperature=0.0,
        )
        return AbstentionDecision(
            predicted_answer=response.answer,
            explanation=response.explanation,
            narrowed=response.narrow_answer,
            abstained=response.answer == "Abstain",
            requested_verification=response.requested_verification,
        )
