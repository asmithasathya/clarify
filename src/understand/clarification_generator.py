"""Generate targeted clarification questions or narrowed responses."""

from __future__ import annotations

import json
from typing import Any

from src.llm.generator import BaseGenerator
from src.llm.prompts import (
    alternatives_prompt,
    clarification_question_prompt,
    direct_answer_prompt,
    narrowed_answer_prompt,
    schema_instruction,
)
from src.llm.schemas import (
    AlternativesResponseSchema,
    AnswerSchema,
    ClarificationQuestionSchema,
    IntentModelSchema,
    NarrowedAnswerSchema,
)


# ---------------------------------------------------------------------------
# Heuristic fallbacks
# ---------------------------------------------------------------------------

def _fallback_clarification(target_variable: str, request: str) -> ClarificationQuestionSchema:
    return ClarificationQuestionSchema(
        question=f"Could you tell me more about {target_variable}? That would help me give you a better answer.",
        target_variable=target_variable,
        why_this_helps=f"Resolving '{target_variable}' would disambiguate the request.",
    )


def _fallback_answer(request: str) -> AnswerSchema:
    return AnswerSchema(
        answer=f"Here is a general response to your request: '{request[:120]}'.",
        assumed_interpretation=None,
        confidence=0.3,
        caveats="This is a fallback response without LLM generation.",
    )


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class ClarificationGenerator:
    def __init__(self, config: dict[str, Any], generator: BaseGenerator | None = None) -> None:
        self.config = config
        self.generator = generator

    # -- Targeted clarification question -----------------------------------

    def generate_clarification(
        self,
        request: str,
        target_variable: str,
        intent_model: IntentModelSchema,
    ) -> ClarificationQuestionSchema:
        """Generate a single targeted clarifying question."""
        if self.generator is None:
            return _fallback_clarification(target_variable, request)

        interp_lines = []
        for idx, interp in enumerate(intent_model.interpretations, 1):
            interp_lines.append(f"{idx}. {interp.description} (plausibility={interp.plausibility:.2f})")
        interp_summary = "\n".join(interp_lines)

        return self.generator.generate_structured(
            clarification_question_prompt(
                request=request,
                target_variable=target_variable,
                interpretations_summary=interp_summary,
                schema_text=schema_instruction(ClarificationQuestionSchema),
            ),
            ClarificationQuestionSchema,
            temperature=0.0,
        )

    # -- Direct answer -----------------------------------------------------

    def generate_direct_answer(self, request: str) -> AnswerSchema:
        """Answer the request directly with no special handling."""
        if self.generator is None:
            return _fallback_answer(request)

        return self.generator.generate_structured(
            direct_answer_prompt(
                request=request,
                schema_text=schema_instruction(AnswerSchema),
            ),
            AnswerSchema,
            temperature=0.0,
        )

    # -- Narrowed answer ---------------------------------------------------

    def generate_narrowed_answer(
        self,
        request: str,
        assumed_interpretation: str,
    ) -> NarrowedAnswerSchema:
        """Answer under an explicitly stated assumption."""
        if self.generator is None:
            return NarrowedAnswerSchema(
                stated_assumption=assumed_interpretation,
                answer=f"Assuming '{assumed_interpretation}': general response to '{request[:80]}'.",
                confidence=0.3,
                caveats="Fallback response.",
            )

        return self.generator.generate_structured(
            narrowed_answer_prompt(
                request=request,
                assumed_interpretation=assumed_interpretation,
                schema_text=schema_instruction(NarrowedAnswerSchema),
            ),
            NarrowedAnswerSchema,
            temperature=0.0,
        )

    # -- Present alternatives -----------------------------------------------

    def generate_alternatives(
        self,
        request: str,
        intent_model: IntentModelSchema,
    ) -> AlternativesResponseSchema:
        """Present multiple interpretations each with a short answer."""
        interps = [
            {"interpretation": i.description, "assumed_context": i.assumed_context}
            for i in intent_model.interpretations
        ]
        if self.generator is None:
            alts = [
                {"interpretation": i["interpretation"], "answer": f"Answer for: {i['interpretation'][:60]}"}
                for i in interps[:3]
            ]
            if len(alts) < 2:
                alts.append({"interpretation": "alternative interpretation", "answer": "alternative answer"})
            return AlternativesResponseSchema(
                preamble="Your request could mean several things:",
                alternatives=alts,
            )

        return self.generator.generate_structured(
            alternatives_prompt(
                request=request,
                interpretations_json=json.dumps(interps, indent=2),
                schema_text=schema_instruction(AlternativesResponseSchema),
            ),
            AlternativesResponseSchema,
            temperature=0.0,
        )
