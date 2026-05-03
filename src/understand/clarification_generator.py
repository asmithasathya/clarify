"""Generate targeted clarification questions or narrowed responses."""

from __future__ import annotations

import json
from typing import Any

from src.llm.generator import BaseGenerator
from src.llm.prompts import (
    alternatives_prompt,
    clarification_question_prompt,
    conversation_answer_prompt,
    direct_answer_prompt,
    generic_clarification_prompt,
    narrowed_answer_prompt,
    simulate_user_reply_prompt,
    schema_instruction,
)
from src.llm.schemas import (
    AlternativesResponseSchema,
    AnswerSchema,
    ClarificationQuestionSchema,
    IntentModelSchema,
    NarrowedAnswerSchema,
    UserReplySchema,
)
from src.utils.logging import get_logger


LOGGER = get_logger(__name__)


# ---------------------------------------------------------------------------
# Heuristic fallbacks
# ---------------------------------------------------------------------------

def _fallback_clarification(target_variable: str, request: str) -> ClarificationQuestionSchema:
    return ClarificationQuestionSchema(
        question=f"Could you tell me more about {target_variable}? That would help me give you a better answer.",
        target_variable=target_variable,
        why_this_helps=f"Resolving '{target_variable}' would disambiguate the request.",
    )


def _fallback_generic_clarification() -> ClarificationQuestionSchema:
    return ClarificationQuestionSchema(
        question="Could you share a bit more detail about what you need?",
        target_variable="general context",
        why_this_helps="More context would make the request easier to answer well.",
    )


def _fallback_answer(request: str) -> AnswerSchema:
    return AnswerSchema(
        answer=f"Here is a general response to your request: '{request[:120]}'.",
        assumed_interpretation=None,
        confidence=0.3,
        caveats="This is a fallback response without LLM generation.",
    )


def _followup_answer_max_new_tokens(config: dict[str, Any]) -> int:
    generation_cfg = config.get("generation", {})
    return int(generation_cfg.get("followup_max_new_tokens", 768))


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
        *,
        temperature: float | None = None,
    ) -> ClarificationQuestionSchema:
        """Generate a single targeted clarifying question."""
        use_targeted_question = self.config.get("ablations", {}).get("targeted_question", True)
        if self.generator is None:
            if not use_targeted_question:
                return _fallback_generic_clarification()
            return _fallback_clarification(target_variable, request)

        interp_lines = []
        for idx, interp in enumerate(intent_model.interpretations, 1):
            interp_lines.append(f"{idx}. {interp.description} (plausibility={interp.plausibility:.2f})")
        interp_summary = "\n".join(interp_lines)

        prompt = (
            clarification_question_prompt(
                request=request,
                target_variable=target_variable,
                interpretations_summary=interp_summary,
                schema_text=schema_instruction(ClarificationQuestionSchema),
            )
            if use_targeted_question
            else generic_clarification_prompt(
                request=request,
                schema_text=schema_instruction(ClarificationQuestionSchema),
            )
        )
        try:
            return self.generator.generate_structured(
                prompt,
                ClarificationQuestionSchema,
                temperature=temperature if temperature is not None else 0.0,
            )
        except Exception as exc:
            LOGGER.warning("Falling back to heuristic clarification question after generation failure: %s", exc)
            if not use_targeted_question:
                return _fallback_generic_clarification()
            return _fallback_clarification(target_variable, request)

    # -- Direct answer -----------------------------------------------------

    def generate_direct_answer(self, request: str) -> AnswerSchema:
        """Answer the request directly with no special handling."""
        if self.generator is None:
            return _fallback_answer(request)

        try:
            return self.generator.generate_structured(
                direct_answer_prompt(
                    request=request,
                    schema_text=schema_instruction(AnswerSchema),
                ),
                AnswerSchema,
                temperature=0.0,
            )
        except Exception as exc:
            LOGGER.warning("Falling back to heuristic direct answer after generation failure: %s", exc)
            return _fallback_answer(request)

    def generate_conversation_answer(
        self,
        conversation: list[dict[str, str]],
        *,
        temperature: float | None = None,
    ) -> AnswerSchema:
        """Answer after a clarification follow-up in a multi-turn setting."""
        if self.generator is None:
            latest_user = conversation[-1]["content"] if conversation else ""
            return _fallback_answer(latest_user)

        try:
            return self.generator.generate_structured(
                conversation_answer_prompt(
                    conversation=conversation,
                    schema_text=schema_instruction(AnswerSchema),
                ),
                AnswerSchema,
                temperature=temperature if temperature is not None else 0.0,
                max_new_tokens=_followup_answer_max_new_tokens(self.config),
            )
        except Exception as exc:
            LOGGER.warning("Falling back to heuristic follow-up answer after generation failure: %s", exc)
            return _fallback_answer(latest_user)

    def simulate_user_reply(
        self,
        request: str,
        assistant_message: str,
        hidden_context: str,
        *,
        temperature: float | None = None,
    ) -> UserReplySchema:
        """Produce a synthetic user follow-up grounded in hidden context."""
        if self.generator is None:
            return UserReplySchema(
                user_reply=f"Here are the relevant details: {hidden_context}",
                grounded_in_hidden_context=True,
            )

        try:
            return self.generator.generate_structured(
                simulate_user_reply_prompt(
                    request=request,
                    assistant_message=assistant_message,
                    hidden_context=hidden_context,
                    schema_text=schema_instruction(UserReplySchema),
                ),
                UserReplySchema,
                temperature=temperature if temperature is not None else 0.0,
            )
        except Exception as exc:
            LOGGER.warning("Falling back to heuristic user reply simulation after generation failure: %s", exc)
            return UserReplySchema(
                user_reply=f"Here are the relevant details: {hidden_context}",
                grounded_in_hidden_context=True,
            )

    # -- Narrowed answer ---------------------------------------------------

    def generate_narrowed_answer(
        self,
        request: str,
        assumed_interpretation: str,
        *,
        temperature: float | None = None,
    ) -> NarrowedAnswerSchema:
        """Answer under an explicitly stated assumption."""
        if self.generator is None:
            return NarrowedAnswerSchema(
                stated_assumption=assumed_interpretation,
                answer=f"Assuming '{assumed_interpretation}': general response to '{request[:80]}'.",
                confidence=0.3,
                caveats="Fallback response.",
            )

        try:
            return self.generator.generate_structured(
                narrowed_answer_prompt(
                    request=request,
                    assumed_interpretation=assumed_interpretation,
                    schema_text=schema_instruction(NarrowedAnswerSchema),
                ),
                NarrowedAnswerSchema,
                temperature=temperature if temperature is not None else 0.0,
            )
        except Exception as exc:
            LOGGER.warning("Falling back to heuristic narrowed answer after generation failure: %s", exc)
            return NarrowedAnswerSchema(
                stated_assumption=assumed_interpretation,
                answer=f"Assuming '{assumed_interpretation}': general response to '{request[:80]}'.",
                confidence=0.3,
                caveats="Fallback response.",
            )

    # -- Present alternatives -----------------------------------------------

    def generate_alternatives(
        self,
        request: str,
        intent_model: IntentModelSchema,
        *,
        temperature: float | None = None,
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

        try:
            return self.generator.generate_structured(
                alternatives_prompt(
                    request=request,
                    interpretations_json=json.dumps(interps, indent=2),
                    schema_text=schema_instruction(AlternativesResponseSchema),
                ),
                AlternativesResponseSchema,
                temperature=temperature if temperature is not None else 0.0,
            )
        except Exception as exc:
            LOGGER.warning("Falling back to heuristic alternatives response after generation failure: %s", exc)
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
