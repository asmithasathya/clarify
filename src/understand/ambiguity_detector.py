"""Detect whether a user request is ambiguous or underspecified."""

from __future__ import annotations

import re
from typing import Any

from src.data.schema import AmbiguityRecord
from src.llm.generator import BaseGenerator
from src.llm.prompts import ambiguity_detection_prompt, schema_instruction
from src.llm.schemas import AmbiguityAssessmentSchema
from src.utils.logging import get_logger


LOGGER = get_logger(__name__)


# ---------------------------------------------------------------------------
# Heuristic fallback
# ---------------------------------------------------------------------------

_WH_WORDS = {"who", "what", "where", "when", "which", "how", "why"}
_VAGUE_MARKERS = {
    "something", "stuff", "things", "help", "issue", "problem",
    "it", "this", "that", "some", "good", "best", "nice",
}


def _word_tokens(text: str) -> list[str]:
    return [t.lower() for t in re.findall(r"[A-Za-z0-9']+", text)]


def heuristic_ambiguity(request: str) -> AmbiguityAssessmentSchema:
    """Quick heuristic-based ambiguity check (no LLM call)."""
    tokens = _word_tokens(request)
    token_set = set(tokens)
    signals: list[str] = []

    # Very short requests are often underspecified
    if len(tokens) <= 4:
        signals.append("very short request")

    # High ratio of vague / pronoun tokens
    vague_count = len(token_set & _VAGUE_MARKERS)
    if vague_count >= 2:
        signals.append("multiple vague references")

    # No wh-word and no verb-like specificity
    if not (token_set & _WH_WORDS) and len(tokens) <= 8:
        signals.append("no specificity markers")

    is_ambiguous = len(signals) >= 1
    missing: list[dict[str, Any]] = []
    if is_ambiguous:
        missing.append({
            "variable": "user context / specific details",
            "why_missing": "; ".join(signals),
            "importance": 0.8,
        })

    return AmbiguityAssessmentSchema(
        is_ambiguous=is_ambiguous,
        ambiguity_type="underspecified" if is_ambiguous else "none",
        missing_variables=missing,
        confidence=0.5,
        rationale="Heuristic: " + ("; ".join(signals) if signals else "request appears specific enough."),
    )


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class AmbiguityDetector:
    def __init__(self, config: dict[str, Any], generator: BaseGenerator | None = None) -> None:
        self.config = config
        self.generator = generator

    def detect(
        self,
        request: str,
        *,
        temperature: float | None = None,
    ) -> AmbiguityAssessmentSchema:
        """Detect ambiguity in a user request.

        Uses the LLM when available, falls back to heuristics otherwise.
        If the ``ambiguity_detection`` ablation flag is disabled, the detector
        always returns "not ambiguous" (forcing downstream methods to answer).
        """
        if not self.config.get("ablations", {}).get("ambiguity_detection", True):
            return AmbiguityAssessmentSchema(
                is_ambiguous=False,
                ambiguity_type="none",
                missing_variables=[],
                confidence=1.0,
                rationale="Ablation: ambiguity detection disabled.",
            )

        if self.generator is None or self.config.get("ablations", {}).get("heuristic_ambiguity_detection", False):
            return heuristic_ambiguity(request)

        try:
            return self.generator.generate_structured(
                ambiguity_detection_prompt(
                    request=request,
                    schema_text=schema_instruction(AmbiguityAssessmentSchema),
                ),
                AmbiguityAssessmentSchema,
                temperature=temperature if temperature is not None else 0.0,
            )
        except Exception as exc:
            LOGGER.warning("Falling back to heuristic ambiguity detection after generation failure: %s", exc)
            return heuristic_ambiguity(request)

    def to_records(self, assessment: AmbiguityAssessmentSchema) -> list[AmbiguityRecord]:
        """Convert a schema result to a list of AmbiguityRecord objects."""
        records: list[AmbiguityRecord] = []
        for mv in assessment.missing_variables:
            records.append(AmbiguityRecord(
                ambiguity_type=assessment.ambiguity_type if assessment.ambiguity_type != "none" else "underspecified",
                missing_variable=mv.variable,
                confidence=mv.importance,
                source_span=None,
            ))
        return records
