"""Build a lightweight model of candidate user interpretations."""

from __future__ import annotations

from typing import Any

from src.llm.generator import BaseGenerator
from src.llm.prompts import intent_modeling_prompt, schema_instruction
from src.llm.schemas import AmbiguityAssessmentSchema, IntentModelSchema, InterpretationSchema


# ---------------------------------------------------------------------------
# Heuristic fallback
# ---------------------------------------------------------------------------

def heuristic_intent_model(request: str, assessment: AmbiguityAssessmentSchema) -> IntentModelSchema:
    """Produce a minimal intent model without an LLM."""
    missing = [mv.variable for mv in assessment.missing_variables]
    gap = "; ".join(missing) if missing else "unknown gap"

    interp = InterpretationSchema(
        description=f"Default interpretation of: {request[:120]}",
        assumed_context="No additional context assumed.",
        plausibility=0.5,
    )
    return IntentModelSchema(
        interpretations=[interp],
        most_likely_index=0,
        entropy_estimate="high" if assessment.is_ambiguous else "low",
        gap_description=gap,
    )


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class IntentModeler:
    def __init__(self, config: dict[str, Any], generator: BaseGenerator | None = None) -> None:
        self.config = config
        self.generator = generator

    def model(
        self,
        request: str,
        assessment: AmbiguityAssessmentSchema,
    ) -> IntentModelSchema:
        """Produce candidate interpretations of the user's request.

        If the ``intent_modeling`` ablation flag is disabled, returns a single
        default interpretation (forcing downstream to skip interpretation ranking).
        """
        if not self.config.get("ablations", {}).get("intent_modeling", True):
            return heuristic_intent_model(request, assessment)

        if self.generator is None:
            return heuristic_intent_model(request, assessment)

        missing_vars = [mv.variable for mv in assessment.missing_variables]

        return self.generator.generate_structured(
            intent_modeling_prompt(
                request=request,
                ambiguity_rationale=assessment.rationale,
                missing_variables=missing_vars,
                schema_text=schema_instruction(IntentModelSchema),
            ),
            IntentModelSchema,
            temperature=0.0,
        )
