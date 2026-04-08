"""Choose the best response strategy given an ambiguity assessment."""

from __future__ import annotations

from typing import Any

from src.data.schema import ResponseStrategy
from src.llm.generator import BaseGenerator
from src.llm.prompts import schema_instruction, strategy_selection_prompt
from src.llm.schemas import AmbiguityAssessmentSchema, IntentModelSchema, StrategyDecisionSchema


# ---------------------------------------------------------------------------
# Heuristic fallback
# ---------------------------------------------------------------------------

def heuristic_strategy(
    assessment: AmbiguityAssessmentSchema,
    intent_model: IntentModelSchema,
    config: dict[str, Any],
) -> StrategyDecisionSchema:
    """Rule-based strategy selection (no LLM)."""
    threshold = config.get("understand", {}).get("ambiguity_threshold", 0.5)

    if not assessment.is_ambiguous or assessment.confidence < threshold:
        return StrategyDecisionSchema(
            strategy="answer_directly",
            rationale="Request appears clear enough based on ambiguity assessment.",
            confidence=assessment.confidence,
        )

    n_missing = len(assessment.missing_variables)
    n_interps = len(intent_model.interpretations)

    if n_interps >= 3 and intent_model.entropy_estimate == "high":
        return StrategyDecisionSchema(
            strategy="present_alternatives",
            rationale=f"High entropy with {n_interps} distinct interpretations.",
            confidence=0.6,
        )

    if n_missing >= 1:
        return StrategyDecisionSchema(
            strategy="ask_clarification",
            rationale=f"Key missing variable: {assessment.missing_variables[0].variable}",
            confidence=0.7,
        )

    return StrategyDecisionSchema(
        strategy="narrow_and_answer",
        rationale="Ambiguous but no single dominant missing variable; narrowing is safest.",
        confidence=0.5,
    )


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class StrategySelector:
    def __init__(self, config: dict[str, Any], generator: BaseGenerator | None = None) -> None:
        self.config = config
        self.generator = generator

    def select(
        self,
        request: str,
        assessment: AmbiguityAssessmentSchema,
        intent_model: IntentModelSchema,
    ) -> StrategyDecisionSchema:
        """Choose the response strategy.

        If the ``strategy_selection`` ablation is disabled, the selector always
        returns ``ask_clarification`` when ambiguous and ``answer_directly``
        otherwise (bypassing nuanced strategy choice).
        """
        if not self.config.get("ablations", {}).get("strategy_selection", True):
            strategy: ResponseStrategy = (
                "ask_clarification" if assessment.is_ambiguous else "answer_directly"
            )
            return StrategyDecisionSchema(
                strategy=strategy,
                rationale="Ablation: strategy selection disabled; binary clarify/answer.",
                confidence=0.5,
            )

        if self.generator is None:
            return heuristic_strategy(assessment, intent_model, self.config)

        return self.generator.generate_structured(
            strategy_selection_prompt(
                request=request,
                is_ambiguous=assessment.is_ambiguous,
                num_missing_variables=len(assessment.missing_variables),
                entropy=intent_model.entropy_estimate,
                gap_description=intent_model.gap_description,
                schema_text=schema_instruction(StrategyDecisionSchema),
            ),
            StrategyDecisionSchema,
            temperature=0.0,
        )
