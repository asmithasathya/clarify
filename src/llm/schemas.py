"""Structured output schemas for model-consumed prompts."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Ambiguity detection
# ---------------------------------------------------------------------------

class MissingVariableSchema(BaseModel):
    variable: str
    why_missing: str
    importance: float = Field(ge=0.0, le=1.0)


class AmbiguityAssessmentSchema(BaseModel):
    is_ambiguous: bool
    ambiguity_type: Literal["lexical", "referential", "underspecified", "missing_context", "none"]
    missing_variables: list[MissingVariableSchema] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str


# ---------------------------------------------------------------------------
# Intent modeling
# ---------------------------------------------------------------------------

class InterpretationSchema(BaseModel):
    description: str
    assumed_context: str
    plausibility: float = Field(ge=0.0, le=1.0)


class IntentModelSchema(BaseModel):
    interpretations: list[InterpretationSchema] = Field(min_length=1, max_length=5)
    most_likely_index: int = 0
    entropy_estimate: Literal["low", "medium", "high"]
    gap_description: str


class IntentSampleBundleSchema(BaseModel):
    stage_name: Literal["ambiguity_detection", "intent_modeling", "strategy_selection", "clarification_target"]
    raw_confidence: float = Field(ge=0.0, le=1.0)
    sample_size: int = Field(ge=1)
    agreement: float = Field(ge=0.0, le=1.0)
    dominant_label: str
    supporting_labels: list[str] = Field(default_factory=list)


class IntentWeakPointSchema(BaseModel):
    stage_name: Literal["ambiguity_detection", "intent_modeling", "strategy_selection", "clarification_target"]
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str


class IntentStabilityReportSchema(BaseModel):
    overall_confidence: float = Field(ge=0.0, le=1.0)
    confidence_band: Literal["low", "medium", "high"]
    uncertainty_explanation: str
    stage_reports: list[IntentSampleBundleSchema] = Field(default_factory=list)
    weak_points: list[IntentWeakPointSchema] = Field(default_factory=list)
    should_clarify: bool = False


# ---------------------------------------------------------------------------
# Strategy selection
# ---------------------------------------------------------------------------

class StrategyDecisionSchema(BaseModel):
    strategy: Literal[
        "answer_directly",
        "ask_clarification",
        "narrow_and_answer",
        "present_alternatives",
        "abstain",
    ]
    rationale: str
    confidence: float = Field(ge=0.0, le=1.0)


# ---------------------------------------------------------------------------
# Clarification question generation
# ---------------------------------------------------------------------------

class ClarificationQuestionSchema(BaseModel):
    question: str
    target_variable: str
    why_this_helps: str


# ---------------------------------------------------------------------------
# Answer generation
# ---------------------------------------------------------------------------

class AnswerSchema(BaseModel):
    answer: str
    assumed_interpretation: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    caveats: str | None = None


class UserReplySchema(BaseModel):
    user_reply: str
    grounded_in_hidden_context: bool = True


# ---------------------------------------------------------------------------
# Narrowed answer (answer under explicit assumptions)
# ---------------------------------------------------------------------------

class NarrowedAnswerSchema(BaseModel):
    stated_assumption: str
    answer: str
    confidence: float = Field(ge=0.0, le=1.0)
    caveats: str | None = None


# ---------------------------------------------------------------------------
# Alternatives presentation
# ---------------------------------------------------------------------------

class AlternativeAnswerSchema(BaseModel):
    interpretation: str
    answer: str


class AlternativesResponseSchema(BaseModel):
    preamble: str
    alternatives: list[AlternativeAnswerSchema] = Field(min_length=2, max_length=4)


# ---------------------------------------------------------------------------
# Hedged answer
# ---------------------------------------------------------------------------

class HedgedAnswerSchema(BaseModel):
    answer: str
    hedge_reason: str
    confidence: float = Field(ge=0.0, le=1.0)


class AnswerEvaluationSchema(BaseModel):
    is_correct: bool
    score: float = Field(ge=0.0, le=1.0)
    rationale: str


class ClarificationEvaluationSchema(BaseModel):
    is_targeted: bool
    score: float = Field(ge=0.0, le=1.0)
    missing_variable: str
    rationale: str


class AlternativesEvaluationSchema(BaseModel):
    is_useful: bool
    score: float = Field(ge=0.0, le=1.0)
    rationale: str
